# Sentinel-1A — physically-based path tracing with RayMakie + Hikari + Lava (Vulkan/MoltenVK).
#
#   julia --project=. Sentinel/sentinel.jl     # renders BOTH looks:
#       sentinel_studio.png  — neutral studio environment (matches the Sketchfab viewer)
#       sentinel_space.png   — ESO Milky Way deep-space setting
#
# The glTF has NO textures and NO UVs — every one of its 19 materials is a flat PBR
# factor (baseColor + metallic + roughness). 15 of the 19 are authored metallic = 1.0
# (most at roughness ~ 0), i.e. polished COLORED METAL: the solar cells are blue chrome,
# the bus-bars red chrome, the foil gold, the structure silver/aluminium.
#
# So `to_hikari` honours the glTF metallic-roughness model faithfully:
#   - metallic >= 0.5  ->  a true Conductor whose normal-incidence reflectance (F0) equals
#     the authored base colour. (Hikari's Conductor ignores its `reflectance` field; the
#     tint must live in eta/k. We use pbrt-v4's reflectance->IOR inversion: eta = 1,
#     k = 2*sqrt(r)/sqrt(1-r) per channel, so F0 = r = base colour. A blue base => blue metal.)
#   - metallic <  0.5  ->  CoatedDiffuse (Fresnel-coated diffuse) for the few dielectric parts.
#
# A metal's shine IS what it reflects, so each look ships its own environment map (an area
# light): a bright neutral studio for the clean Sketchfab look, or the Milky Way + a soft
# Earthshine/Sun glow for the deep-space look. Path-traced on the GPU through Lava.
#
# Requires the macOS/MoltenVK port: Lava branch sd/nvidia-macos (5 SPIR-V categories) and the
# vendored Raycore with the multi-dispatch BVH refit.

include(joinpath(@__DIR__, "..", "common", "common.jl"))   # DEVICE, GLMakie, RayMakie, Hikari, FileIO
include(joinpath(@__DIR__, "bloom.jl"))                    # bloom! (Sketchfab-style soft glow)
using MeshIO, FileIO, LinearAlgebra, GeometryBasics, Serialization

const SENTINEL_DIR = get(ENV, "SENTINEL_DIR", "/Users/anshul/Downloads/sentinel_1a")
const MODEL       = joinpath(SENTINEL_DIR, "scene.glb")
const MODEL_CACHE = joinpath(@__DIR__, "..", "..", "model.jls")          # _rt/model.jls (fast reload)
const SKYMAP      = joinpath(@__DIR__, "..", "assets", "skymap.png")      # 6000x3000 equirect, ESO/S. Brunier

# ----------------------------------------------------------------------------- model
function load_model(; path = MODEL)
    if isfile(MODEL_CACHE)
        try
            @info "Deserializing cached model $MODEL_CACHE"
            return deserialize(MODEL_CACHE)
        catch e
            @warn "cache load failed ($e) — loading from $path"
        end
    end
    @info "Loading $path"
    m = FileIO.load(path; up = Vec3f(0, 1, 0))
    try; serialize(MODEL_CACHE, m); catch; end
    return m
end

function model_bounds(model)
    P = model.position
    mn = reduce((a, b) -> min.(a, b), P); mx = reduce((a, b) -> max.(a, b), P)
    return Point3f((mn .+ mx) ./ 2), Float32(norm(mx .- mn) / 2)
end

# identical camera for the beauty pass and the matte pass (so they composite pixel-perfectly)
function place_camera!(scene, center, radius; eye_dir, zoom, fov)
    cam = cameracontrols(scene)
    cam.eyeposition[] = Point3f(center .+ radius * zoom .* normalize(eye_dir))
    cam.lookat[] = center; cam.upvector[] = Vec3f(0, 1, 0); cam.fov[] = Float32(fov)
    Makie.update_cam!(scene, cam)
    return cam
end

# ----------------------------------------------------------------------------- materials
# pbrt-v4 reflectance -> conductor IOR: with eta = 1, normal-incidence reflectance is
# R = k^2 / (4 + k^2); solving R = r gives k = 2*sqrt(r)/sqrt(1-r). So F0 = base colour.
function metal_from_reflectance(c; roughness)
    r = clamp.(Float32.(c), 0f0, 0.9999f0)
    k = ntuple(i -> 2f0 * sqrt(r[i]) / sqrt(1f0 - r[i]), 3)
    return Hikari.Conductor(eta = (1f0, 1f0, 1f0), k = k,
                            roughness = Float32(roughness), remap_roughness = true)
end

# one glTF PBR material dict -> a true Hikari BSDF (honours metallic-roughness)
function to_hikari(v)
    d = get(v, "diffuse", Vec3f(1, 1, 1)); r = Float32(get(v, "roughness", 0.5))
    mm = get(v, "metallic", nothing); metal = mm === nothing ? 1.0f0 : Float32(mm)  # glTF default metallic = 1
    c = (Float32(d[1]), Float32(d[2]), Float32(d[3]))
    if metal >= 0.5f0
        cr, cg, cb = c
        if cb > 0.45f0 && cb >= cr && cb >= cg          # bright-blue solar cell → deepen to navy
            c = (cr, min(cg, 0.10f0), min(cb, 0.55f0))  # (Sketchfab renders both wings deep navy, not cyan)
        end
        return metal_from_reflectance(c; roughness = max(r, 0.02f0))   # blue/red/gold/silver chrome (low floor = crisp reflections)
    end
    return Hikari.CoatedDiffuse(reflectance = c, roughness = max(r, 0.06f0))  # dielectric parts
end

# split the loaded MetaMesh into one plain mesh per material (positions/normals shared)
function build_submeshes(model)
    mats = model.meta[:materials]; names = model.meta[:material_names]; views = model.views
    F = model.faces; P = model.position; N = model.normal
    groups = Dict{String, Vector{eltype(F)}}()
    for (i, vr) in enumerate(views)
        g = get!(() -> eltype(F)[], groups, names[i])
        @inbounds for fi in vr; push!(g, F[fi]); end
    end
    return [(to_hikari(get(mats, nm, Dict{String, Any}())), GeometryBasics.Mesh(P, fs; normal = N))
            for (nm, fs) in groups if !isempty(fs)]
end

# ----------------------------------------------------------------------------- environments
# add a soft (Gaussian) glow centred at latitude clat (0=zenith,1=nadir), longitude clong,
# wrapping in longitude; `col` is added (HDR, may exceed 1) so metals get bright reflections.
function add_glow!(img, clat, clong, σlat, σlong, col)
    H, W = size(img)
    @inbounds for j in 1:W
        u = (j - 1) / (W - 1); dl = abs(u - clong); dl = min(dl, 1 - dl)
        for i in 1:H
            t = (i - 1) / (H - 1)
            g = exp(-((t - clat)^2 / (2σlat^2) + dl^2 / (2σlong^2)))
            p = img[i, j]
            img[i, j] = RGBf(p.r + col[1] * g, p.g + col[2] * g, p.b + col[3] * g)
        end
    end
    return img
end

# bright neutral photographic studio: soft vertical gradient + a few elongated soft-boxes
# (the soft-boxes are what give polished metal its crisp streak highlights).
function make_studio_env(; W = 1536, H = 768)
    img = Matrix{RGBf}(undef, H, W)
    @inbounds for i in 1:H
        t = (i - 1) / (H - 1)                                  # 0 zenith .. 1 nadir
        base = 0.18f0 + 0.97f0 * (cos(t * Float32(π)) * 0.5f0 + 0.5f0)   # 1.15 top -> 0.18 bottom
        wr = 1.0f0 - 0.08f0 * t; wg = 0.985f0; wb = 0.92f0 + 0.10f0 * t  # warm top, cool floor
        for j in 1:W
            img[i, j] = RGBf(base * wr, base * wg, base * wb)
        end
    end
    add_glow!(img, 0.24, 0.30, 0.050, 0.11, (3.2, 3.2, 3.3))   # key soft-box
    add_glow!(img, 0.20, 0.72, 0.045, 0.09, (3.0, 3.0, 3.0))   # fill soft-box
    add_glow!(img, 0.33, 0.93, 0.040, 0.07, (2.1, 2.1, 2.3))   # rim
    return img
end

# deep-space: the ESO Milky Way panorama, lifted slightly, plus a broad cool Earthshine and a
# hot Sun glow so the now-correct metals have bright, coloured things to reflect.
function make_space_env(; lift = 1.15f0)
    sky = FileIO.load(SKYMAP)
    img = map(p -> RGBf(lift * Float32(p.r), lift * Float32(p.g), lift * Float32(p.b)), sky)
    add_glow!(img, 0.66, 0.40, 0.22, 0.30, (0.18, 0.34, 0.62))   # broad Earthshine (cool fill)
    add_glow!(img, 0.30, 0.84, 0.035, 0.045, (7.0, 6.6, 5.8))    # hot Sun (specular highlight)
    return img
end

# Sketchfab-match env: a STRUCTURED dark-space studio so the metals have something to REFLECT.
# A hemisphere gradient (bright zenith → dark nadir) gives curved metal a real reflection gradient;
# a small bright sun disc gives a sharp specular glint; a soft cool kicker fills the other side.
# Low overall level keeps the navy cells dark; high-albedo aluminium still reads bright. The visible
# background is composited to black by the matte, so this env is reflections/IBL only.
function make_match_env(; W = 1536, H = 768)
    img = Matrix{RGBf}(undef, H, W)
    @inbounds for i in 1:H
        t = (i - 1) / (H - 1)                                  # 0 zenith .. 1 nadir
        s = cos(t * Float32(π)) * 0.5f0 + 0.5f0                # 1 top .. 0 bottom
        base = 0.03f0 + 0.30f0 * s^1.4f0                       # ~0.33 zenith → 0.03 nadir (dark → navy cells)
        for j in 1:W
            img[i, j] = RGBf(base * 0.86f0, base * 0.93f0, base * 1.0f0)   # cool
        end
    end
    add_glow!(img, 0.18, 0.50, 0.030, 0.045, (13.0, 12.2, 11.0))  # bright sun (centred) → sharp symmetric glints
    add_glow!(img, 0.46, 0.50, 0.14,  0.55,  (0.45, 0.6, 0.9))    # broad cool kicker band (symmetric)
    return img
end

# ----------------------------------------------------------------------------- scene
function create_scene(model; mode::Symbol = :studio,
        resolution = (1500, 1125),
        eye_dir = Vec3f(0.8, 0.45, 1.0), zoom = 2.05f0, fov = 30.0)
    center, radius = model_bounds(model)
    lights = Makie.AbstractLight[]
    if mode === :studio
        push!(lights, EnvironmentLight(1.9f0, make_studio_env()))
        push!(lights, DirectionalLight(RGBf(1.0, 0.98, 0.95) * 3.0f0, Vec3f(-0.5, -0.55, -0.65)))  # key
        push!(lights, DirectionalLight(RGBf(0.85, 0.9, 1.0) * 0.8f0, Vec3f(0.6, 0.2, 0.4)))        # cool kicker
    elseif mode === :match
        # Sketchfab reproduction: DARK & high-contrast. Dim Milky Way IBL + one strong dominant
        # key so only the bright aluminium frames / gold dome clip to white (and bloom), while the
        # navy panels & bronze foil sit dark. Shadows stay deep (just one faint fill, no flat lift).
        push!(lights, EnvironmentLight(1.5f0, make_match_env()))                                    # structured studio env (reflections)
        push!(lights, DirectionalLight(RGBf(1.0, 0.97, 0.90) * 12.0f0, Vec3f(0.0, -0.62, -0.5)))    # CENTRED key → symmetric wings; blows aluminium/dome; env carries reflections
    else  # :space
        push!(lights, EnvironmentLight(1.7f0, make_space_env()))
        push!(lights, DirectionalLight(RGBf(1.0, 0.96, 0.88) * 6.0f0, Vec3f(-0.5, -0.35, -0.78)))  # Sun
        push!(lights, DirectionalLight(RGBf(0.4, 0.55, 1.0) * 0.7f0, Vec3f(0.25, 0.85, 0.35)))     # Earthshine fill
    end
    fig = Figure(size = resolution)
    ax = LScene(fig[1, 1]; show_axis = false, scenekw = (; backgroundcolor = RGBf(0, 0, 0), lights = lights))
    for (mat, sm) in build_submeshes(model)
        mesh!(ax, sm; material = mat)
    end
    place_camera!(ax.scene, center, radius; eye_dir = eye_dir, zoom = zoom, fov = fov)
    return fig, ax
end

# Object coverage matte (white diffuse, no env, ring of white directionals, BLACK background) so we
# can composite the env-lit beauty over pure black — the renderer otherwise shows the env as the
# background (miss-depth is finite when an infinite light is present, bypassing its own bg masking).
function render_matte(model; resolution, eye_dir, zoom, fov, samples = 12)
    center, radius = model_bounds(model)
    white = Hikari.CoatedDiffuse(reflectance = (1f0, 1f0, 1f0), roughness = 1f0)
    dirs = [Vec3f(-0.6, -0.5, -0.7), Vec3f(0.6, 0.5, 0.7), Vec3f(0.7, -0.4, 0.5), Vec3f(-0.7, 0.4, -0.5),
            Vec3f(0.0, -1.0, 0.2),  Vec3f(0.0, 1.0, -0.2), Vec3f(0.5, 0.2, -0.8), Vec3f(-0.5, -0.2, 0.8)]
    lights = Makie.AbstractLight[DirectionalLight(RGBf(1, 1, 1) * 2.2f0, d) for d in dirs]
    fig = Figure(size = resolution, backgroundcolor = RGBf(0, 0, 0))
    ax = LScene(fig[1, 1]; show_axis = false, scenekw = (; backgroundcolor = RGBf(0, 0, 0), lights = lights))
    for (_, sm) in build_submeshes(model)
        mesh!(ax, sm; material = white)
    end
    place_camera!(ax.scene, center, radius; eye_dir = eye_dir, zoom = zoom, fov = fov)
    integ = Hikari.VolPath(samples = samples, max_depth = 3)
    integ.sensor = Hikari.PixelSensor(iso = 120, whitebalance = 6500)
    img = Makie.colorbuffer(ax.scene; device = DEVICE, integrator = integ, update = false)
    H, W = size(img); a = Matrix{Float32}(undef, H, W)
    @inbounds for i in eachindex(img)
        c = img[i]; lum = 0.2126f0 * c.r + 0.7152f0 * c.g + 0.0722f0 * c.b
        a[i] = clamp((lum - 0.004f0) / 0.02f0, 0f0, 1f0)   # bg is exactly 0 → safe low threshold
    end
    return a
end

_default_out(mode) = joinpath(SENTINEL_DIR,
    mode === :studio ? "sentinel_studio.png" :
    mode === :match  ? "sentinel_match.png"  : "sentinel_space.png")

function render(model; mode::Symbol = :match, resolution = (1500, 1125),
        samples = 256, max_depth = 20, iso = nothing, whitebalance = nothing,
        eye_dir = nothing, zoom = nothing, fov = 30.0,
        bloom = (mode === :match), bloom_threshold = 0.60f0, bloom_intensity = 1.00f0,
        tonemap = (mode === :match ? nothing : :aces),   # Sketchfab uses NO tone curve (linear)
        output = nothing)
    eye_dir      = something(eye_dir,      mode === :match ? Vec3f(0.0, 0.22, 1.0) : Vec3f(0.8, 0.45, 1.0))  # match = head-on
    zoom         = something(zoom,         mode === :match ? 2.35f0 : 2.05f0)
    iso          = something(iso,          mode === :studio ? 100  : mode === :match ? 90 : 160)
    whitebalance = something(whitebalance, mode === :space  ? 6000 : 6500)
    output === nothing && (output = _default_out(mode))
    fig, ax = create_scene(model; mode = mode, resolution = resolution,
                           eye_dir = eye_dir, zoom = zoom, fov = fov)
    integ = Hikari.VolPath(samples = samples, max_depth = max_depth)
    integ.sensor = Hikari.PixelSensor(iso = iso, whitebalance = whitebalance)
    @info "[$mode] path tracing $(resolution[1])x$(resolution[2]) @ $samples spp (iso $iso, wb $whitebalance, tonemap $tonemap) -> $output"
    t = @elapsed beauty = Makie.colorbuffer(ax.scene; device = DEVICE, integrator = integ, tonemap = tonemap, update = false)
    img = RGBf.(beauty)   # mutable RGBf matrix for compositing / bloom
    if mode === :match     # composite the env-lit object over pure black (Sketchfab uses a black bg)
        @info "[$mode] rendering object matte for black-background composite"
        alpha = render_matte(model; resolution = resolution, eye_dir = eye_dir, zoom = zoom, fov = fov)
        @inbounds for i in eachindex(img)
            c = img[i]; img[i] = RGBf(c.r * alpha[i], c.g * alpha[i], c.b * alpha[i])
        end
    end
    bloom && bloom!(img; threshold = Float32(bloom_threshold), intensity = Float32(bloom_intensity))
    @info "[$mode] rendered in $(round(t; digits = 1)) s -> $output"
    FileIO.save(output, img)
    return img
end

if abspath(PROGRAM_FILE) == @__FILE__
    m = load_model()
    render(m; mode = :match,  samples = 256, resolution = (1600, 900))   # Sketchfab-matched hero (head-on, 16:9)
    render(m; mode = :studio, samples = 256)
    render(m; mode = :space,  samples = 256)
end
