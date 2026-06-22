# Fast GLMakie *rasterized* render of Sentinel-1A using per-material MATCAPS.
#
#   julia --project=. Sentinel/raster.jl     ->  sentinel_raster.png   (milliseconds, not minutes)
#
# The glTF has NO textures/UVs, so there is nothing to "unwrap": each of the 19 materials is a
# flat PBR factor. GLMakie's rasterizer can't do real image-based env reflections, but it supports
# MATCAPS (a sphere image sampled by the surface normal — Makie docs: `mesh(..., matcap=img,
# shading=NoShading)`). We bake a glossy-metal matcap per material (F0 tint + reflection gradient +
# specular + Fresnel rim), so the rasterizer fakes the shiny/reflective look in real time.

include(joinpath(@__DIR__, "..", "common", "common.jl"))   # GLMakie, RayMakie, Hikari, FileIO
using FileIO, GeometryBasics, LinearAlgebra, Serialization, Colors

const SENTINEL_DIR = get(ENV, "SENTINEL_DIR", "/Users/anshul/Downloads/sentinel_1a")
const MODEL_CACHE  = joinpath(@__DIR__, "..", "..", "model.jls")

load_model() = deserialize(MODEL_CACHE)

function model_bounds(model)
    P = model.position
    mn = reduce((a, b) -> min.(a, b), P); mx = reduce((a, b) -> max.(a, b), P)
    return Point3f((mn .+ mx) ./ 2), Float32(norm(mx .- mn) / 2)
end

# split into one compact (non-indexed) GB.Mesh per material (avoids re-uploading the full 1.6M
# vertex buffer 19×: each face gets its own 3 verts → ~5.7M verts total, split across materials).
function raster_submeshes(model)
    mats = model.meta[:materials]; names = model.meta[:material_names]; views = model.views
    F = model.faces; P = model.position; N = model.normal
    groups = Dict{String, Vector{eltype(F)}}()
    for (i, vr) in enumerate(views)
        g = get!(() -> eltype(F)[], groups, names[i])
        @inbounds for fi in vr; push!(g, F[fi]); end
    end
    out = Tuple{String, Dict{String, Any}, GeometryBasics.Mesh}[]
    for (nm, fs) in groups
        isempty(fs) && continue
        verts = Point3f[]; norms = Vec3f[]; faces = GLTriangleFace[]
        @inbounds for f in fs
            pts = P[f]; ns = N[f]                       # 3 points / 3 normals of this triangle
            k = length(verts)
            push!(verts, pts[1], pts[2], pts[3])
            push!(norms, ns[1], ns[2], ns[3])
            push!(faces, GLTriangleFace(k + 1, k + 2, k + 3))
        end
        push!(out, (nm, get(mats, nm, Dict{String, Any}()), GeometryBasics.Mesh(verts, faces; normal = norms)))
    end
    return out
end

# procedurally bake a glossy-metal matcap: a sphere whose view-space normal (nx,ny,nz=√(1-r²))
# maps to F0-tinted reflection (vertical gradient) + a sharp specular + a bright Fresnel rim.
function make_matcap(F0; sz = 512, shininess = 90f0, rim = 0.75f0)
    cr, cg, cb = clamp.(Float32.(F0), 0f0, 0.999f0)
    img = Matrix{RGBf}(undef, sz, sz)
    lx, ly = -0.38f0, 0.52f0; lz = sqrt(max(0f0, 1 - lx^2 - ly^2))   # key light, upper-left
    @inbounds for j in 1:sz, i in 1:sz
        x = (j - 0.5f0) / sz * 2 - 1; y = 1 - (i - 0.5f0) / sz * 2
        r2 = x * x + y * y
        if r2 > 1f0
            img[i, j] = RGBf(0, 0, 0); continue
        end
        nz = sqrt(1 - r2); nx = x; ny = y
        env  = 0.28f0 + 0.72f0 * clamp(ny * 0.5f0 + 0.5f0, 0f0, 1f0)^1.2f0   # reflected env: dark floor → bright sky
        ndl  = clamp(nx * lx + ny * ly + nz * lz, 0f0, 1f0)
        spec = ndl^shininess                                                # sharp specular highlight
        fres = (1f0 - nz)^4f0                                               # grazing → white (Fresnel)
        tr = cr + (1f0 - cr) * fres; tg = cg + (1f0 - cg) * fres; tb = cb + (1f0 - cb) * fres
        img[i, j] = RGBf(clamp(env * tr + spec, 0f0, 1f0),
                         clamp(env * tg + spec, 0f0, 1f0),
                         clamp(env * tb + spec, 0f0, 1f0))
    end
    return img
end

function to_matcap(md)
    d = get(md, "diffuse", Vec3f(1, 1, 1))
    cr, cg, cb = Float32(d[1]), Float32(d[2]), Float32(d[3])
    if cb > 0.45f0 && cb >= cr && cb >= cg          # deepen bright-blue cells to navy
        cg = min(cg, 0.10f0); cb = min(cb, 0.55f0)
    end
    return make_matcap((cr, cg, cb))
end

function render_raster(model; resolution = (1600, 900), eye_dir = Vec3f(0, 0.22, 1), zoom = 2.35f0,
        fov = 30.0, output = joinpath(SENTINEL_DIR, "sentinel_raster.png"))
    GLMakie.activate!()
    center, radius = model_bounds(model)
    fig = Figure(size = resolution, backgroundcolor = RGBf(0, 0, 0))
    ax = LScene(fig[1, 1]; show_axis = false, scenekw = (; backgroundcolor = RGBf(0, 0, 0), clear = true))
    ax.scene.backgroundcolor[] = RGBf(0, 0, 0)        # force black (LScene ignores scenekw bg in colorbuffer)
    for (_, md, m) in raster_submeshes(model)
        mesh!(ax, m; matcap = to_matcap(md), shading = NoShading)
    end
    cam = Makie.cameracontrols(ax.scene)
    cam.eyeposition[] = Point3f(center .+ radius * zoom .* normalize(eye_dir))
    cam.lookat[] = center; cam.upvector[] = Vec3f(0, 1, 0); cam.fov[] = Float32(fov)
    Makie.update_cam!(ax.scene, cam)
    t = @elapsed img = Makie.colorbuffer(ax.scene)
    @info "GLMakie rasterized $(resolution[1])x$(resolution[2]) in $(round(t; digits = 2)) s → $output"
    FileIO.save(output, img)
    return img
end

if abspath(PROGRAM_FILE) == @__FILE__
    render_raster(load_model())
end
