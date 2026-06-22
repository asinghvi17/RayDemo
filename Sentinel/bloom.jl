# Self-contained bloom post-process (Sketchfab-style soft glow on bright metal).
# Pure image op — no GPU. Used both as a standalone tuner and included by sentinel.jl.
using FileIO, Colors
const RGBf = RGB{Float32}   # Makie defines this too (same type) — safe when included by sentinel.jl

# sRGB <-> linear
@inline _s2l(x) = x <= 0.04045f0 ? x / 12.92f0 : ((x + 0.055f0) / 1.055f0)^2.4f0
@inline _l2s(x) = (x <= 0.0031308f0 ? 12.92f0 * x : 1.055f0 * max(x, 0f0)^(1f0 / 2.4f0) - 0.055f0)

# average f×f blocks -> (H÷f)×(W÷f)
function _boxdown(A::Matrix{Float32}, f::Int)
    H, W = size(A); oh = H ÷ f; ow = W ÷ f; O = zeros(Float32, oh, ow)
    @inbounds for j in 1:ow, i in 1:oh
        s = 0f0
        for dj in 1:f, di in 1:f
            s += A[(i - 1) * f + di, (j - 1) * f + dj]
        end
        O[i, j] = s / (f * f)
    end
    return O
end

# separable Gaussian blur
function _gauss(A::Matrix{Float32}, σ::Float32)
    r = max(1, ceil(Int, 3σ))
    ks = Float32[exp(-(k^2) / (2σ^2)) for k in -r:r]; ks ./= sum(ks)
    H, W = size(A); T = similar(A); O = similar(A)
    @inbounds for j in 1:W, i in 1:H
        s = 0f0
        for k in -r:r; s += A[clamp(i + k, 1, H), j] * ks[k + r + 1]; end
        T[i, j] = s
    end
    @inbounds for j in 1:W, i in 1:H
        s = 0f0
        for k in -r:r; s += T[i, clamp(j + k, 1, W)] * ks[k + r + 1]; end
        O[i, j] = s
    end
    return O
end

# bilinear upsample S to size(acc) and add into acc
function _addup!(acc::Matrix{Float32}, S::Matrix{Float32})
    H, W = size(acc); oh, ow = size(S)
    @inbounds for j in 1:W, i in 1:H
        fy = (i - 0.5f0) * oh / H + 0.5f0; fx = (j - 0.5f0) * ow / W + 0.5f0
        y0 = clamp(floor(Int, fy), 1, oh); y1 = min(y0 + 1, oh)
        x0 = clamp(floor(Int, fx), 1, ow); x1 = min(x0 + 1, ow)
        ty = clamp(fy - y0, 0f0, 1f0); tx = clamp(fx - x0, 0f0, 1f0)
        v = (1 - ty) * ((1 - tx) * S[y0, x0] + tx * S[y0, x1]) +
                 ty  * ((1 - tx) * S[y1, x0] + tx * S[y1, x1])
        acc[i, j] += v
    end
end

"""
    bloom!(img; threshold, intensity, scales)

Add a multi-scale Gaussian bloom to an `RGBf` image in place. `threshold` is in
linear luminance (bright metal/highlights ~0.6+ clip and bloom). `intensity`
scales the added glow. `scales` = list of (downsample_factor, blur_sigma).
"""
function bloom!(img::Matrix{RGBf}; threshold::Float32 = 0.60f0, intensity::Float32 = 0.50f0,
        scales = ((4, 11f0), (8, 9f0)), knee::Float32 = 0.15f0)
    H, W = size(img)
    R = Array{Float32}(undef, H, W); G = similar(R); B = similar(R)
    @inbounds for i in eachindex(img)
        c = img[i]; R[i] = _s2l(c.r); G[i] = _s2l(c.g); B[i] = _s2l(c.b)
    end
    bR = zeros(Float32, H, W); bG = zeros(Float32, H, W); bB = zeros(Float32, H, W)
    @inbounds for i in eachindex(R)
        lum = 0.2126f0 * R[i] + 0.7152f0 * G[i] + 0.0722f0 * B[i]
        f = clamp((lum - threshold) / knee, 0f0, 1f0)
        bR[i] = R[i] * f; bG[i] = G[i] * f; bB[i] = B[i] * f
    end
    aR = zeros(Float32, H, W); aG = zeros(Float32, H, W); aB = zeros(Float32, H, W)
    for (ds, σ) in scales
        _addup!(aR, _gauss(_boxdown(bR, ds), Float32(σ)))
        _addup!(aG, _gauss(_boxdown(bG, ds), Float32(σ)))
        _addup!(aB, _gauss(_boxdown(bB, ds), Float32(σ)))
    end
    @inbounds for i in eachindex(R)
        img[i] = RGBf(clamp(_l2s(R[i] + intensity * aR[i]), 0f0, 1f0),
                      clamp(_l2s(G[i] + intensity * aG[i]), 0f0, 1f0),
                      clamp(_l2s(B[i] + intensity * aB[i]), 0f0, 1f0))
    end
    return img
end

# standalone tuner:  julia --project=. Sentinel/bloom.jl in.png out.png [threshold] [intensity]
if abspath(PROGRAM_FILE) == @__FILE__
    inp, outp = ARGS[1], ARGS[2]
    th = length(ARGS) >= 3 ? parse(Float32, ARGS[3]) : 0.60f0
    inten = length(ARGS) >= 4 ? parse(Float32, ARGS[4]) : 0.50f0
    img = RGBf.(FileIO.load(inp))
    bloom!(img; threshold = th, intensity = inten)
    FileIO.save(outp, img)
    println("bloom -> $outp (threshold=$th intensity=$inten)")
end
