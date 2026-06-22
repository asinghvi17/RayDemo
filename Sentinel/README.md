# Sentinel-1A on Apple Silicon — GPU path tracing (RayMakie/Hikari/Lava) + GLMakie raster

A worked example that renders the **Sentinel-1A** satellite glTF on an **Apple M1 Max** two ways:

1. **Physically-based GPU path tracing** — RayMakie + Hikari (spectral, pbrt-v4 port) + Lava (Vulkan compute) running on **MoltenVK/Metal**. ~160 s/frame, physically correct.
2. **Real-time rasterized preview** — pure GLMakie with per-material **matcaps**. ~2.5 s offscreen (real-time interactively).

This is the first time this stack has run on Apple Silicon; getting there required porting Lava's SPIR-V emitter to MoltenVK and fixing a Metal memory-coherency bug in Raycore's BVH refit (both documented below).

---

## The stack (sources & branches)

Julia **1.12**; `DEVICE = GPUSelect.Backend(:Lava)` → a `LavaBackend` (Vulkan; MoltenVK on macOS). See `../Project.toml [sources]`.

| Package | Upstream | rev | Modified? | Our fork |
|---|---|---|---|---|
| GPUSelect | `SimonDanisch/GPUSelect.jl` | `main` | no | — |
| Makie / GLMakie / RayMakie / ComputePipeline | `MakieOrg/Makie.jl` (subdirs) | `sd/lava` | no | — |
| Hikari | `JuliaGraphics/Hikari.jl` | `sd/vk-hw-accel` | no | — |
| MeshIO | `JuliaIO/MeshIO.jl` | `sd/gltf-glb` | no | — |
| **Lava** | `SimonDanisch/Lava.jl` | `sd/nvidia` → **`sd/nvidia-macos`** | **YES** (5 commits) | **`asinghvi17/Lava.jl` @ `sd/nvidia-macos`** |
| **Raycore** | `JuliaGeometry/Raycore.jl` | `master` (`5fd0a102`) → **macOS refit** | **YES** | **`asinghvi17/Raycore.jl` @ `macos-fixpoint-refit`** |

> **CRITICAL constraint:** stay on the Lava `sd/nvidia` line. RayDemo pins Hikari `sd/vk-hw-accel`, which hard-requires sd/nvidia-only symbols (`Lava.HWTLAS`, `HWAdaptedAccel`, `concurrent_dispatch_group`). The `sd/ci` / PR#3 branch can't even load Hikari. So **macOS support = patch `sd/nvidia` (→ `sd/nvidia-macos`), not switch branches.**

---

## Our fixes (what made it run on MoltenVK)

### 1. Lava `sd/nvidia-macos` — 5 SPIR-V → Metal categories
All five failures stem from `PhysicalStorageBuffer64` (buffer-device-address) codegen emitting MSL that SPIRV-Cross/MoltenVK reject. Every MoltenVK path is gated on a new `Lava.is_moltenvk()` so NVIDIA codegen is byte-for-byte unchanged. Full writeup: **[`METAL_CATEGORIES.md`](METAL_CATEGORIES.md)**.

- **Phase 1 — device feature negotiation:** negotiate `ENABLED_OPTIONAL_FEATURES` / `has_device_feature` instead of hardcoding (MoltenVK lacks fp64/geometry/wide-lines); `supports_fp64 = has_device_feature(:shader_float_64)`.
- **Cat 1** — don't re-`reinterpret` a PSB pointer that `resolve_struct_field_load!` already drilled via `OpAccessChain`.
- **Cat 2** — atomic on a raw PSB/BDA pointer → route through a 1-member Block struct + `OpAccessChain` (lvalue) in `emit_psb_atomic_lvalue_ptr!`.
- **Cat 3a/3b** — fold identity `OpConvertPtrToU(OpConvertUToPtr(int))→int`, and lower `OpConvertPtrToU(OpAccessChain)` to an integer address recomputed from the chain root (never `ConvertPtrToU` an access-chain).
- **Cat 4** — Float32-only `ComplexF32` `/`, `inv` (Base widened Float32→Float64 → GPUCompiler `unsupported use of double value`).

### 2. Raycore — multi-dispatch (fixpoint) BVH refit
Metal/MoltenVK gives cross-threadgroup **device-memory coherency only at dispatch boundaries**, never within a single dispatch (proven: even seq_cst device fences don't help; `vulkanMemoryModelAvailabilityVisibilityChains = FALSE`). Raycore's single-dispatch atomic bottom-up AABB refit therefore reads **stale child AABBs** → degenerate parent boxes → pruned subtrees → blocky **missing geometry** on flat panels.

**Fix:** run the refit to a fixpoint. `build_blas` / `build_tlas` / `refit_tlas!` now reset the visitor flags and re-dispatch the refit kernel `n_refit_passes = 2*ceil(log2 n) + 8` times (in `src/instanced-bvh.jl`, 3 sites). Verified solid from 6 angles via flat-white silhouette renders.

---

## Files in this example

| File | What |
|---|---|
| `sentinel.jl` | Path tracer. Modes `:match` (Sketchfab-matched, head-on), `:studio`, `:space`. `julia --project=. Sentinel/sentinel.jl` |
| `bloom.jl` | Self-contained multi-scale Gaussian bloom post-pass (works on any image). |
| `raster.jl` | **GLMakie** matcap rasterizer — fast preview. `julia --project=. Sentinel/raster.jl` |
| `METAL_CATEGORIES.md` | Full writeup of the 5 Lava SPIR-V→Metal fixes + the Raycore refit. |
| `../common/common.jl` | `DEVICE = GPUSelect.Backend(:Lava)`. |
| `../assets/skymap.png` | ESO/S. Brunier Milky Way panorama (environment light / IBL background). |

The model path is read from `ENV["SENTINEL_DIR"]` (default `/Users/anshul/Downloads/sentinel_1a`); outputs land there too.

---

## Reproduce

1. **Model.** Download Sentinel-1A from [Sketchfab (Absideon)](https://sketchfab.com/3d-models/sentinel-1a-0b44ab92dc714999a3d0df2f4c572895) — **CC-BY-4.0** — and pack it to a self-contained `scene.glb` (drop the buffer `uri` so it references the GLB BIN chunk → MeshIO's fast binary path). Point `ENV["SENTINEL_DIR"]` at the folder.
2. **Environment.** Instantiate from the pinned `Manifest.toml` (don't `resolve` — the Makie monorepo tip drifts). For the modified Lava/Raycore, `Pkg.develop(path=…; preserve=Pkg.PRESERVE_ALL)`.
3. **Run.** `julia --project=. Sentinel/sentinel.jl` (path trace) or `Sentinel/raster.jl` (GLMakie).

**Gotcha:** after editing Lava/Raycore, `rm -rf ~/.julia/compiled/v1.12/{Lava,Raycore}` then `Pkg.precompile` — Julia serves a stale cache otherwise (Revise is unreliable for `module.jl`).

---

## Notable findings (the "match Sketchfab" story)

- The glTF has **no textures and no UVs** — all 19 materials are flat PBR factors (baseColor + metallic + roughness). 15/19 are `metallic=1` (polished colored metal).
- Metals → true Hikari `Conductor` with **F0 = baseColor**. Hikari's `Conductor.reflectance` field is **ignored**; the tint must live in `eta`/`k` (pbrt's `eta=1, k=2√r/√(1−r)`). RGB `eta`/`k` go through `uplift_rgb_unbounded`, so `k>1` is fine.
- The Sketchfab viewer's look = its **default Milky Way IBL + black background + bloom + a bright key + linear tone mapping (NOT ACES)**. ACES was silently compressing all our contrast.
- GLMakie's rasterizer has **no IBL** — `raster.jl` fakes reflections with per-material **matcaps** (view-space, baked F0 + gradient + Fresnel).

---

## Next: a real-time PBR pipeline in Makie (v0.25)

GLMakie is a plotting backend (Blinn-Phong + matcaps + SSAO + FXAA), not a PBR engine — no IBL, SSR, or shadows. But the **GLMakie render-pipeline rework** makes building one feasible:

- **PR [#4689 "Rework postprocessor handling in GLMakie"](https://github.com/MakieOrg/Makie.jl/pull/4689)** (`ff/render_pipeline_master`) + **[#5436 "Allow GLMakie render objects to render in different ways"](https://github.com/MakieOrg/Makie.jl/pull/5436)**, landing in the breaking **v0.25** line.
- They turn the hardcoded `render_frame()` into a composable **render graph** (Blender/Unreal-blueprint-style stages with connectable buffers): **custom stages**, **multi-attachment framebuffers** (a G-buffer — `fragment_output` derived from the pipeline), and `BufferFormat` with HDR/mipmap/multisample requirements.

**Plan:** implement deferred **GGX PBR + split-sum IBL + SSAO + SSR + bloom + tonemap** as custom Makie render stages → a Sketchfab-class real-time viewer natively in Makie. Requires moving onto a v0.25-line Makie (our `sd/lava` predates it) — a fresh top-level env.

---

## Attribution

- **Model:** *Sentinel-1A* by **Absideon** (Sketchfab) — **CC-BY-4.0**. Credit if shared/reused.
- **Skymap:** ESO / S. Brunier, Milky Way panorama.
