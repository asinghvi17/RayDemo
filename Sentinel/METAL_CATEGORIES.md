# Lava sd/nvidia → MoltenVK (Apple M1 Max): SPIR-V→Metal failure categories

Branch: `sd/nvidia-macos` (off `sd/nvidia` tip 53f8705). Repro: Diffuse sphere via
RayMakie `colorbuffer(...; device=DEVICE, integrator=VolPath)` in the RayDemo env.
Device init + trivial kernels already work (Phase-1 feature-negotiation fixes).
The render advances kernel-by-kernel; each fix unmasks the next failing kernel.

SPIR-V dumped via `ENV["LAVA_SPIRV_DUMP_DIR"]`; disassembled with `spirv-dis`,
validated with `spirv-val` (both from SPIRV_Tools_jll). MoltenVK 1.4.1, MSL 3.2.

All failures are `VK_ERROR_INITIALIZATION_FAILED: Shader library compile failed`
during `vkCreateComputePipelines` — i.e. MoltenVK's SPIRV-Cross→MSL translation
emits invalid Metal Shading Language. Three root-cause categories found so far.

---

## Category 1 — redundant PSB pointer ptr→int→ptr round-trip on a drilled scalar
### (struct-field load) [ROOT CAUSE FOUND, FIXED, VERIFIED]

**Symptom (MSL):**
```
program_source:63: error: reinterpret_cast from 'float' to 'ulong' is not allowed
  device float* _232 = reinterpret_cast<device float*>(reinterpret_cast<ulong>(_201->_m0[0u][0u][0u][0u]));
```
**Failing kernel:** `AcceleratedKernels.gpu__mapreduce_block!` over
`LavaDeviceArray{Raycore.Triangle{Hikari.TriangleMeta}}` → `Raycore.Bounds3`
(`world_bound`/`union`). Call path: `Raycore.build_blas` → BVH scene-AABB
`mapreduce` → AK block reduce. (This is the lead's seed A.)

**SPIR-V pattern:**
```
%229 = OpAccessChain %_ptr_PhysicalStorageBuffer_float %201 0 0 0 0 0  ; already a typed float ptr
%231 = OpConvertPtrToU %ulong %229                                     ; ptr -> int  (redundant)
%232 = OpConvertUToPtr %_ptr_PhysicalStorageBuffer_float %231          ; int -> ptr
%230 = OpLoad %float %232
```
The `OpAccessChain` already produced a correctly-typed `float*`. The subsequent
`OpConvertPtrToU` of that access-chain pointer is what SPIRV-Cross renders as
`reinterpret_cast<ulong>(<dereferenced float lvalue>)` → illegal.

**Root-cause emitter location:** `src/compiler/spirv/emit.jl`, the load path.
`resolve_struct_field_load!` drills the struct pointer to the exact scalar via
`OpAccessChain` and returns it (`did_drill_for_load = true`). But the PSB branch
then re-checks `pointee_ty_ld` against the **original (struct)** pointer's pointee,
sees a "mismatch" (float vs struct), and calls `emit_psb_ptr_reinterpret!` on the
already-correct drilled pointer — emitting the spurious round-trip. (The same guard
already exists for the Workgroup/Function path via `!did_drill_for_load`.)

**Fix:** guard that reinterpret with `!did_drill_for_load`. (Done — emit.jl ~L1210.)
Vendor-neutral correctness fix (no device gate): emitting `OpLoad %float %229`
directly is valid on all vendors; the round-trip was only ever an NVIDIA-OpBitcast
workaround over-applied to a case that didn't need it.

---

## Category 2 — atomic on a raw PhysicalStorageBuffer (BDA) pointer
### [ROOT CAUSE FOUND, FIXED, SPIR-V-VALIDATED]

**Symptom (MSL):**
```
error: cannot take the address of an rvalue of type 'device uint *'
  uint _163 = atomic_fetch_add_explicit((device atomic_uint*)&(reinterpret_cast<device uint*>(
              reinterpret_cast<ulong>(*_91) + (_157 << 2ul))), 1u, memory_order_relaxed);
```
**Failing kernel:** `Raycore.gpu_refit_aabbs_kernel!` over `LavaDeviceArray{BVHNode2}`,
`LavaDeviceArray{UInt32}` — the bottom-up parallel BVH AABB refit uses an
`atomicrmw add` on a per-internal-node `UInt32` counter (`update_flags`). Call path:
`Raycore.build_blas` → `refit_aabbs_kernel!`. (Lead's atomic seed.)

**SPIR-V pattern:** the atomic's pointer operand is built purely from BDA address
arithmetic and ends in `OpConvertUToPtr` (an rvalue pointer):
```
%160 = OpConvertPtrToU %ulong %97      ; loaded device base ptr -> int
%161 = OpIAdd %ulong %160 %159         ; + (idx<<2)
%167 = OpConvertUToPtr %_ptr_PhysicalStorageBuffer_uint %161
%163 = OpAtomicIAdd %uint %167 ...
```
SPIRV-Cross renders `OpAtomic*` on such a pointer as
`(device atomic_uint*)&(reinterpret_cast<device uint*>(addr))` and `&` on a
`reinterpret_cast` **rvalue** is illegal MSL. Confirmed structural, not a
round-trip artifact: hand-collapsing to a single `OpConvertUToPtr`, and routing
through a Function-storage variable, both still produce the `&(rvalue)`.

**Fix (MoltenVK-only, validated at SPIR-V level):** route the atomic pointer
through a one-member `Block` struct `{T}` + `OpAccessChain` member 0:
```
%162 = OpConvertUToPtr %_ptr_PSB_struct{uint} %161
%167 = OpAccessChain %_ptr_PSB_uint %162 %uint_0   ; addressable lvalue
%163 = OpAtomicIAdd %uint %167 ...
```
SPIRV-Cross then treats the member as an lvalue and emits a plain
`reinterpret_cast<device atomic_uint*>` with no `&`. Hand-built variant →
`vkCreateComputePipelines` returns OK on MoltenVK. (Done — new helper
`emit_psb_atomic_lvalue_ptr!`, wired into `emit_atomicrmw!` and the cmpxchg path,
gated on `is_moltenvk()` so NVIDIA/AMD keep the raw pointer. The wrapper struct
carries `Block` + member `Offset 0` decorations so it passes `spirv-val`.)
MoltenVK detection: new `Lava.is_moltenvk()` (set in `init_vulkan!` from the
`VK_KHR_portability_subset` extension; precompile-safe like `has_device_feature`).

---

## Category 3 — redundant PSB ptr→int→ptr round-trips in BVH traversal/trace
### [ROOT CAUSE FOUND — same family as Cat 1; general fix in progress]

**Symptom (MSL), several variants, all "reinterpret_cast from <value> to 'ulong'
is not allowed":**
```
ulong _1021 = reinterpret_cast<ulong>(reinterpret_cast<device _418*>( ... )) + (-144);
device _399* _1078 = reinterpret_cast<device _399*>(reinterpret_cast<ulong>(_496->_m1) + ...);
ulong _1095 = reinterpret_cast<ulong>((*(reinterpret_cast<device spvUnsafeArray<spvUnsafeArray<uint,2>,3>*>(_1091)))[1u]);
```
**Failing kernel:** the path-trace / BVH-traversal kernel (e.g.
`gpu_detect_camera_medium_kernel!` and the closest-hit trace kernels) reading
`BVHNode2` child pointers/indices (`_m1`) and nested index arrays.

**SPIR-V pattern (the dominant one):** a pure ptr↔int **round-trip** around an
intermediate `OpConvertUToPtr`, then more address arithmetic:
```
%1016 = OpIAdd %ulong ...                                  ; byte address (integer)
%1017 = OpConvertUToPtr %_ptr_PSB_struct_418 %1016         ; int -> ptr
%1019 = OpConvertPtrToU %ulong %1017                        ; ptr -> int   <-- redundant
%1021 = OpIAdd %ulong %1019 %ulong(-144)                    ; + offset
%1022 = OpConvertUToPtr %_ptr_PSB_struct_418 %1021
```
`OpConvertPtrToU(OpConvertUToPtr(x)) ≡ x` is an exact identity, but the emitter
materializes the intermediate pointer `%1017` and converts it back. SPIRV-Cross
inlines `%1017` into the `reinterpret_cast<ulong>(...)` at `%1019` and MoltenVK
rejects casting that pointer-expression-reduced-to-value to `ulong`. Same
mistranslation family as Cat 1, but appearing wherever the GEP-lowering /
PSB-access-chain code emits an intermediate pointer that is immediately re-cast to
an integer (`gep(gep(...))`, 1-based-index `base + (-1)`, struct base + member
offset, nested-array element addressing).

**Root-cause emitter location:** `src/compiler/spirv/emit.jl` GEP/PSB address
lowering (`emit_gep!`, `emit_psb_ptr_reinterpret!`, the PSB access-chain folding
that produces `base ± offset` chains). The intermediate `OpConvertUToPtr` is never
needed when its only consumer is another `OpConvertPtrToU`.

### Cat 3a — pure round-trip [FIXED + CONFIRMED]
`fold_convert_roundtrips!(mod)` (module.jl, run unconditionally in `serialize`)
folds the exact identity `OpConvertPtrToU(OpConvertUToPtr(int)) → int` over the
function instruction stream: drop the `OpConvertPtrToU`, replace its uses with the
original integer. Confirmed working — after the fold, the `_1021` /
`reinterpret_cast<ulong>(reinterpret_cast<device _418*>(...))` errors are gone
(prototype on the disassembly: 14 round-trips folded, passes `spirv-val`; the
in-emitter pass produces the same post-fold MSL). spirv-opt does NOT do this fold.
Vendor-neutral (no gate). This was necessary but **not sufficient** for the trace
kernel — it exposes Cat 3b underneath.

### Cat 3b — OpConvertPtrToU of a *loaded* / *access-chained* PSB pointer [WALL]
After 3a, the path-trace / BVH-traversal kernel still fails with:
```
error: reinterpret_cast from 'uint' to 'ulong' is not allowed
  ... reinterpret_cast<ulong>(_217->_m1) ...          // load of a pointer-typed struct member
  device _11* _976 = reinterpret_cast<device _11*>(reinterpret_cast<ulong>(_220->_m1) + ...);
error: reinterpret_cast from 'device spvUnsafeArray<uint,2>' to 'ulong' is not allowed
  ulong _989 = reinterpret_cast<ulong>((*(reinterpret_cast<device spvUnsafeArray<spvUnsafeArray<uint,2>,3>*>(_986)))[1u]);
```
These are `OpConvertPtrToU` whose operand is **not** an `OpConvertUToPtr` (so 3a
can't fold them), but an `OpLoad` of a pointer-typed PSB struct member (`_m1`, a
`device T*` child pointer in `BVHNode2`-like structs) or an `OpAccessChain` into a
nested index array (`uint[2][3]`). SPIRV-Cross materializes the operand as the
*dereferenced value* (`_217->_m1` typed `uint`, or the `uint[2]` array element) and
then `reinterpret_cast<ulong>` of that value is illegal MSL.

**Why it's a wall (not a quick fix):** the legal MSL form needs the *address* of
that member/element, computed arithmetically (`base_int + member_offset`), rather
than `OpConvertPtrToU(load/access-chain)`. That requires the emitter, at every
`OpConvertPtrToU` site whose operand is a loaded/access-chained PSB pointer, to
recover the underlying byte address and emit integer arithmetic instead — a
broad, invasive change to the GEP/PSB-access-chain lowering, touching the core of
how Lava chases nested BDA pointers through the BVH. It is the inverse of the Cat-2
problem (Cat 2 needed an access-chain to make a pointer addressable; Cat 3b needs
to *avoid* the access-chain/load form so the address-cast is legal). It also can't
be prototyped against the exact kernel yet: the **RT/trace SPIR-V is not written by
`dump_spirv_to_disk`** (the `raytracing.jl` serialize path bypasses
`LAVA_SPIRV_DUMP_DIR`), so only the MSL diagnostic is available, not the SPIR-V.

**Refined findings (after extracting the exact kernel + 4 prototype attempts):**
- The failing SPIR-V is `_rt/tmp_kernels/kernel_29_main.spv` (Lava's own debug
  dump at compilation.jl:653 — PRE-`serialize`, so PRE-fold; apply the same fold to
  reproduce the post-`serialize` bytes). After the Cat-3a fold, exactly **8**
  `OpConvertPtrToU` remain; **4** of them produce the 4 surviving MSL errors.
- The kernel's LLVM IR has **no `ptrtoint`** — every `OpConvertPtrToU` here is
  emitted by Lava's own GEP/PSB address lowering (`emit_gep!` →
  `emit_psb_byte_offset_with_user_type!`), which takes an `OpAccessChain` to a PSB
  struct member / nested-array element and converts it to an integer to do
  `base ± offset` arithmetic. SPIRV-Cross renders `OpConvertPtrToU(OpAccessChain(p, m))`
  as `reinterpret_cast<ulong>(p->_m)` — the member **value**, not `&p->_m` — and
  since the member is a 4-byte `uint` (struct offsets 0/4/8…), `uint → ulong` via
  `reinterpret_cast` is illegal. Equivalently: a `uint` index/reference field of the
  BVH node struct is being used as a device-pointer base (the seed-A/B prediction).
- Prototypes tried, all on kernel_29 vs MoltenVK: (a) Cat-3a round-trip fold —
  removes 14, leaves 4. (b) rewrite the 4 access-chain `OpConvertPtrToU` →
  `OpIAdd(base_int, member_offset)` — made it **worse, 4→8** (the access-chain
  pointers are *also* consumed as load bases; naively integerizing the address use
  breaks the SPIRV-Cross handling of the other consumers). (c) `OpCopyObject` on the
  pointer before the cast — no change (SPIRV-Cross sees through it). (d) Function-var
  store/load round-trip (Cat-2-style) — SPIRV-Cross optimizes it away.

**FIXED** (commit 840ad6f) — new peephole `lower_psb_addr_casts_moltenvk!`
(module.jl), run in `serialize()` before `fold_convert_roundtrips!`, gated on
`is_moltenvk()`. For each `OpConvertPtrToU` whose operand is an `OpAccessChain`,
recompute the byte address from the chain's *root* via a recursive `addr_int`:
- `addr_int(OpConvertUToPtr(int)) = int`
- `addr_int(OpAccessChain(base, idx...)) = addr_int(base) + Σ(member Offset |
  idx×ArrayStride)` from the module's `Offset`/`ArrayStride` decorations, recursing
  on `base`, **never** `OpConvertPtrToU`-ing an access chain.
- `addr_int(genuine pointer: arg / OpVariable / OpLoad / OpPhi) =
  OpConvertPtrToU(ptr)` — legal on all backends (`reinterpret_cast<ulong>` of a real
  pointer).
Then replace **only** the `OpConvertPtrToU` with `OpCopyObject` of the integer; the
typed `OpAccessChain` and its load/store consumers are left intact (the separation
that prototype (b) broke). A comprehensive id→result-type map over
`global_vars`+`functions` resolves each base's pointee type. The earlier (b)
regression was a red herring caused by an off-by-one in the binary parser (the
result id was mis-read as the first operand); with correct operand indexing the
recursion lowers all 8 access-chain casts and kernel_29 compiles. spirv-val clean,
MoltenVK `vkCreateComputePipelines` OK, sphere renders.

---

## Category 4 — ComplexF32 division/inverse widen to Float64 [FIXED]
**Symptom:** GPUCompiler `InvalidIRError: ... unsupported use of double value`,
call chain `Base ./complex.jl:377 (/)`, `:474 (inv)`. **Failing kernel:**
`Hikari.vp_shade_material_kernel!` specialized for `Conductor` (and `CoatedDiffuse`)
— the Fresnel/conductor BRDF does complex arithmetic. **Root cause:** `Base.:/(
::Complex{<:Union{Float16,Float32}}, ...)` and `Base.inv` `widen(z)` Float32→Float64
internally; with `shader_float_64` disabled on MoltenVK (the correct Cat-1
negotiation), the resulting `double` ops can't be lowered. **Fix** (commit c12b5e7,
`src/device/math.jl`): `@lava_device_override` Float32-only `Base.:/(ComplexF32,
ComplexF32)`, `Base.inv(ComplexF32)`, and ComplexF32/Float32 & Float32/ComplexF32
divisions (same Float32 tradeoff as the existing `abs(ComplexF32)` override).
Conductor + CoatedDiffuse spheres now render.

---

## Status summary — ALL CATEGORIES FIXED
- **Cat 1** — FIXED + verified. Vendor-neutral. (emit.jl load path, `!did_drill_for_load`.)
- **Cat 2** — FIXED + verified. MoltenVK-gated. (`emit_psb_atomic_lvalue_ptr!`, `is_moltenvk()`.)
- **Cat 3a** — FIXED + verified. Vendor-neutral. (`fold_convert_roundtrips!`.)
- **Cat 3b** — FIXED + verified. MoltenVK-gated. (`lower_psb_addr_casts_moltenvk!`.)
- **Cat 4** — FIXED + verified. Device override. (ComplexF32 `/`/`inv` in `device/math.jl`.)

Renders confirmed on Apple M1 Max / MoltenVK 1.4.1:
- Diffuse sphere → `/Users/anshul/Downloads/sentinel_1a/rt_sphere.png`
- Conductor sphere → `rt_conductor.png`
- CoatedDiffuse sphere → `rt_coated.png`
- Full `scene.glb` → `rt_scene.png` (Sentinel pipeline, 19 PBR materials)

All edits confined to `_rt/Lava.jl`. Commits on `sd/nvidia-macos`: `eb1b512`
(Phase 1 feature negotiation), `b119801` (Cat 1/2/3a), `840ad6f` (Cat 3b),
`c12b5e7` (Cat 4).

The full `scene.glb` render COMPILES and traces (every kernel incl. seed B's
multi-material dedup compiles clean — no MSL error surfaced), AND is now geometrically
CORRECT after the Category 5 fix below (a flat-white silhouette is solid from all
angles). NOTE: "compiles" ≠ "correct" — the scene compiled long before it was correct;
Cat 5 was the difference, and it is the true finish line.

## Category 5 — BVH refit: cross-threadgroup data race on Metal [CORRECTNESS, NOT codegen]
**Symptom:** the scene compiles and renders, but a flat-white plain-mesh silhouette
shows structured BLOCKY BLACK VOIDS across the large flat panels (solar wings, SAR
antenna) while the dense body is solid (`silh_hero.png`). Deterministic, structured,
not noise/material/culling → whole BVH subtrees pruned (triangles never tested).

**Root cause (proven):** Raycore's `refit_aabbs_kernel!` (instanced-bvh-kernels.jl)
is a single-dispatch persistent-thread bottom-up AABB refit: each thread does
`@atomic update_flags[parent] += 1`; the "last child" (counter→2) reads BOTH
children's AABBs (non-atomic `BVHNode2` loads) and writes the parent's AABB
(non-atomic store), then walks up. The ONLY cross-workgroup sync is that atomic.
On Metal this is a data race: the atomic COUNTER is globally coherent (Apple GPU
HW special case — confirmed: `degenerate_internal=0`, every node visited), but the
48-byte `BVHNode2` AABB written by a thread in ANOTHER threadgroup is NOT made
visible mid-dispatch. Metal provides no cross-threadgroup *bulk* device-memory
coherency within one dispatch (only at dispatch boundaries). So parents are built
from STALE child boxes → degenerate/partial parent AABBs → traversal prunes those
subtrees → missing flat-panel triangles.

**Proven NOT fixable via atomics/fences in Lava** (exhausted): emitter now emits the
strongest Metal ordering — verified in the MoltenVK-generated MSL:
`atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device)`
bracketing `atomic_fetch_add_explicit(...)`, with the atomic at SPIR-V `Scope.Device`
+ `MakeAvailable|MakeVisible`. Still stale across threadgroups. (Reproduced with a
fast multi-level tree-refit microbenchmark: correct ≤1 workgroup, wrong across
workgroups, coherent-but-stale subranges.) Lava's inter-dispatch read-after-write
IS correct on MoltenVK (2-dispatch dependent test: 0 mismatches) — so the fix must
use dispatch granularity.

**Emitter changes made (kept; correct + harmless, MoltenVK-gated, but NOT sufficient
alone):** `init_vulkan!` enables `vulkan_memory_model_device_scope` where supported;
`emit_atomicrmw!`/cmpxchg use `Scope.Device` + bracketing Device-scope
`OpMemoryBarrier` on MoltenVK (`atomic_device_scope()`, `emit_device_memory_barrier!`,
`VK_MEMORY_MODEL_DEVICE_SCOPE`, `Cap.VulkanMemoryModelDeviceScope`). NVIDIA/AMD keep
`Scope.QueueFamily`, no barriers — unchanged.

**FIX (implemented + verified): fixpoint MULTI-DISPATCH refit.** The only correct fix
changes the data-visibility granularity to dispatch boundaries (where Metal DOES give
cross-threadgroup coherency — confirmed: a 2-dispatch dependent read-after-write is
correct on MoltenVK). Raycore was VENDORED into `_rt/Raycore` (RayDemo's
`Project.toml` `[sources]` now points there) and patched at all 3 refit call sites
(`build_blas`, `build_tlas`, `refit_tlas!` in `instanced-bvh.jl`): replace the single
`refit_kernel!(...)` dispatch with a loop that re-dispatches the EXISTING
`refit_aabbs_kernel!`/`refit_tlas_aabbs_kernel!` to a fixpoint —
`n_refit_passes = max(2, 2*ceil(log2 n) + 8)`, resetting `update_flags` each pass.
Each pass propagates correct AABBs up ~one tree level (a node reads children made
coherent by the *previous* dispatch); ~log2(n) passes converge. Left UNCONDITIONAL
(not `is_moltenvk`-gated): this vendored copy is used only for the Metal render, and
the extra passes are cheap one-time work at scene load (they recompute identical
values on backends where one pass already converges). A root-stability early-out was
tried and is NOT safe (the root box can be transiently stable while a deep min/max is
still propagating) — a fixed safe pass count is used instead.

**VERIFIED:**
- Fast ground-truth gate (real Raycore BVH on a triangle mesh): GPU `root_aabb` ==
  CPU reference at N = 8 / 256 / 4096 / 65536 (was wrong for N ≥ 4096 before).
- Flat-white plain-mesh silhouette (no env, 6 directional lights, VolPath 20spp):
  SOLID white from hero / top / front — zero interior black voids. The blocky
  subtree voids on the solar wings / SAR antenna / body are gone. (`silh_hero.png`,
  `silh_top.png`, `silh_front.png` on disk are now the FIXED renders.)

**Lava emitter changes for Cat 5 (kept, harmless, but NOT what fixed it):** the
Device-scope atomic + bracketing `OpMemoryBarrier` (above) are correct and stay in
`_rt/Lava.jl`; they were proven insufficient on their own (Metal limit), and the
actual fix is the Raycore multi-dispatch refit.

NOTE: the earlier "all categories fixed / scene renders" status referred to
COMPILATION. Geometric CORRECTNESS (Cat 5) is the real finish line — now CLOSED.

## Reproducibility / environment gotchas
- **Stale precompile cache (cost the most time):** Julia repeatedly served a STALE
  precompiled Lava after source edits — both `Revise.revise()` and auto-precompile
  ran OLD code, so fixes appeared not to work. RELIABLE iteration after editing
  `_rt/Lava.jl`:
  `rm -rf ~/.julia/compiled/v1.12/Lava && julia --project=_rt/RayDemo -e 'using Pkg; Pkg.precompile("Lava")'`
  then run. Verify fresh code with a unique `println(stderr, ...)` marker, not
  `code_lowered` (the `@info` string isn't surfaced there).
- **MoltenVK log spam:** `ENV["MVK_CONFIG_LOG_LEVEL"]="0"` silences the ~150-line
  extension dump per device init.
- **scene.glb loader:** needs `using MeshIO` to register the FileIO `.glb` loader
  (the project's `Sentinel/sentinel.jl` does this; it's the canonical full-res
  renderer — `julia --project=. Sentinel/sentinel.jl` → 1600x1200/256spp).
- **Fast emitter iteration:** prototype SPIR-V transforms against
  `_rt/Lava.jl/../tmp_kernels/kernel_N_main.spv` (Lava's own per-session debug
  dump; apply `fold_convert_roundtrips!` first to match post-serialize) + `spirv-val`
  + `Lava.get_compute_pipeline(ctx, bytes, "main")` against MoltenVK — seconds per
  cycle vs ~10-min full renders. Run ONE full render to validate the whole
  ~29-kernel suite at once (fixes live in shared lowering, `is_moltenvk()`-gated).
