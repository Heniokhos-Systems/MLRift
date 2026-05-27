# Phase 3 — General @kernel AST → AMDGPU ISA Compiler — Design Survey

This document captures the state of the existing infrastructure as of
commit `1e559ec` (post-Phase 2) and the design choices Phase 3 needs
to make. It is a survey, not a plan-of-record — it exists so we can
walk into the implementation work knowing what already exists, what is
missing, and where the seams are.

---

## What works today

### Hand-tuned blob emitters (covered)

`src/format_amdgpu.mlr` ships hand-rolled emitters for the kernels
each milestone needed:

| Source surface | Emitter | Routing |
|----------------|---------|---------|
| `examples/llm/gemv_f32.mlr` | `_emit_gemv_f32_body` (N1) | name match |
| `examples/llm/gemm_f32.mlr` | `_emit_gemm_f32_body` (N2) | name match |
| `examples/llm/gemm_f32_lds.mlr` | `_emit_gemm_f32_lds_body` (N3) | name match |
| `examples/llm/gemm_f16f32.mlr` | `_emit_gemm_f16f32_body` (N4) | name match |
| `examples/llm/gemm_f16f32_wmma.mlr` | `_emit_gemm_f16f32_wmma_body` (N5) | name match |
| `noesis_60m_gpu.mlr` `decay_step` etc. | `amdgpu_lower_noesis_*` | name match |
| `_kfd_copy_u64` | `amdgpu_lower_kfd_copy_u64` | name match |
| Trivial PtrStore variants (M1d/M1e/grid_*) | structural recogniser | shape match |

Everything else hits the structural recogniser's "unsupported shape"
diagnostic. The diagnostic is fine; the gap is the long tail.

### AST + token plumbing (mature)

`src/ast.mlr` defines ~30 node kinds and a flat node arena (32 bytes
per node, indexed by 1-based id). Accessors: `ast_kind`, `ast_data1`,
`ast_data2`, `ast_data3`, `ast_data4`, `ast_child`, `ast_next`,
`ast_fn_is_kernel`. Tokens have `tok_kind_at_raw`, `tok_text_start`,
`tok_text_len`, `tok_matches`, `tok_text_eq`. All of this is reused by
Phase 1/2 lowerers and is solid.

Relevant node kinds for kernel bodies:

```
VarDecl(10)   data1=type_tok, data2=name_tok, child=init_expr
Assign(11)    data1=name_tok, child=value_expr
Return(12)    child=expr
If(13)        child=cond, data1=then_body, data2=else_body
While(14)     child=cond, data1=body
Block(15)     child=first_stmt
ExprStmt(16)  child=expr
IntLit(30)    data1=tok_idx
Ident(32)     data1=tok_idx
BinOp(33)     data1=op_tok, child=lhs, data2=rhs
UnaryOp(34)   data1=op_tok, child=operand
Call(35)      data1=callee_tok, child=first_arg
Compare(38)   data1=op_tok, child=lhs, data2=rhs
LogicalAnd(39) child=lhs, data1=rhs
LogicalOr(40)  child=lhs, data1=rhs
FloatLitNode(46) data1=tok_idx
```

There is no SSA layer, no IR-level type, no liveness pass — just the
parsed AST.

### asm_* helper library (119 functions)

`src/format_amdgpu.mlr` exposes one `asm_X` per RDNA3 instruction we
have needed so far. They write directly into `asm_text_buf` via
`asm_u32`, with no pretty-printing layer. Coverage by class:

- SOPP (s_endpgm, s_branch family, s_waitcnt, s_clause, s_delay_alu,
  s_barrier) — ~12 helpers.
- SOPK / SOP1 / SOP2 (s_mov, s_load_b32/b64/b256, s_lshl/lshr,
  s_cmp_lt/eq/ge, s_add_u32_imm) — ~15 helpers.
- VOP1/2/3 (v_mov, v_add_nc_u32, v_lshlrev_b32, v_lshlrev_b64,
  v_mul_lo_u32, v_fmac_f32, v_fma_f64 + variants, v_cmp_*,
  v_cndmask_b32, v_xor_b32, v_or_b32, v_and_b32, v_cvt_f32_f16,
  v_rcp_f64, v_rsq_f32) — ~50 helpers.
- VOPC + saveexec dance — ~6 helpers.
- FLAT/GLOBAL (global_load_b32/b64/b128/d16_b16/d16_hi_b16/u8,
  global_store_b32/b64/b8, global_atomic_add_f32/cmpswap_b64) — ~12.
- DS (ds_load_b32/b128, ds_store_b32/b128, ds_bpermute_b32) — ~4.
- WMMA (one variant inline, no helper) — 0.

Most VALU helpers take vgpr indices as raw `uint64`; SGPR helpers take
SGPR-pair indices (so `0` means s[0:1]). The convention is documented
inline at each helper.

### Existing AST → ISA pipeline

`emit_amdgpu_from_module` (line ~6238) walks the module, calls
`amdgpu_lower_kernel_to_table` per `@kernel` fn, then drains the
kernel table via `emit_amdgpu_multi_code_object`. Each lowerer is
expected to:

1. Call `asm_init()` (clears `asm_text_buf`).
2. Emit ISA via `asm_*` / raw `asm_u32`.
3. Snapshot `asm_text_buf` into a private allocation (because the
   next kernel's `asm_init` will clobber the buffer).
4. Build an `args_desc` array describing kernarg layout.
5. Call `kt_push` (or `kt_push_lds` for LDS-using kernels).

The structural recogniser around line 8390 is the "polymorphic"
lowerer: it handles M1d/M1e, plus four `is_*_shape` variants
(tid_guarded, decay, grid_decay, grid_fill, grid_relax). Each shape
has a hand-written emit branch (~50-100 lines of `asm_u32` calls).

This is what Phase 3 grows out of. The pattern works but doesn't
generalise — every new shape adds another `is_*_shape` branch, the
ISA is written by hand against the recognised shape, and there is no
reuse between branches.

---

## What is missing for a general compiler

### 1. No IR / SSA layer

The structural recognisers walk the AST directly and emit ISA inline.
They get away with it because each shape uses a fixed register layout
hardcoded in the emitter: e.g. decay uses `v0=tid_in_wg`, `v1=gid`,
`v[2:3]=byte_addr`, `v[4:5]=value`. There is no mechanism for "this
expression should land in some VGPR; pick one for me".

Phase 3 needs at minimum:

- A way to assign each AST expression result to a register (VGPR or
  SGPR depending on uniformity).
- A way to track types (u32 vs u64 vs f32 — picks different
  instructions).
- A way to track LIVENESS so registers can be reused once a value is
  consumed.

The cheapest first cut is: walk the AST, maintain a `next_vgpr` /
`next_sgpr` cursor, and never reclaim. Assigns one expression →
exactly one register, never reuses. Wastes registers but always
correct. That is what Phase 1's hand emitters do implicitly.

### 2. No type folding from the AST

`VarDecl` carries a `type_tok` — but the recognisers don't look at it
beyond gating "must be `uint32`" or "must be `f64`". For general code
we need to compute, for every expression, "what's the result type?"
so we know which instruction to pick (`v_add_nc_u32` vs
`v_add_f32_e32`). This is a small typechecker over the AST scope
walker.

We can lean on the existing typechecker (`src/analysis.mlr` /
`src/codegen.mlr` already has one — they accept/reject programs at
parse time) but that one tracks types at codegen time, not at
amdgpu-emit time. For Phase 3 we either reuse it via a side-channel
or run a small type pass over each `@kernel` body.

### 3. No HIP-intrinsic mapping table

The recognisers treat names like `tid_x`, `block_id_x`, `block_dim_x`,
`tid_local_x`, `tid_local_y`, `lds_load_f32`, `lds_store_f32`,
`sync_block` as anchors for shape detection — they don't actually
"compile" them, they just check for their presence as a sentinel that
"this is shape X" and then emit the corresponding hand-written ISA.

A general compiler needs to map each of these intrinsics to ISA:

```
tid_local_x()   → v0 (with rsrc2 ENABLE_VGPR_WORKITEM_ID set)
tid_local_y()   → v_bfe_u32 v_dst, v0, 10, 10  (gfx11 packs X+Y in v0)
block_idx_x()   → s2 (with rsrc2 WORKGROUP_ID_X bit 7 set)
block_idx_y()   → s3 (bit 8)
sync_block()    → s_waitcnt lgkmcnt(0) ; s_barrier ; buffer_gl0_inv
lds_load_f32(i) → ds_load_b32 v_dst, v_addr  (where v_addr = i*4)
lds_store_f32(i, v) → ds_store_b32 v_addr, v_val
```

This is a small lookup table, not a hard problem — but it has to be
defined explicitly somewhere.

### 4. Branch resolution / forward refs

Hand emitters compute branch displacements by counting dwords by
hand: `BFA50054 // s_cbranch_execz +84`. For general code we need:

- A label allocator (each `If`/`While`/`Break`/`Continue` gets a
  label).
- A backpatch list: when a branch's target is forward, record the
  branch position; resolve it when the target label is emitted.
- Branch-displacement encoding (signed 16-bit dword offset, range
  ±131072 dwords = ±524288 bytes — enough for any single kernel).

### 5. Kernarg layout calculation

The blob emitters hardcode `kernarg_size` (40 for gemv with 5 args of
8 bytes; 48 for the 6-arg gemms). General code needs to compute it
from the param list, with proper alignment per param type.

Conventions inherited from the structural recogniser: every param is
8 bytes, `uint64`/pointer kinds map to `AMD_ARG_KIND_GLOBAL_BUFFER`,
scalars to `AMD_ARG_KIND_BY_VALUE`. That's a fine starting point.

### 6. Dispatch geometry: who picks `block_*`?

Phase 1/2 emitters know `block_x=256` (decay) or `(16,16,1)` (gemm
LDS) or `(32,1,1)` (WMMA) because the launcher passes those values to
`hipModuleLaunchKernel`. The kernel itself just uses tid + block_id;
the dispatch geometry is the launcher's responsibility.

For Phase 3 we keep that: the kernel is geometry-agnostic, the
launcher chooses block size, and the lowerer emits whatever bounds
check the source actually wrote. (gfx11 wave32 → block_x must be a
multiple of 32; we don't enforce that here.)

### 7. Wait states / hazards

N3 fix taught us: gfx11 needs explicit `s_delay_alu` between a VALU
write of a VGPR and a VMEM/LDS read of the same VGPR. The hand
emitters now insert these by hand. A general compiler needs a hazard
pass: walk the emitted instruction stream, insert
`s_delay_alu instid0(VALU_DEP_N)` (or `s_nop`) anywhere a VALU→VMEM
RAW dependency exists.

This is the most subtle piece — getting it wrong produces clock-
dependent heisenbugs (see `project_gfx11_valu_vmem_hazard.md`). The
safe but slow approach: insert `s_delay_alu instid0(VALU_DEP_1)`
before every VMEM/LDS instruction that follows a VALU. Optimising
later is fine.

---

## Recommended Phase 3 implementation order

**3a — Pointwise 1-D scalar kernels.** Body shape:

```
@kernel fn pointwise(uint64 buf, uint64 n, [scalar args...]) {
    uint64 gid = block_idx_x() * 256 + tid_local_x()
    if gid < n {
        uint64 p = buf + gid * <SIZE>
        <T> v = 0
        unsafe { *(p as <T>) -> v }
        v = <single binop or chain over scalar args>
        unsafe { *(p as <T>) = v }
    }
}
```

This unlocks: relu, leaky_relu, scale, bias_add (with broadcast),
sigmoid (if we have v_exp_f32), tanh, gelu approx, layernorm-stats
prep, etc. Real LLM-adjacent surface, comparatively small AST shape
(linear control flow + one if).

What 3a needs from the missing list:
- (1) trivial reg alloc — ~3 VGPRs, no reuse.
- (2) type folding — only u32/u64/f32 / no overloading.
- (3) HIP intrinsics table — only `tid_local_x` + `block_idx_x`.
- (4) branch resolution — exactly one forward branch (the bounds
  check) — can be hand-resolved.
- (5) kernarg layout — straightforward 8-byte stride.
- (7) wait states — dump `s_delay_alu instid0(VALU_DEP_1)` before
  every global_load/global_store that follows a VALU.

Estimated size: ~600 lines in `format_amdgpu.mlr`, ~1 day of work.

**3b — Reduction / softmax / layernorm.** Adds: cross-lane reductions
(`v_readlane_b32`, ds_bpermute or pair-reduce in LDS), block-level
sync, multi-pass body. Significantly bigger.

**3c — Multi-WG tile kernels.** Adds: 2-D dispatch, 2-D LDS staging,
multi-statement bodies with locals, K-loop generation. Approaches the
generality of the hand-written N1-N5 emitters.

**3d — Atomics + structured control flow.** Adds: CAS loops, exec-
masked branches, full backpatching.

**3e — General SSA layer.** At this point, the right move is to
introduce a real IR (similar to `src/ir.mlr`'s host IR) and run a
proper register allocator + scheduler + hazard pass over it, instead
of growing the ad-hoc walker further.

---

## Open questions to resolve before 3a

1. **Re-use the host IR (`src/ir.mlr`) or build a separate AMDGPU
   IR?** The host IR is fairly type-aware and has SSA-ish properties
   already; we might be able to lower @kernel bodies into it and then
   run a one-off "ir-to-amdgpu" pass. Risk: host IR's instruction set
   doesn't have GPU ops (lane id, ds, atomics), so we'd be stretching.

2. **Where does the hazard pass live?** Easiest: emit a list of
   `(opcode, vgpr_writes, vgpr_reads)` records as we go, then walk
   that list at the end and inject `s_delay_alu` where needed. Costs
   another buffer.

3. **Where do HIP intrinsics live syntactically?** They're currently
   defined in MLRift source as `fn tid_local_x() -> uint64 { return
   0 }` stubs that exist only so the source typechecks. The AMDGPU
   lowerer ignores the body and recognises the call by name. Phase 3
   should formalise this — perhaps a `@gpu_intrinsic` attribute or a
   well-known module path so the lowerer doesn't have to name-match.

4. **Where do we draw the line on register reuse?** With granulated
   counts of 4-8 (32-64 VGPRs) we have plenty of headroom for
   pointwise shapes. Skip reuse for 3a and revisit for 3c.

---

## Recommended next step

Implement 3a as outlined above, with the simplifying choices on every
open question. Land it behind an "experimental" path that doesn't
displace the existing structural recogniser (so trivial M1d/M1e
kernels don't regress). When 3a is solid, write 3b's design doc
before starting it.
