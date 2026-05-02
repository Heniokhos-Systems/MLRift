# Phase 3 — Kernel DSL primitive set, distilled from gemv_coop_f32

This document is the design step that the strategic memo
(`memory/project_no_handwritten_asm_evolve_compiler.md`) pinned as
the prerequisite for Phase 3 implementation.  The exercise: walk a
known-good hand-tuned kernel block-by-block and name each block by
the high-level primitive it represents.  The result is the **target
vocabulary the MLRift `@kernel` compiler must support** to produce
the same kernel from a high-level source.

We use `_emit_gemv_coop_f32_body` (`src/format_amdgpu.mlr` line 5917)
because:

- Smallest hot kernel (~80 ASM bytes worth of distinct logic, plus
  loop unroll).
- Already-validated correctness (270× speedup vs naive gemv on
  Qwen3-0.6B, bit-identical to PyTorch).
- Exercises every primitive class except cross-WG sync: kernarg
  load, scalar arith, scalar branch, per-lane init, K-loop with VMEM
  loads + FMAC chain, LDS reduce, single-lane store, exec-mask
  dance, forward-branch backpatch.
- Does NOT exercise: wmma, atomics, multi-D dispatch, double-buffer
  staging, tiled blocking — those land in later slices (3c+/3d).

## Block walk

Each block below shows the existing `asm_u32` calls, what they're
doing semantically, and the **primitive** the compiler would emit
to reproduce them.

### Block 1: kernarg prologue

```asm
s_load_b256 s[4:11], s[0:1], 0x0     ; 32 bytes of kernarg → s4..s11
s_load_b64  s[12:13], s[0:1], 0x20   ;  8 bytes more → s12..s13
s_waitcnt   lgkmcnt(0)               ; drain SMEM
```

Loads 5 args (a_ptr, b_ptr, c_ptr, m, k) at fixed kernarg offsets.

**Primitive**: `kernarg_load(args)` — the compiler computes total
kernarg size from the function signature, picks the largest
power-of-2-≤16 chunked load (`s_load_b256` for 32-byte chunks,
`s_load_b64` for the tail), and emits one `s_waitcnt lgkmcnt(0)` at
the end.

### Block 2: workgroup-level bounds check

```asm
s_cmp_ge_u32  s15, s10               ; wg_id_x (s15) >= m (s10) ?
s_cbranch_scc1 .Lend                 ; if yes, whole wave bails
```

The MLRift source equivalent:
```
if block_idx_x() >= m { return }
```

**Primitives**:
- `block_idx_x()` → s15 (wg_id_x is preloaded by hw at s15 when
  `rsrc2 ENABLE_SGPR_WORKGROUP_ID_X` is set; user_sgpr_count=15
  pushes wg_id_x to s15).
- `scalar_compare(op, lhs_sreg, rhs_sreg)` → `s_cmp_*_u32`.
- `scalar_branch_if(label)` → `s_cbranch_scc1` with backpatch when
  target is forward.

### Block 3: row-base address computation (uniform)

```asm
s_mul_i32   s16, s15, s12            ; s16 = wg * k
s_lshl_b32  s16, s16, 2              ; s16 *= 4 (bytes)
s_add_u32   s16, s4, s16             ; s[16:17] = a_ptr + s16
s_addc_u32  s17, s5, 0               ;   (with carry into hi)
```

A 64-bit address computed in scalar registers (uniform across the
wave because every lane in this WG accesses the same row).

**Primitive**: `scalar_addr_compute(base_sreg_pair, offset_expr)` —
computes a 64-bit ptr-arithmetic into a fresh sreg pair, with proper
carry handling.  The compiler folds `wg * K * 4` into mul+lshl
when K isn't a constant (or just `wg * K_bytes` if K_bytes is a
folded constant).

### Block 4: per-lane scalar init

```asm
v_lshlrev_b32 v6, 2, v0              ; v6 = lane * 4 (LDS slot)
v_mov_b32     v7, v6                 ; v7 = lane * 4 (byte offset)
v_mov_b32     v5, 0                  ; v5 = sum accumulator
s_mov_b32     s18, 0                 ; s18 = chunk counter
s_lshr_b32    s19, s12, 8            ; s19 = num_chunks = K/256
```

Sets up loop variables.  v0 is preloaded by HW with lane id (0..31).

**Primitives**:
- `lane_id()` → v0 (preloaded when `rsrc2 ENABLE_VGPR_WORKITEM_ID`
  is set; in wave32 mode lane_id is 0..31).
- `vector_init(vreg, expr)` — picks `v_mov_b32` or
  `v_lshlrev_b32 ..., shift, v0` based on `expr`.
- `scalar_init(sreg, expr)`.
- `scalar_arith(op, dst, src1, src2)` — `s_lshr_b32` etc.

### Block 5: K-loop (8× unrolled VMEM + FMAC chain)

```asm
.Lloop:
   ; 16 b32 global loads (8 from A row, 8 from B vector), interleaved
   global_load_b32 v8,  v7, s[16:17]              ; A[lane]
   global_load_b32 v9,  v7, s[6:7]                ; B[lane]
   global_load_b32 v10, v7, s[16:17] offset:128   ; A[lane+32]
   global_load_b32 v11, v7, s[6:7]   offset:128   ; B[lane+32]
   ... [12 more loads, offsets 256/384/512/640/768/896] ...
   s_waitcnt vmcnt(0)                              ; drain VMEM
   v_fmac_f32 v5, v8,  v9                          ; sum += A*B
   v_fmac_f32 v5, v10, v11
   ... [6 more fmacs] ...
   v_add_nc_u32 v7, 1024, v7                      ; advance byte_off by 8 strides
   s_add_u32 s18, s18, 1                          ; chunk++
   s_cmp_lt_u32 s18, s19                          ; chunk < num_chunks ?
   s_cbranch_scc1 .Lloop                          ; back-edge (backpatch)
```

This is the perf-critical block.  The 8× unroll exposes ILP for the
GFX11 dual-issue VALU; the interleaved A/B loads keep both VMEM
channels busy; the single drained-once `s_waitcnt vmcnt(0)`
amortises the wait across 16 loads.

**Primitives**:
- `vmem_load(dst_vreg, addr_sreg_pair, vgpr_offset, byte_offset_imm)`
  → `global_load_b32` with optional offset.  `byte_offset_imm` is in
  the 12-bit immediate range (0..4095).
- `vmem_drain()` → `s_waitcnt vmcnt(0)`.
- `fmac_f32(acc_vreg, a_vreg, b_vreg)` → `v_fmac_f32 acc, a, b`.
- `vector_arith_imm(op, dst, src, imm)` → `v_add_nc_u32 v_dst, imm,
  v_src` for the byte_off increment.

**Composer**: `fmac_chain(N, inputs[N], output_acc)` emits the N
fmacs in sequence (no compiler magic — just unrolled).

**Composer**: `vmem_load_chain(N, addr1, addr2, stride, dsts1[N],
dsts2[N])` emits the 2N interleaved loads + the single
`s_waitcnt vmcnt(0)`.

**Higher-level composer**: `cooperative_kreduce_loop(K, unroll,
weight_loader, input_loader, accumulator)` emits the entire K-loop
including the chunk counter, back branch, and offset increment.
This is the central abstraction the compiler needs for cooperative
gemv-class kernels.

### Block 6: LDS-broadcast + 32-lane reduce

```asm
ds_store_b32  v6, v5                 ; LDS[lane*4] = my_partial_sum
s_waitcnt lgkmcnt(0)                 ; drain LDS
s_barrier                            ; wave-sync
{lds_reduce_pow2(32) — 5 stages of paired add over LDS}
```

The lds_reduce_pow2 helper itself emits 5 stages, each:

```asm
ds_load_b32  v7, v6 offset:K*4       ; load partner
s_waitcnt lgkmcnt(0)
v_add_f32 v5, v5, v7                  ; combine
v_cmp_gt_u32 vcc_lo, K, v0            ; lane mask
s_and_saveexec_b32 sN, vcc_lo
ds_store_b32 v6, v5                   ; only first K lanes write back
s_mov_b32 exec_lo, sN                 ; restore exec
s_waitcnt lgkmcnt(0)
s_barrier
```

(K = 16, 8, 4, 2, 1 across the 5 stages.)

**Primitives**:
- `lds_store(addr_vreg, val_vreg)` → `ds_store_b32`.
- `lds_load(dst_vreg, addr_vreg, byte_offset_imm)` → `ds_load_b32`.
- `lds_drain()` → `s_waitcnt lgkmcnt(0)`.
- `workgroup_barrier()` → `s_barrier`.
- `exec_mask_save_lt(saved_sreg, lit)` → cmp + and_saveexec_b32.
- `exec_mask_restore(saved_sreg)` → `s_mov_b32 exec_lo, sN`.

**Composer**: `lds_reduce_pow2(N, accum_vreg, scratch_vreg,
slot_vreg, op)` emits the full reduction tree.  The compiler can
generate this from a `__cooperative_reduce(accum, op)` intrinsic in
the `@kernel` source.

### Block 7: lane-0 store + exec restore

```asm
v_cmp_eq_u32 vcc_lo, 0, v0           ; lane==0 ?
s_and_saveexec_b32 s2, vcc_lo        ; mask exec to lane 0
v_mov_b32 v3, s15                    ; v3 = row
v_lshlrev_b32 v3, 2, v3              ; v3 *= 4 (bytes)
global_store_b32 v3, v5, s[8:9]      ; c[row] = total_sum
s_waitcnt_vscnt null, 0
s_mov_b32 exec_lo, s2                ; restore exec
```

**Primitives**:
- `vmem_store(addr_sreg_pair, vgpr_offset, val_vreg)` →
  `global_store_b32`.
- `vmem_store_drain()` → `s_waitcnt_vscnt null, 0`.

**Composer**: `single_lane_emit(lane, body)` wraps the body in the
exec-mask save/restore dance.

### Block 8: epilogue + branch backpatch

```asm
.Lend:
s_endpgm
```

Plus, after emitting all blocks, the bounds-check forward branch
from Block 2 gets its target patched to `.Lend`.

**Primitives**:
- `kernel_end()` → `s_endpgm`.
- `label_resolve(name)` — backpatches any pending forward refs to
  `name` with their now-known dword distance.

### Block 9: kernarg metadata

```c
args_desc[a0] = 8; args_desc[a1] = AMD_ARG_KIND_GLOBAL_BUFFER;
args_desc[a2] = 8; args_desc[a3] = AMD_ARG_KIND_GLOBAL_BUFFER;
args_desc[a4] = 8; args_desc[a5] = AMD_ARG_KIND_GLOBAL_BUFFER;
args_desc[a6] = 8; args_desc[a7] = AMD_ARG_KIND_BY_VALUE;
args_desc[a8] = 8; args_desc[a9] = AMD_ARG_KIND_BY_VALUE;
```

**Primitive**: synthesised from the `@kernel` function's parameter
list — `uint64`/pointer types map to `GLOBAL_BUFFER`, scalars to
`BY_VALUE`.  Already done implicitly by the existing recogniser
pattern; lift to an explicit pass.

## Distilled primitive set

Grouped by category, this is the **vocabulary the gfx1100 backend
must support** to compile `cooperative_gemv` from `@kernel` source:

```
Scalar:
  scalar_compare(op, sreg, sreg)
  scalar_branch_if(label)
  scalar_arith(op, dst, src1, src2)        [s_mul_i32, s_lshl_b32,
                                             s_lshr_b32, s_add_u32,
                                             s_addc_u32, s_mov_b32]
  scalar_addr_compute                       (composer of the above)
  kernarg_load(args)                        (composer using s_load_b{64..512})

Vector:
  lane_id()                                 → preloaded v0
  vector_init(vreg, expr)
  vector_arith_imm(op, vreg, vreg, imm)     [v_add_nc_u32, v_lshlrev_b32]
  fmac_f32(acc, a, b)                       → v_fmac_f32

Memory:
  vmem_load(dst, base_sreg, vgpr_off, imm)  → global_load_b32
  vmem_drain()                              → s_waitcnt vmcnt(0)
  vmem_store(base_sreg, vgpr_off, val)      → global_store_b32
  vmem_store_drain()                        → s_waitcnt_vscnt null, 0

LDS:
  lds_store(addr_vreg, val_vreg)            → ds_store_b32
  lds_load(dst, addr_vreg, imm)             → ds_load_b32
  lds_drain()                               → s_waitcnt lgkmcnt(0)

Synchronisation:
  workgroup_barrier()                       → s_barrier
  exec_mask_save_lt(saved, lit)             (cmp + and_saveexec_b32)
  exec_mask_restore(saved)                  → s_mov_b32 exec_lo, sN

Block intrinsics:
  block_idx_x()                             → s15 (rsrc2 dependent)
  block_idx_y()                             → s16
  thread_idx_x()                            → v0[0..9]   (or v0 in wave32)

Composers (compose primitives, may unroll):
  fmac_chain(N, ins[N], acc)
  vmem_load_chain(N, base_a, base_b, stride, dst_a[N], dst_b[N])
  cooperative_kreduce_loop                  (the K-loop above)
  lds_reduce_pow2(N, accum, scratch, slot)
  single_lane_emit(lane, body)
  scalar_kernel_bounds_check(idx_sreg, max_sreg, exit_label)
```

About **20 primitives** + **6 composers**.  That's the full surface
the gfx1100 backend needs to expose for cooperative gemv.

## What this implies for the compiler

A `@kernel` source for `cooperative_gemv_f32` would look roughly:

```mlrift
@kernel
fn cooperative_gemv_f32(uint64 a, uint64 b, uint64 c, uint64 m, uint64 k) {
    uint64 row = block_idx_x()
    if row >= m { return }
    uint64 lane = thread_idx_x()

    f32 sum = 0.0f
    uint64 chunks = k >> 8           // K / 256
    uint64 i = 0
    while i < chunks {
        // 8× unroll across stride 128
        @unroll(8)
        for s in 0..8 {
            f32 av = vmem_load_f32(a, row*k*4 + i*1024 + s*128 + lane*4)
            f32 bv = vmem_load_f32(b, i*1024 + s*128 + lane*4)
            sum = fmac(sum, av, bv)
        }
        i = i + 1
    }
    f32 total = cooperative_reduce_add_pow2(sum)
    if lane == 0 {
        vmem_store_f32(c, row*4, total)
    }
}
```

The compiler sees:
1. `block_idx_x()`, `thread_idx_x()` — intrinsic table lookup.
2. `if row >= m { return }` — bounds check pattern.
3. The K-loop with `@unroll(8)` — emit 8× unrolled VMEM/FMAC chain.
4. `cooperative_reduce_add_pow2(sum)` — emit lds_store + barrier +
   lds_reduce_pow2(32) tree.
5. `if lane == 0 { ... }` — single-lane emit pattern.
6. `vmem_load_f32` / `vmem_store_f32` — primitive.
7. The hazard pass auto-inserts `s_delay_alu` between any VALU
   write and following VMEM/LDS read.
8. `s_endpgm` at function end.

That's the target.  Slice A.0 (Phase 3a — pointwise) builds the
intrinsic table + bounds-check pattern + scalar arith + simple
VMEM loads.  Slice A.1 (Phase 3b — reductions) builds
`cooperative_reduce_*` and the LDS reduction tree.  Slice A.2
(Phase 3c — cooperative tile) ties them together with the K-loop +
unroll directive — and at that point we should be able to compile
the @kernel source above into the gfx1100 ISA bytes that match
`/tmp/gemv_coop_f32.co` byte-for-byte.

That's the milestone for the strategic memo's "compiler matches
hand-tuned within 1 %".

## Salvage of existing work

Every encoding decision and gotcha already documented stays
load-bearing:

- The 119 `asm_X` helpers ARE the gfx1100 backend's primitive
  emission table.  They get called from compiler passes instead of
  from host code.
- The `s_delay_alu` rule (gfx11 VALU→VMEM hazard from the N3 fix
  memo) becomes the hazard pass.
- The WG_ID_X-at-s15 rule (with rsrc2=0x9E + user_sgpr_count=15)
  becomes the `block_idx_x()` lowering rule.
- The magic-number kv_head divide in attn_decode_14b becomes a
  general "scalar integer divide by constant" pattern in the
  compiler (any odd nh/nkv ratio uses mul_hi + lshr).
- The VGPR-aliasing-with-input-loads gotcha (Q4 unpack scratch
  collision) becomes the register-allocator's "don't reuse VGPRs
  while their live range overlaps" rule — the compiler's job, not
  the kernel writer's.

None of the work is wasted.  All of it gets lifted into the
compiler.
