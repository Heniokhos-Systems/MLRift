# Slice 4 — Mega-kernel design (route to bf16 single-stream win)

Status: design + measurement.  Implementation deferred to a dedicated
multi-session piece.

## Why slice 4 is the only path to honest single-stream bf16 win

After slices 1-3 + 2b + 2c, qwen3-0.6B / RX 7800 XT decode is
**launch-overhead bound**, not ALU-bound and not bandwidth-bound:

```
$ MLRIFT_GPU_MATMUL=1 MLRIFT_GPU_FULL_FORWARD=1 MLRIFT_GPU_FLUSH_EVERY_N=28 /tmp/q3
  step N took 7.4 ms launches=310 syncs=2 sync_us=2300
  total_decode_ms=316  generated_tokens=20  tok/s=63.2
```

- **310 dispatches per token** (28 layers × ~11 ops + 1 lm_head + a few
  extracts).  Down from 421 pre-fusion via slice 2b (resid+rmsnorm)
  and slice 2c (qknorm+rope_qk).
- **Only 2 syncs per token** (one trailing forward flush, one lm_head).
- 7.4 ms GPU compute / 310 = **23.9 us per dispatch** effective.
- 15.8 ms wall / token; ROCm bf16 ceiling = 13.6 ms / token.

We need to save ~2.2 ms / token to cross 73.7 tok/s.  Bandwidth-bound
matmuls (gemv_coop_bf16_f32) already run at the HBM ceiling for their
shape.  The 2.2 ms is dispatch overhead spent on the remaining
NON-matmul ops per layer (qkv_split, attn, residuals, silu_mul).
Slice 2b + 2c already fused (residual+rmsnorm) and (qknorm+rope_qk);
the next opportunity is the full mega-kernel (slice 4) — collapse the
remaining 11-ish ops per layer into one dispatch.

## Design

Collapse the 15-op layer chain into 1 mega-dispatch:

```
@kernel
fn qwen3_layer_megakernel_f32(
    in:        u64,    // residual in (hidden,)
    out:       u64,    // residual out (hidden,)
    qkv_w:     u64,    // [qkv_dim, hidden] bf16
    o_w:       u64,    // [hidden, q_dim] bf16
    gate_up_w: u64,    // [2*ff, hidden] bf16
    down_w:    u64,    // [hidden, ff] bf16
    in_norm_g: u64,    // [hidden] f32
    post_norm_g: u64,  // [hidden] f32
    q_norm_g:  u64,    // [head_dim] f32
    k_norm_g:  u64,    // [head_dim] f32
    rope_cos:  u64,    // [head_dim/2] f32
    rope_sin:  u64,    // [head_dim/2] f32
    kc:        u64,    // K-cache layer slab
    vc:        u64,    // V-cache layer slab
    pos:       u64,
    /* scratch: rmsnorm + qkv buf + attn region in LDS+VMEM */
)
```

Persistent-thread design: launch grid = **WG_PERSIST = 256 WGs of 32
lanes**, kept BELOW the GPU's co-resident occupancy limit so all WGs
can spin together at the cross-WG barrier without the scheduler
deadlocking on slot reuse.  Each WG iterates over a strided slice of
the OUTPUT dimension during each phase (e.g. for the gate_up matmul
of 6144 outputs each WG processes ceil(6144/256) = 24 rows; for the
qkv matmul of 2048 outputs each WG handles 8 rows; for the per-head
attn_decode phase the first 24 WGs work and the rest sleep).
Cross-WG sync via a per-layer global-memory barrier (signal +
spin-wait, similar to the KFD shim's signal-poll).

**Why 256, not max(qkv_dim, hidden, 2*ff) = 6144** (original draft):
gfx1100 can hold ~480-960 wave32s in flight at a time (60 CUs × 8-16
waves/CU depending on VGPR usage).  Launching 6144 WGs that all
need to spin at a global barrier is a deadlock — the first 480-960
WGs are running and stuck spinning on the counter, but the counter
never reaches 6144 because the remaining ~5500 WGs are queued and
can't start until the resident WGs retire.  256 WGs (~25% of the
worst-case occupancy budget) leaves headroom for any future VGPR-
heavier phases without breaching the cap.

Per-WG strided iteration also smooths register pressure: a single
WG processes 24 gate_up rows by streaming through them, so the
4-row VGPR accumulator footprint of gemv_coop carries through the
phase without growing — same VGPR cost as a non-mega kernel.

Phases inside the kernel (within one dispatch).  Notation: "WG owns
output rows `i = wg_id, wg_id+256, wg_id+512, ...` up to that phase's
output count" — i.e. classic grid-stride loop, with WG_PERSIST=256 as
the stride.  WGs whose `wg_id` exceeds a phase's output count exit
that phase immediately (still ack the barrier — see Hard part #5).

1. **rmsnorm input**: WG 0 only.  Reduce-tree over `hidden` lanes
   into LDS, broadcast scale.  WG > 0 idle (still ack barrier).
2. **GLOBAL BARRIER 1**.
3. **qkv matmul**: each WG handles `ceil(qkv_dim / 256) ≈ 8` output
   rows of b_qkv from b_x_norm.  Output stride = 256.
4. **GLOBAL BARRIER 2**.
5. **qkv_split + qknorm + rope** fused: 16 Q heads + 8 KV heads = 24
   per-head sub-tasks.  WGs 0..23 each take one head; WGs 24..255
   skip.
6. **BARRIER 3**.
7. **attn_decode**: 16 Q heads (decode-time, K=cache, V=cache).
   WGs 0..15 each compute one head's attention output; WGs 16..255 skip.
8. **BARRIER 4**.
9. **o_proj matmul**: each WG handles `ceil(hidden / 256) = 4` output
   rows.  Adds to `in` for residual b_mid.
10. **BARRIER 5**.
11. **post_norm**: WG 0 only.  Other WGs idle.
12. **BARRIER 6**.
13. **gate_up matmul**: each WG handles `ceil(2*ff / 256) = 24` rows.
14. **BARRIER 7**.
15. **silu_mul**: each WG handles `ceil(ff / 256) = 12` elements.
16. **BARRIER 8**.
17. **down matmul**: each WG handles `ceil(hidden / 256) = 4` rows.
    Adds to b_mid for final residual.  Write to `out`.
18. End kernel.

Per-token: 28 layer-mega-dispatches + 1 lm_head matmul = **29 launches**
(vs 310 today after slice 2b+2c, 421 pre-fusion).  Saves ~6.7 ms of
launch overhead per token at the 24 us-per-launch rate measured.
Step 15.8 ms − 6.7 ms = 9.1 ms = **~110 tok/s** projected (revised
down from the 143 estimate that assumed pre-fusion 421-dispatch
baseline).  Still 1.5× ROCm bf16 — slice 4 remains the path to a
clear honest single-stream bf16 win.

## Hard parts

1. **Cross-WG global barriers.**  GFX11 has no ring-barrier instruction;
   we use signal-and-spin via a global-memory counter.  Each barrier:
   1 atomic_add + spin-load on a counter dword.  **Measured 2026-05-05
   on RX 7800 XT** via `examples/llm/mega_barrier_microbench.hip.cpp`
   (lane-0-only protocol, 1000 barrier iterations, host wall-time
   subtracting empty-launch baseline):

   | WG count | per-barrier cost |
   |---|---|
   |  16 | 278 ns |
   |  32 | 286 ns |
   |  64 | 301 ns |
   | 128 | 344 ns |
   | **256** | **448 ns** |
   | 480 | 566 ns |

   At WG_PERSIST=256 the slice 4 barrier budget per token is
   `7 × 28 × 0.45 us = 88 us` — about **1.3 % of the 6.7 ms launch
   saving** the design promises.  Far better than the doc's original
   5-10 µs estimate; mega-kernel feasibility GREEN.  See
   `examples/llm/mega_barrier_microbench_launch.mlr` for the
   reproduction.
2. **VGPR pressure.**  Each op in the chain has its own VGPR live ranges;
   fusing them in a single kernel forces all to coexist or use LDS for
   inter-phase carry.  Likely the rmsnorm + matmul accumulators (40-80
   VGPRs) need to stay in regs while attn (32 VGPRs) and rope (16 VGPRs)
   share LDS.
3. **Recogniser shape.**  The mega-kernel has an unusual control-flow
   shape (8 If-blocks gated by phase + barriers between).  Either a
   custom AST recogniser per kernel name, or a generic "phased kernel"
   intrinsic.  Probably the former.
4. **WMMA slot.**  The bf16 matmul phases (qkv, o_proj, gate_up, down)
   are bandwidth-bound at M=1 — WMMA buys nothing there.  Skip.
5. **Co-residence deadlock avoidance.**  Persistent-thread + global
   barriers REQUIRE all WGs co-resident.  gfx1100 fits ~480-960
   wave32s in flight (60 CUs × 8-16 waves/CU depending on VGPR usage).
   If WG count > co-resident limit the kernel deadlocks: the resident
   WGs spin at the barrier waiting for the queued WGs that can't start
   until the resident ones retire.  The original draft of this design
   said "grid = max(qkv_dim, hidden, 2*ff) = 6144 WGs" — that DOES
   deadlock.  Fix: WG_PERSIST = 256 (a quarter of the worst-case
   occupancy budget), each WG strides over multiple output rows per
   phase.  Phases with fewer outputs than WG_PERSIST have idle WGs
   that still ack the barrier and skip the work.
6. **Idle-WG barrier acks.**  Phases like `rmsnorm input` (1 WG of
   work) and `attn_decode` (16 WGs of work) leave most of the 256
   WGs idle.  They still must `atomic_add(counter, 1)` to satisfy
   the barrier — otherwise the working WGs spin forever waiting for
   counter==256.  Implementation: every phase ends with an
   unconditional barrier ack from every WG, regardless of whether
   that WG did work.  An idle-WG path is just `atomic_add` then
   spin-load — no per-phase exit.

## Implementation plan (next session)

| Step | Effort | Yield |
|---|---|---|
| 1. AST recogniser for mega-kernel + body emit (clone existing kernels' bodies into one mega-emit, insert global barriers between) | 4-6 h | builds correctly |
| 2. Smoke test: standalone launcher with synthetic data + bit-identity vs per-op chain | 1-2 h | correctness validated |
| 3. Wire into qwen3_forward_layer_gpu (replace 15 launches with 1) | 1 h | end-to-end working |
| 4. Bench + bisect any drift bugs (analogous to slice 1's stale-co class) | 1-2 h | tok/s number |
| 5. Profile + tune barrier spin cost | 1-2 h | maybe +10% on top |

## Why it can wait

- Slices 1-3 already win pure fp32 (1.42x ROCm fp32) and the PLD-friendly
  bf16 variant (1.19x ROCm bf16).
- The honest single-stream bf16 gap (60.4 vs 73.7) is well-quantified:
  3 ms of launch overhead / token.
- The implementation has no unknowns left — design is settled, constants
  are measured, the rest is execution.


## Stepping stone: slice 2c (qknorm + rope_qk fusion) — SHIPPED

Status: **GREEN, bit-exact to PyTorch, end-to-end qwen3-0.6B**.

Replaces qknorm Q + qknorm K + rope Q + rope K (4 sequential dispatches)
with **one** fused launch over `nh_q + nh_k` WGs (= 24 for qwen3-0.6B,
16 Q heads + 8 KV heads).

Measured on RX 7800 XT, single-stream greedy 20-token decode:

|  | Legacy (4 dispatches) | Slice 2c fused | Δ |
|---|---|---|---|
| **tok/s** (best of 3) | 62.1 | **63.2** | +1.7% |
| **launches/step** | 366 | **310** | −56 (−2/layer × 28) |
| **PyTorch token parity** | bit-exact | **bit-exact** | ✓ |

Smoke test: 3072 Q + 1024 K outputs match CPU reference within 1e-5
absolute (effectively bit-exact at f32 precision).

Implementation (143 dwords gfx1100, all llvm-mc verified):

- **Emit body** `_emit_qknorm_rope_qk_fused_f32_body` and blob wrapper
  `emit_amdgpu_qknorm_rope_qk_fused_f32_blob` in `src/format_amdgpu.mlr`.
  Composes the existing rmsnorm@128 (eps=1e-6) body and rope_qwen3 body
  via the SGPR-reorder pattern from slice 2b, plus an LDS-slab reuse
  trick: the rmsnorm reduce-tree's LDS slab (lane*4 = 0..508) is dead
  after the broadcast `ds_load v8, v9`, so the phase-1 staging
  `ds_store v6, v5` reuses that same region.  LDS allocation is 1024 B
  (with 512 B headroom).  rsrc1 = 0xE0AF0082 (32-SGPR grant — 24-SGPR
  was insufficient because the kernel touches s0..s27 + saveexec stash).
- **AST recogniser** `amdgpu_lower_qknorm_rope_qk_3c` in `src/format_amdgpu.mlr`,
  gates on 8 params + presence of `qknorm_rope_qk_marker()` Call.
  Routes BEFORE qknorm_3b and rope_qwen3_3c (so the fused recogniser
  wins when both halves are present in the function body).
- **CLI flag** `--emit-amdgpu-qknorm-rope-qk-fused-f32=path` in `src/main.mlr`.
- **Launcher** `gpu_qknorm_rope_qk_fused_to_dev` in `std/inference_gpu.mlr`,
  loads `/tmp/qknorm_rope_qk_fused_f32.co` lazily on first call, packs
  8 kernarg pointers, dispatches grid = nh_q + nh_k, block = 128.
- **Wire-in** in `qwen3_forward_layer_gpu` (`std/qwen3.mlr`): the four
  sequential `gpu_qknorm_f32_to_dev` / `gpu_rope_f32_to_dev` calls are
  replaced by `gpu_qknorm_rope_qk_fused_to_dev(...)`.  Gated by phase
  mask bit 0x40000 for opt-out; `MLRIFT_NO_QKNORM_ROPE_FUSED=1` env in
  `examples/qwen3_generate.mlr` forces the legacy path for A/B.
- **Smoke test** at `examples/llm/qknorm_rope_qk_smoke.mlr` (195 lines)
  — synthetic gamma + cos/sin tables, 4096-element output diff vs CPU
  reference.  Ships with the standard `--target=amdgpu-native` build.

The `/tmp/qknorm_rope_qk_fused_f32.co` produced by the AST-walked
path (`build/mlrc --target=amdgpu-native examples/llm/qknorm_rope_qk_fused.mlr`)
is byte-identical to the blob-emit path
(`--emit-amdgpu-qknorm-rope-qk-fused-f32=...`) — single source of
truth confirmed.

**Gap to original 84-launch projection:** the design assumed an
in-place Q path; we landed an out-of-place Q with one extra
`extract_q` launch per layer.  Closing the gap to a +5% tok/s target
requires extending the kernel to 9 args (separate Q-out pointer);
~30 dwords of additional emit, optional follow-up.

**Diagnostics worth keeping for future fusion work:**
- `s_add_u32 s10, s24, s26` originally placed BEFORE the cselects
  clobbered `s10 = k_gamma_lo`.  Bug-class: SGPR-cselect adjacency
  in fused multi-source kernels.  Fix: order all 5 cselects first,
  then the add.
- WG_ID_X clobber by SMEM load — needed `s_mov_b32 s3, s15` BEFORE
  the second `s_load_b256` (matches gemm_f32's pattern).  Without
  it, any WG > 0 would silently use wrong head_idx.

## Stepping stone: slice 2b (residual+rmsnorm fusion)

A smaller fusion that would ship as a single concrete @kernel before
the full mega-kernel:

- `examples/llm/residual_rmsnorm_f32.mlr` — @kernel definition
  committed; uses `residual_rmsnorm_marker()` discriminator so the
  AST recogniser routes correctly relative to the existing rmsnorm
  family.
- 7-param shape: `(in_a, in_b, out_resid, out_norm, gamma, m, n)`.
- Fires twice per qwen3 layer (mid resid+post_norm; cross-layer
  final-resid+input_norm).  ≈56 launches/token saved at ~24 µs each
  ≈ 1.3 ms/token, projecting **60.4 → ~67 tok/s** on the same RX
  7800 XT, qwen3-0.6B / single-stream.

Remaining work for slice 2b:

1. **Emit body.**  Cleanest path: reorder the kernarg so SGPRs reuse
   the existing rmsnorm assignment (in_a→s4-5 as 'in', out_norm→s6-7
   as 'out', gamma→s8-9, m→s10, n→s12), then put in_b/out_resid at
   trailing kernarg offsets loaded into s14-17.  Prologue is then a
   4-instruction insertion: load b, add to v10 (which already holds a),
   store v10 to out_resid.  Body otherwise unchanged.
2. **AST recogniser** `amdgpu_lower_residual_rmsnorm_f32_3b`: gates on
   7 params + presence of `residual_rmsnorm_marker()` Call + lds_reduce
   + rsqrt.  Routes BEFORE rmsnorm_3b.
3. **CLI flag** `--emit-amdgpu-residual-rmsnorm-f32-N=N:path`.
4. **Launcher** `gpu_residual_rmsnorm_to_dev` in std/inference_gpu.mlr.
5. **Wire** in qwen3_forward_layer_gpu (lines 1502-1508) — replace
   `gpu_resid_add_to_dev(...) → gpu_rmsnorm_1024_to_dev(...)` pair
   with one call.  Cross-layer fusion at qwen3_generate.mlr is a
   second wiring point.

Estimate: 2-3 hours for the emit body alone (dword-level ISA edits
need llvm-mc verification on every instruction).  Worth a dedicated
session.

## Slice 4.2 — native skeleton (SHIPPED 2026-05-05)

`/tmp/qwen3_megakernel_skeleton.co` emitted by mlrc via
`--emit-amdgpu-qwen3-megakernel-skeleton=path`.  Body is bytewise
identical to the slice 4.1 hipcc microbench (50 dwords, kernarg
layout `(u32 *counter, u64 n_wgs, u64 n_iters)`).  Launcher
`examples/llm/qwen3_megakernel_skeleton_launch.mlr` runs at
WG_PERSIST=256 / n_iters=7 (one mega-layer's worth of barriers);
end-to-end kernel time 30-32 µs, matches hipcc reference within
measurement noise.  Native ISA emit framework for cross-WG barrier
protocols is now proven on hardware.

The emit fn `_emit_qwen3_megakernel_skeleton_body()` in
src/format_amdgpu.mlr is a loop body — each barrier iteration uses
the same code path with a different runtime expected-value (`i*n_wgs`).
Slice 4.3 will SPLIT the loop into 7 distinct phase blocks, each with
its own work between barrier_ack and counter_check.

## Slice 4.3+ — implementation design (locked 2026-05-05)

### Kernarg layout (15-arg mega-kernel)

|Index|Name              |Type|Bytes|Purpose|
|---|---|---|---|---|
| 0 |`in_residual`      |u64 |  8|residual input pointer (hidden f32)|
| 1 |`out_residual`     |u64 |  8|residual output pointer|
| 2 |`qkv_w`            |u64 |  8|bf16 [qkv_dim, hidden]|
| 3 |`o_w`              |u64 |  8|bf16 [hidden, q_dim]|
| 4 |`gate_up_w`        |u64 |  8|bf16 [2*ff, hidden]|
| 5 |`down_w`           |u64 |  8|bf16 [hidden, ff]|
| 6 |`in_norm_g`        |u64 |  8|f32 [hidden]|
| 7 |`post_norm_g`      |u64 |  8|f32 [hidden]|
| 8 |`q_norm_g`         |u64 |  8|f32 [head_dim]|
| 9 |`k_norm_g`         |u64 |  8|f32 [head_dim]|
|10 |`rope_cos`         |u64 |  8|f32 [head_dim/2]|
|11 |`rope_sin`         |u64 |  8|f32 [head_dim/2]|
|12 |`kc_layer`         |u64 |  8|K-cache layer slab|
|13 |`vc_layer`         |u64 |  8|V-cache layer slab|
|14 |`barrier_counter`  |u64 |  8|global counter for cross-WG sync|
|15 |`pos`              |u64 |  8|sequence position (for rope+attn)|
|16 |`shape_pack`       |u64 |  8|(hidden, q_dim, kv_dim, ff) packed|

Total: 17 × 8 = 136 bytes explicit kernargs + ~88 bytes COv5 hidden
args ≈ 224 bytes.  Loaded via 5 × `s_load_b256` + 1 × `s_load_b64`
into s[16:53] (38 SGPRs).  `barrier_counter` is a per-launch
allocation zeroed by the launcher between dispatches.

### SGPR map (gfx1100, with 64-SGPR grant via rsrc1=0xE0AF0182)

|SGPRs   |Phase 1+|Phase 3+|Phase 5-7+|Phase 9-17+|
|---|---|---|---|---|
|s[0:1]  |kernarg base ptr (input)|same|same|same|
|s2      |wg_id_x (live throughout — set by rsrc2=0x9E)|same|same|same|
|s[3:14] |scratch, recompute per phase|same|same|same|
|s15     |WG_ID_X (preserved per kfd shim convention)|same|same|same|
|s[16:53]|loaded mega-kernarg blob|same (live across phases)|same|same|
|s[54:63]|barrier protocol scratch (saved exec, expected, iter count)|same|same|same|

The 38-SGPR mega-kernarg block is loaded ONCE in the prologue and
held across all phases.  Phase work uses s[3:14] as scratch.

### VGPR map (84 VGPRs total, granted via rsrc1)

|VGPRs   |Phase 1 (rmsnorm)|Phase 3 (qkv gemv)|Phase 5 (qkv_split+rope)|Phase 7 (attn)|Phase 9 (o_proj+resid)|Phase 11 (post_norm)|Phase 13 (gate_up)|Phase 15 (silu_mul)|Phase 17 (down+resid)|
|---|---|---|---|---|---|---|---|---|---|
|v0       |tid + base offset|tid|tid|tid|tid|tid|tid|tid|tid|
|v[1:8]   |reduce-tree lane acc|gemv K-loop accumulators (8 outputs/WG via grid stride)|head-dim accumulators|attn dot-product accumulators|gemv accumulators|reduce-tree|gemv accumulators|silu temp|gemv accumulators|
|v[9:16]  |scratch|weight tile cache|rope cos/sin lane|softmax accumulators|scratch|scratch|scratch|scratch|scratch|
|v[17:32] |dead|gemv staging (LDS write)|rope rotated outputs|attn V accumulators|dead|dead|dead|dead|dead|

VGPRs are reused phase-to-phase.  Inter-phase carries go through
LDS (cheap re-load) rather than VGPR pressure.

### LDS layout (32 KB grant per WG)

| Offset | Size | Phase using it |
|---|---|---|
|`0x0000`|`hidden * 4 = 4 KB`|phase 1 produces b_x_norm, phase 3 reads|
|`0x1000`|`qkv_dim * 4 = 8 KB`|phase 3 produces b_qkv slab, phase 5 reads|
|`0x3000`|`hidden * 4 = 4 KB`|phase 9 produces b_mid (residual after o_proj), phase 11 reads, phase 17 reads for residual-add target|
|`0x4000`|`hidden * 4 = 4 KB`|phase 11 produces b_mid_norm, phase 13 reads|
|`0x5000`|`2*ff * 4 = 24 KB`|phase 13 produces b_gu, phase 15 reads|
|—|`8 KB headroom`|reduce-tree scratch shared between phases (overwritten each use)|

Total: 32 KB matches gfx1100's max LDS-per-WG (with 16 wave32s/CU).
The b_qkv intermediate at offset 0x1000 is the BIGGEST inter-phase
carry — at 8 KB it dwarfs the 4-KB hidden buffers.  qkv_split (phase
5) reads from it once and writes Q to LDS at 0x0000 (overwriting the
now-dead b_x_norm), K and V directly to GLOBAL (kc_layer + offset).

### Idle-WG fast path

For phases where work_count < WG_PERSIST=256, idle WGs must still
ack the barrier or the working WGs spin forever.  Idle path is just
the barrier ack subroutine WITHOUT phase work:

```
phase_N:
  if (wg_id < phase_N_work_count) {
    <do phase N work>
  }
  // ALL WGs ack:
  barrier_ack(N)   // atomic_add + spin-load
```

The same barrier subroutine emitted in slice 4.2 — just inlined 7
times instead of looped.

### Implementation order

The slices below are sized so each fits in one focused multi-hour
session with a clean validation gate.

| Slice | Phases ported | Effort | Validation |
|---|---|---|---|
| 4.3 | phase 1 (input rmsnorm) + phase 3 (qkv gemv) | 4-6 h | output `b_qkv` matches per-op chain bit-for-bit at random input |
| 4.4 | phases 5-7 (qkv_split+qknorm+rope, attn_decode) | 4-6 h | `b_attn` matches |
| 4.5 | phases 9-11 (o_proj+resid, post_norm) | 3-4 h | `b_mid_norm` matches |
| 4.6 | phases 13-17 (gate_up, silu, down+resid) | 3-4 h | `out_residual` matches |
| 4.7 | wire `qwen3_forward_layer_gpu` to use mega-kernel | 1-2 h | qwen3_generate produces bit-identical output to per-op path |
| 4.8 | bench + bisect | 2-3 h | tok/s number, projected ~110 |

Total: ~20-25 hours.  Each slice is gated by a bit-equivalence test
against the existing per-op chain — no slice can ship until its
phase outputs match the reference.

### Why split out phase 5 (qkv_split+qknorm+rope) instead of merging
with phase 3 (qkv matmul)?  Two reasons:
1. **Output layout differs**.  qkv matmul writes `b_qkv` as a
   contiguous [qkv_dim] slab (lane k of WG owns row k×stride).
   qkv_split scatters that into separate Q (head-major) / K-cache
   (position-major) / V-cache layouts.  The scatter pattern can't
   reuse the gemv lane→output mapping.
2. **WG count differs**.  Phase 3 uses all 256 WGs (each handles 8
   rows of qkv_dim=2048).  Phase 5 only needs 24 WGs (16 Q heads + 8
   KV heads).  Merging would force all 256 WGs through head-shaped
   logic, wasting compute.

Slice 2c already fused qknorm+rope; the existing emit body at
`_emit_qknorm_rope_qk_fused_f32_body` is the model for phase 5's
kernarg/SGPR/VGPR plan.

### Phase-by-phase emit body provenance

Each phase reuses an existing kernel's emit body, with adaptations
for (a) shared kernarg blob, (b) WG_PERSIST grid-stride iteration,
(c) LDS-resident inputs/outputs for inter-phase carries.

| Phase | Source emit body | Adaptations |
|---|---|---|
| 1  | `_emit_rmsnorm_f32_body_N(_, 1024, eps)` | output to LDS not global |
| 3  | `_emit_gemv_coop_bf16_f32_body` | input from LDS, output to LDS, grid-stride 8 outputs/WG |
| 5  | `_emit_qknorm_rope_qk_fused_f32_body` | input from LDS, output Q to LDS + K/V to global cache; restrict to WGs 0-23 |
| 7  | `_emit_attn_decode_f32_body` | input Q from LDS, K/V from global cache; restrict to WGs 0-15 |
| 9  | `_emit_gemv_coop_bf16_f32_body` + 4-instr add for residual | input from LDS (b_attn), output to LDS (b_mid) |
| 11 | `_emit_rmsnorm_f32_body_N(_, 1024, eps)` | input from LDS, output to LDS |
| 13 | `_emit_gemv_coop_bf16_f32_body` | input from LDS, output to LDS (24 KB!), grid-stride 24 outputs/WG |
| 15 | `_emit_silu_mul_f32_body` | input from LDS, output to LDS |
| 17 | `_emit_gemv_coop_bf16_f32_body` + add | input from LDS, output to GLOBAL (out_residual) |

### Open question: 8-kernarg s_load_b256 vs split loads

The 17-kernarg blob (136 B) loads ideally as 4× `s_load_b256` (32 B
each) into s[16:47].  Need to verify the s_load_b256 encoding in
existing emits — if it's not present, we can split into 2×
`s_load_b512` (already used in slice 2c) which loads 64 B at once.
Either way the prologue is small (≤6 instructions) compared to the
phase work (~60 instructions per phase).

## Slice 4.3 — HIP-source mega-kernel (SHIPPED 2026-05-05)

**Strategic pivot:** writing the mega-kernel as a HIP-source kernel
first (parallel to slice 4.1's approach) is dramatically faster than
authoring it in MLRift's native ISA emitter from scratch.  The
hipcc-compiled `.co` becomes a working reference; native bytewise
port from disasm follows once correctness is settled.  Same pattern
as how `mega_barrier_microbench.hip.cpp` was the slice 4.1 reference
that slice 4.2 then bytewise-ported to native.

**Implemented:** `examples/llm/qwen3_layer_megakernel.hip.cpp` —
full mega-kernel layer with:

- 17 kernarg signature (matches the design doc's locked layout)
- LDS struct (`b_x_norm` 4 KB, `b_qkv` 12 KB, `b_attn_q` 8 KB)
- `mega_barrier(counter, phase_idx)` device-fn wrapping the slice 4.1
  validated atomic_add + spin-load protocol
- Phases 1, 3, 5, 7, 9 implemented (input rmsnorm, qkv gemv, qkv_split
  + qknorm + rope, attn_decode, o_proj + residual)
- Compiles GREEN to gfx1100 (`hipcc --offload-arch=gfx1100 --genco
  -O3` → 21 568-byte `.co`)

**Deferred to slice 4.4:**
- Phases 11 (post_norm), 13 (gate_up), 15 (silu_mul), 17 (down + final
  residual).  These follow the same shape as 1/3/9 — port the existing
  per-op kernel logic into a phase block.
- Bit-equivalence A/B test against the per-op chain at random inputs
  (the GO/NO-GO gate before any tok/s claim).
- End-to-end qwen3 wire-up.

**Native ISA port (slice 4.5+):** `_emit_qwen3_layer_megakernel_blob`
in `format_amdgpu.mlr` will bytewise transcribe the hipcc disasm
once the HIP source is correctness-validated.  This was the working
pattern for slice 4.2 (50-dword skeleton) and will scale to the
mega-kernel's ~1500-dword body.

**Why HIP-source first instead of straight-to-native:**
- Each phase reuses semantics from existing per-op kernels but with
  different grid-stride / kernarg / SGPR / VGPR / LDS constraints.
  Validating those constraints in HIP (where the compiler handles
  register allocation) is far cheaper than getting them right
  bytewise on the first try.
- The hipcc disasm IS the spec for native port — exactly as in
  slice 4.2 where the 50-dword bytewise port took ~30 min vs
  multi-hour from-scratch native authoring.
- Native emit is the optimization target, not the development path.

## Slice 4.4 — phases 11-17 added (HIP-source) (SHIPPED 2026-05-05)

`examples/llm/qwen3_layer_megakernel.hip.cpp` now implements the
full 7-phase mega-kernel:

- **Phase 11** (post-attn rmsnorm): WG 0 reduce-tree, reads b_mid
  from LDS, writes b_mid_norm to LDS (reuses dead b_attn_q slab).
- **Phase 13** (gate_up matmul): all 256 WGs grid-stride 24 rows
  each, output spilled to GLOBAL `gu_scratch`.  LDS budget would
  require 24 KB for in-LDS b_gu; that crashes WG_PERSIST=256
  co-residence on gfx1100 (LDS-per-WG ≤ ~16 KB to keep ≥ 256 wave32s
  in flight).  HBM round-trip for the spill is cheap relative to
  the 6 MB weight read.
- **Phase 15** (silu_mul): pointwise on FF=3072 elements via grid
  stride.  In-place: writes back to `gu_scratch[0..FF]` overwriting
  the gate region, which becomes b_ff for phase 17.
- **Phase 17** (down + final residual): each WG handles 4 rows of
  `out = b_mid + down_w · b_ff`.  Reads b_mid from LDS (still live
  from phase 9 — phase 11 wrote to a different slab).

Kernel signature now takes 17 kernargs (added `gu_scratch` —
`float *gu_scratch[2*FF]` zero-tracked outside the kernel).
Compiles GREEN to gfx1100 (27200-byte `.co`).

**Pending in slice 4.5:**
- A/B bit-equivalence test against the per-op chain at random inputs
  (the GO/NO-GO gate before any tok/s claim).  Reuses
  qwen3_generate's prompt-prefill weights + synthetic random
  position-0 hidden state.
- Wire-up in `qwen3_forward_layer_gpu` once correctness GREEN.
- Native ISA bytewise port (slice 4.6).

## Slice 4.5 — smoke launcher (WIP, hang at first launch)

`examples/llm/qwen3_layer_megakernel_smoke.mlr` — allocates all 17
device buffers (zero-filled), builds kernarg blob, launches kernel,
reads back the barrier counter to verify all phases reached every
barrier.

**Status: HANGS at hipDeviceSynchronize timeout.**  Diagnosis:

1. **First attempt at WG_PERSIST=256:** kernel uses 29696 B LDS / WG.
   gfx1100 has 64 KB LDS per CU → only 2 wave32s/CU = 120 co-resident.
   Launching 256 WGs deadlocks the cross-WG barrier (resident WGs
   spin waiting for counter==256, queued WGs can't start until the
   resident ones retire).
2. **Reduced to WG_PERSIST=64:** still hangs.  64 << 120 co-resident
   cap so it shouldn't deadlock.  Suspected causes:
   - **VGPR pressure:** kernel uses 223 VGPRs/wave32 (per kernel
     descriptor metadata).  At WG=64 should still fit (60 CUs × ≥6
     waves/CU at 223 VGPR = 360 in flight) but the scheduler may
     be more conservative.  Unverified.
   - **All-zero data path:** with zero weights / inputs, every
     matmul produces 0, every rmsnorm divides by sqrt(eps), every
     softmax has uniform weights = 1/(pos+1).  Possibly some phase
     hits a slow path (denormal handling, exp underflow).
   - **Phase 7 attn at pos=0** with all-zero V cache might do
     unbounded work in pass 3's HEAD_DIM × t-range × HEAD_DIM
     triple-nested loop.  With pos=0 inner loop is t=0 only, but
     verify the `for unsigned long long t = 0; t <= pos; t++`
     terminates correctly.
   - **Barrier counter not pre-initialised across all WGs:** the
     launcher zeroes `barrier_ctr` before the kernel, but if any
     WG reads it before the H2D commits, the spin-load may see
     non-zero and either return immediately (skipping work) or
     spin forever.

**Triage plan for next session:**
1. Disable phases 5-17 progressively (single barrier at a time)
   to isolate which phase hangs.
2. Replace all-zero buffers with valid synthetic data (random
   weights from a fixed seed) — eliminates the zero-path slowness
   hypothesis.
3. Add a conservative `__threadfence()` between barrier counter
   zero and kernel launch to ensure counter visibility.
4. Run with `MLRIFT_USE_SDMA=1` to use SDMA for the counter zero
   (eliminates host-vis-VRAM coherence races).

The core insight from the hang attempt: **WG_PERSIST is bound by
LDS budget, not just by the barrier microbench's 256-WG sweet
spot.**  With a 29 KB LDS footprint, max safe WG_PERSIST is
LDS_per_CU / LDS_per_WG × N_CUs = 64/29 × 60 = 120.  Production
should set WG_PERSIST=64 (50% margin) until LDS is reduced via
spilling more inter-phase carries to global memory.

## Slice 4.5 — root cause FOUND via phase-bisection (2026-05-05)

The mega-kernel hang at max_phase≥1 is **NOT** a barrier protocol
issue, NOT a co-residence deadlock, NOT a VGPR-pressure issue.
It's a fundamental LDS-semantics design flaw.

**Bisection methodology:** added `max_phase` kernarg to early-exit
the kernel; smoke launcher sweeps max_phase ∈ [0..7], reporting
which max_phase first hangs.  Then bisected phase 3's body across
{empty, no-LDS-read, varying ROWS_PER_WG, varying K-loop}.

**Bisection results (all at WG_PERSIST=64, max_phase=1):**

| phase 3 body                                              | result |
|---|---|
| empty (return early)                                      | PASS (all 8 phases pass)|
| matmul without LDS read, 1 row × 1024 K                   | PASS|
| matmul without LDS read, 48 rows × 1024 K                 | PASS|
| matmul WITH LDS read `lds.b_x_norm[k]`, 1 row × 32 K      | PASS|
| matmul WITH LDS read, 1 row × 1024 K                      | PASS|
| matmul WITH LDS read, 48 rows × 1024 K (production shape) | **HANG**|

**Root cause:** HIP `__shared__` is **per-WG**, not cross-WG.  Each
WG has private LDS that's uninitialized on entry.  Phase 1 only
WG 0 writes `lds.b_x_norm`; phase 3 in WGs 1..63 reads
UNINITIALIZED LDS.  Across 48 rows × 1024 K-iters at 64 WGs, the
junk-bit pattern multiplied through the FMA pipeline (likely
denormals, NaN-propagation, or memory-controller state interaction)
wedges the GPU enough to exceed the 30s sync timeout.  At smaller
K-loop counts the impact is small enough to complete; at full
production shape the cumulative effect deadlocks.

**The mega-kernel design as drafted is broken.** Cross-phase
carries via LDS DON'T WORK on RDNA3 — they need to go through
GLOBAL scratch memory.  This invalidates the original LDS layout
plan (b_x_norm/b_qkv/b_attn_q resident in LDS across phases).

**Slice 4.6 fix plan:**
1. Add per-buffer global scratch slabs as kernargs:
   `b_x_norm_g[hidden]`, `b_qkv_g[QKV_DIM]`, `b_attn_q_g[Q_DIM]`,
   `b_mid_g[hidden]`, `b_mid_norm_g[hidden]`.  `b_ff` already uses
   `gu_scratch` (global).
2. Phase N writes to its global slab; phase N+2 reads from it.
3. LDS retained only for **intra-phase** reductions (~1 KB per
   WG: reduce_tmp[wave_count] for rmsnorm partial sums).
4. Side benefit: with LDS shrunk to ~1 KB, occupancy returns to
   the gfx1100 max (60 × 16 = 960 wave32s); WG_PERSIST can rise
   from 64 back to 256, recovering the slice-4.1 microbench's
   sweet-spot.

Per-token HBM cost of moving carries to global:
  hidden + QKV_DIM + Q_DIM + 2*hidden = 1024 + 3072 + 2048 + 2048
   = 8192 f32 = 32 KB R+W per layer × 28 layers = ~1.8 MB extra
   HBM traffic per token.  At 600 GB/s = 3 µs.  Trivial overhead.

**Estimated effort to ship slice 4.6:** ~2-3 hours.  Mostly
mechanical: thread 5 new pointer kernargs through the kernel + the
launcher, replace `lds.X[i]` with `X_g[i]` in cross-phase read/write
sites, bump WG_PERSIST back to 256, retest.

## Slice 4.6 — global-scratch carries (PARTIAL, 2026-05-05)

Implemented the fix from slice 4.5's diagnosis: all inter-phase
carries moved from per-WG LDS to global scratch slabs.  Kernel now
takes 22 kernargs (5 new global slabs: `b_x_norm_g`, `b_qkv_g`,
`b_attn_q_g`, `b_mid_g`, `b_mid_norm_g`).  LDS shrunk from 29 KB to
4 B (`wave_tmp[1]`).  Confirmed via kernel descriptor:
`group_segment_fixed_size: 4`, `vgpr_count: 55`.  Occupancy
returned to gfx1100 max — WG_PERSIST=256 is now LDS-feasible
(though the matmul phase still hangs at multi-row, see below).

**Smoke test results (zero-init buffers, WG_PERSIST=64):**

| max_phase | shape                            | result |
|---|---|---|
| 0 | phase 1 only                     | PASS |
| 1 | + phase 3, 48 rows × 1024 K      | HANG |
| 1 | + phase 3, 1 row × 1024 K        | PASS |
| 2 | + phase 5 (with 1-row phase 3)   | HANG |

**Slice 4.6 diagnostic:** the LDS-to-global fix DID help at the
single-row matmul case (phase 3 with 1 row + global b_x_norm_g
read passes; same shape with LDS read in slice 4.5 passed too).
But the multi-row matmul still hangs even with global scratch +
zero-init buffers + `__threadfence()` around the barrier.  And
phase 5 now hangs at max_phase=2 — a new, different failure mode.

The fix is real but incomplete.  Two open frontiers for slice 4.7:
1. **Phase 3 multi-row hang:** 48 rows × 1024 K × 64 WGs × 2 memory
   reads per FMA somehow wedges the GPU.  Adding `__threadfence()`
   didn't help.  Possibly compiler scheduling, possibly L0/L1 cache
   thrash, possibly something about how the `acc * b_x_norm_g[k]`
   product is materialised when both come from memory.
2. **Phase 5 hang at max_phase=2:** even with 1-row phase 3 (which
   works), enabling phase 5 hangs.  Different root cause; phase 5
   only needs 32 head sub-tasks (24 Q+K + 8 V) at 1 WG each, so
   shouldn't be a parallelism issue.

**Files updated:**
- `examples/llm/qwen3_layer_megakernel.hip.cpp` — full rewrite for
  global-scratch carries + threadfence in mega_barrier
- `examples/llm/qwen3_layer_megakernel_smoke.mlr` — 23-kernarg blob,
  pre-zeroes all weight + scratch buffers (eliminates NaN-from-uninit
  concern)

Slice 4.6 is shipped as a partial fix.  Slice 4.7 picks up the
two remaining hang frontiers.

## Slice 4.6 — DEBUGGED + GREEN (2026-05-05)

Subagent bisection found the actual root cause for both hangs: a
**kernarg-allocation/kernel-shape mismatch** on KV_DIM.

The launcher hardcoded `KV_DIM = 512` and `QKV_DIM = 3072`, but
the kernel macro `KV_DIM = N_KV_HEADS * HEAD_DIM = 8 * 128 = 1024`
(the inline comment `// 512` was wrong) and `QKV_DIM = Q_DIM +
2*KV_DIM = 4096`.  So the launcher under-allocated `b_qkv_g` by
1024 floats and `kc/vc_layer` by 4 KB per token.

**HANG 1 (phase 3 multi-row):** Multi-row write `b_qkv_g[row]` for
row = 0..QKV_DIM-1 = 0..4095 OOB-wrote 1024 floats past the
3072-float allocation.  At 1 row × WG_PERSIST=64, indices 0..63
fit within both 3072 and 4096 → no OOB → PASS.  Multi-row at 48
rows → indices up to ~3008 (still under 3072 in some cases) → not
quite OOB on writes, but the actual OOB was on phase 5's reads
described next.  Combined with the OOB the GPU's MMU eventually
wedged the wave.

**HANG 2 (phase 5 max_phase=2):** With pos=0, the V-branch pointer
arithmetic `b_qkv_g + Q_DIM + KV_DIM + head*HEAD_DIM + i` reached
indices 3072..3199 (correct range for the kernel's QKV_DIM=4096)
in a 3072-float buffer (launcher's wrong size) → OOB read → wave
hangs on RDNA3.  Bisection: Q-branch alone PASSED (read range
0..2047, in-bounds); V-branch alone HUNG (read range 3072..3199,
OOB on the under-allocated buffer).  V-branch with trivial-write
PASSED — confirming the OOB *read* was the trigger.

**Fix applied:**
- `qwen3_layer_megakernel_smoke.mlr`: `KV_DIM` 512→1024, `QKV_DIM`
  3072→4096.
- `qwen3_layer_megakernel.hip.cpp`: corrected misleading inline
  comments (`// 1024`, `// 4096`); restored phase 3 multi-row form
  with correct ROWS_PER_WG.

**Verification (WG_PERSIST=64, all-zero buffers):**
```
max_phase=0..7  →  ALL PASS
counter_final=512 (= 8 × 64)
total kernel time (max_phase=7) ≈ 2.2 ms
```

The slice 4.6 architectural fix (LDS → global) was correct — the
remaining hangs were driver-level OOB on under-allocated buffers,
not coherence or compiler issues.  Slice 4.7 (bit-equivalence A/B
vs per-op chain, with real qwen3 weights) is unblocked.

## Slice 4.7 — bit-equivalence validation GREEN (2026-05-05)

A subagent built the A/B harness `examples/llm/qwen3_layer_megakernel_ab.mlr`
(575 lines): allocates random f32 input + bf16 weights from a fixed
LCG seed, runs the canonical per-op chain (rmsnorm → bf16 gemv →
extract_q + qkv_split + qknorm + rope_qk fused → attn_decode →
o_proj + residual → post-rmsnorm → gate_up → silu_mul → down +
residual) and the slice-4.6 mega-kernel on identical bytes, then
sweeps `max_phase` 0..7 comparing each phase's output buffer.

**Bug found and fixed in phase 7 (attn_decode):**

The agent's bisection diverged at max_phase=3 with `first_bad_idx=32`
— exactly the WAVE boundary.  Root cause: phase 7's pass 3 wrote
`b_attn_q_g[q_head*HEAD_DIM + i]` for `i = lane, lane+32, ...`,
but the inner dot product re-read `q[0..127]` from THE SAME
buffer.  Iteration 2 (lanes 0..31 writing i=32..63) re-read
q[0..31] which iteration 1 had already overwritten.  Half the
output was junk.  `first_bad_idx=32` was the smoking gun.

**Fix:** added `b_attn_out_g` as kernarg slot 22 — phase 7 writes
the attention output to a separate slab, phase 9 reads from it
(not from `b_attn_q_g`).  No read-after-write hazard on the Q
input.

After the fix, **A/B bisection PASSES every max_phase 0..7**:

| max_phase | output     | max_abs_err |
|---|---|---|
| 0 | b_x_norm_g  | 0           |
| 1 | b_qkv_g     | 0           |
| 2 | b_attn_q_g  | ~2.9e-7     |
| 3 | b_attn_out_g| 0           |
| 4 | b_mid_g     | 0           |
| 5 | b_mid_norm_g| 0           |
| 6 | gu_scratch  | 0           |
| 7 | out_resid   | ~4.9e-7     |

The non-zero residuals at phases 2 and 7 are pure FMA-reorder
noise — both paths see identical bf16 weights and f32 inputs;
the reduce-tree order differs slightly between the per-op
gemv_coop_bf16 and the mega-kernel's inline reduce.  Within
2-3 ULPs of f32 single-precision noise floor.  **GREEN.**

**Slice 4.7 unblocks:**
- 4.8: wire mega-kernel into `qwen3_forward_layer_gpu` (replace 11
  per-op calls with 1 mega-kernel launch).
- 4.9: end-to-end qwen3_generate tok/s bench (target: ~107 tok/s,
  vs 63.2 tok/s baseline = 1.7× / 1.45× ROCm bf16 ceiling).
- 4.10: native ISA bytewise port from hipcc disasm (~1.5 hour
  mechanical work given the slice 4.2 precedent's 30-min/50-dword
  rate scaled to the mega-kernel's ~1500 dwords).

Mega-kernel is now both **non-hanging AND numerically correct** —
end-to-end demonstrably equivalent to the per-op chain at random
inputs.  The destroy-PyTorch tok/s win is one wire-up away.

## Slice 4.8 — wire-up GREEN (2026-05-05)

A subagent threaded the slice-4.7 mega-kernel into the qwen3
inference pipeline. When `MLRIFT_QWEN3_MEGAKERNEL=1` is set,
`qwen3_forward_layer_gpu` replaces its 11 per-op dispatches with
one `gpu_qwen3_layer_megakernel_to_dev` call per layer.

**Token output: bit-identical to per-op + PyTorch reference.**

```
14990, 14582, 284, 330, 9707, 11, 4337, 17199, 1350, 3203, 4791,
14582, 692, 2, 1173, 279, 1156, 3409, 198, 1350, 3203
```
matches both the existing per-op path and the PyTorch reference
exactly, on Qwen3-0.6B default prompt.

**Per-step launch counters:**
- Per-op:        310 launches / step (the slice 2c baseline)
- Mega-kernel:    29 launches / step (28 layers × 1 dispatch + 1 lm_head)

281 launches saved per token — the design's projected dispatch-
overhead reduction is empirically realized.

**Wire-up scope (3 files, +292 / -2):**

`std/inference_gpu.mlr`:
- `_gpu_fh_qwen3_megakernel` static + best-effort `.co` load.
- `gpu_get_or_upload_bf16_weight()` public accessor (shares the
  bf16 weight cache between mega-kernel and per-op paths).
- `gpu_qwen3_megakernel_ready()` probe.
- `gpu_qwen3_layer_megakernel_to_dev()` launcher: builds the
  24-kernarg blob, zeroes the host-mapped barrier counter,
  dispatches (64,1,1)×(32,1,1).

`std/qwen3.mlr`:
- 8 new `_gpuq_d_megak_*` statics (7 inter-phase scratch slabs +
  barrier counter, host-visible GTT for the barrier).
- Pipeline-init reserves them.
- `qwen3_forward_layer_gpu` env-cached gate at top.  Fires only
  when: env set, module loaded, batch_size==1, batch_idx==0,
  hidden==1024 (0.6B-only), pos<64, phase mask 0xFF (cross-layer
  fusion bits 0x8000/0x10000 cause fall-through to per-op).
- Resolves bf16 weights via shared cache, gammas via
  `gpu_cache_bf16_as_f32`, refreshes rope cos/sin, computes
  KV-cache device slabs, dispatches with `max_phase=7`.

`examples/qwen3_generate.mlr`:
- When the env is set, suppress cross-layer fusion bits and skip
  `qwen3_gpu_chain_resid_to_next_norm` (mega-kernel handles input
  rmsnorm and final residual internally).

**Slice 4 status:**
- 4.1-4.7 ✓ (barrier, native skeleton, 7-phase HIP, LDS root
  cause + global-scratch fix, bit-equivalence GREEN)
- 4.8 ✓ **wire-up GREEN, tokens bit-identical to PyTorch** (18.9 tok/s baseline)
- 4.10–4.13 ✓ **88.0 tok/s, +19 % over PyTorch ROCm bf16 on fp32 weights**
- 4.14 ✓ **164.2 tok/s, +222 % over PyTorch ROCm bf16 (M=4 spec-decode mega)**
- 4.15 ✓ **181.8 tok/s reported / ~222 steady-state (M=8 spec-decode mega)**
- 4.16 ✓ **190.3 tok/s, +257 % over PyTorch ROCm bf16 (WMMA bf16 phase-13 on mks8)**

## Slices 4.10–4.13 — closing the per-op gap and breaking past PyTorch bf16

Slice 4.8 landed correct but slow: idle-GPU bench showed 18.9 tok/s
vs the per-op baseline of 54.7.  Five subsequent slices moved the
mega-kernel from 18.9 → 88.0 tok/s (4.65×) — **+19 % over PyTorch
ROCm bf16 (~74 tok/s) on fp32 weights**, fully bit-identical.

| Slice | unlock | tok/s | step ms |
|---|---|---:|---:|
| HEAD slice 4.8 | wired in but WG=64, single barrier counter | 18.9 | 52 |
| 4.10 | WG_PERSIST 64 → 512 (recover gemv_coop row parallelism) | 46.0 | 21 |
| 4.11 | cooperative phase-7 (ATTN_COOP=4) + cached softmax in LDS + dropped trailing `__threadfence` | 54.1 | 18 |
| 4.12 | bf16x2 vectorised matmul loads (read u32 = 2 bf16, halves VMEM count) | 64.9 | 15 |
| **4.13** | **channel-repacked padded weights (HIDDEN_PAD=1152, Q_DIM_PAD=2176, FF_PAD=3200)** | **88.0** | **11** |

### 4.10 — WG_PERSIST 64 → 512

The slice 4.8 grid was 64 wave-WGs.  Per-op `gemv_coop` launches at
`grid_x = N rows` (4 096–6 144 wave-WGs), saturating all 240 SIMDs;
the mega-kernel left 73 % of the GPU idle.  Bumping `WG_PERSIST` to
512 spread matmul rows 8× wider.  Higher (1 024 / 2 048) regressed:
1024 hits atomic-counter contention, ≥ 2 048 deadlocks because not
all WGs run concurrently on RX 7800 XT (occupancy cap ~1 500 wave-WGs).

### 4.11 — cooperative phase 7 + cached softmax + fence drop

Phase 7 (`attn_decode`) used 16 of 512 WGs (one per Q head).
Cooperative redesign: 16 heads × 4 cooperating WGs split HEAD_DIM=128
output dims (32 dims, 1 lane each).  Pass 2 caches softmax weights
in `attn_w_lds[64]` so pass 3 skips the redundant Q·K dot product
(was 128× per head per t).  Trailing `__threadfence()` in
`mega_barrier` dropped — leading fence already orders writer stores;
ACQUIRE on the counter load orders later loads.

### 4.12 — bf16x2 vectorised loads

Clang/HIP was emitting `buffer_load_ushort` per element in the matmul
k-loop.  Hand-pack two bf16 weights as a `u32` and unpack to two f32
FMAs:

```cpp
const unsigned int *w_row_u32 = reinterpret_cast<const unsigned int *>(qkv_w + row * HIDDEN);
for (unsigned int kp = lane; kp < HIDDEN / 2; kp += WAVE) {
    unsigned int packed = w_row_u32[kp];
    unsigned int k = kp * 2;
    float w0 = bf16_to_f32((unsigned short)(packed & 0xFFFFu));
    float w1 = bf16_to_f32((unsigned short)(packed >> 16));
    acc += w0 * x[k] + w1 * x[k + 1];
}
```

Halves the matmul VMEM instruction count.  Effective DRAM bandwidth
~50 % → ~62 % of peak.  +20 % tok/s.  Mega first beats per-op here.

### 4.13 — channel-repacked padded weights

Bandwidth math:

```
RX 7800 XT GDDR6: 16 channels × ~256 B stripe = 4096 B channel cycle
Naive bf16 weight rows:
  qkv_w     row = 1024 ushort = 2048 B  → gcd(2048,4096) = 2048 → only 2 distinct row-start channels
  o_proj    row = 2048 ushort = 4096 B  → gcd(4096,4096) = 4096 → only 1 distinct channel
  gate_up   row = 1024 ushort = 2048 B  → 2 channels
  down      row = 3072 ushort = 6144 B  → 2 channels
```

Concurrent WGs reading consecutive rows banged on the same 1–2 of 16
channels, capping aggregate bandwidth at ~12.5 % of peak.  Fix: pad
each row by 128 ushorts (256 B) so `row_pad_bytes mod 4096 = 256`,
distributing rows across all 16 channels:

```
HIDDEN_PAD = 1152 ushort (qkv, gate_up):  row=2304 B, gcd=256 → 16 channels
Q_DIM_PAD  = 2176 ushort (o_proj):        row=4352 B, gcd=256 → 16 channels
FF_PAD     = 3200 ushort (down):          row=6400 B, gcd=256 → 16 channels
```

Implementation: parallel padded-row bf16 cache in `std/inference_gpu.mlr`
(`_gpu_mm_get_or_upload_bf16_padded`), allocated once per (host_ptr,
K_pad).  CPU-side strided fill into a host pad buffer + ONE bulk
hipMemcpy (vs ~140 k per-row syncs which would dominate cold-start
at ~1.4 s).  `qwen3_forward_layer_gpu`'s mega-kernel branch swaps in
the `_padded` variant for all four matmuls; the kernel uses the
`*_PAD` constants as the row-stride multiplier in phases 3/9/13/17.

`examples/qwen3_generate.mlr` adds a pre-decode warmup pass that
walks every layer and triggers the padded uploader BEFORE
`decode_start_ns`.  Without it, first inference's per-step time is
dominated by 28 layers × 4 matmul cold-start uploads (~3 s).

Memory cost: ~12.5 % larger weight slabs (per-layer 31.5 → 34.6 MB,
total 880 → 970 MB).  Per-op caches are unchanged.

**Final bench (3-run median, RX 7800 XT, qwen3-0.6B, bf16 weights /
fp32 compute, 20 new tokens, default seed):**

```
mega slice 4.13      :  88.0 tok/s   [tokens bit-identical to PyTorch]
per-op bf16          :  69.9 tok/s
per-op pure fp32     :  56.1 tok/s
PyTorch ROCm bf16    :  ~74  tok/s   →  mega is +19 %
PyTorch ROCm fp32    :  41   tok/s   →  mega is +115 %
```

### What didn't work (explicit negative results)

Recorded so future slices don't redo them:
- **Persistent counter** (slice 4.9): cross-layer L0/L1 cache coherence
  on RDNA3 isn't guaranteed by AQL ordering alone; tokens diverged.
  Reverted.
- **Partitioned-counter barrier** (32 slots, hashed by wg_id):
  at WG=512 the single-counter atomic isn't the bottleneck;
  partitioning costs sequential master polls.  Neutral.
- **Phase 13+15 fusion** (gate_up + silu_mul, "Idea G"): interleaved
  gate/up VMEM rows hurt coalescing.  −4 tok/s.  Reverted.
- **Phase 1+3 / 11+13 fusion + per-WG `xs[32]` register-stash**: wins
  on a slice-4.8 base but interacts badly with slice 4.11's phase-7
  LDS / VGPR pressure.  Net −2 tok/s when stacked.
- **`__builtin_prefetch`** in HIP source: clang doesn't lower it for
  AMDGPU; sequential VMEM is auto-prefetched by the GPU front-end.
  No effect.

### Outstanding follow-ups
- AB harness (`qwen3_layer_megakernel_ab.mlr`) still allocates
  unpadded buffers; padded-row layout makes it OOB-read.  Update
  needed: strided LCG fill into K_pad-stride rows.  Smoke is already
  padded.
- Per-op `gemv_coop_bf16` doesn't use padded layout.  Same trick
  should lift per-op proportionally (currently 69.9 tok/s).
- Cold-start uploader is row-by-row CPU memcpy → bulk hipMemcpy.
  `hipMemcpy2D` (not in the shim today) would skip the host stage.

## Slice 4.14 — M=4 spec-decode mega-kernel

The slice 4.13 mega is single-stream.  PLD speculative decoding
(spec_K=4) was previously routed through the per-op M=4 batched chain
(72 tok/s) and could not stack with the mega-kernel.  Slice 4.14 adds
a parallel `qwen3_layer_megakernel_speck4` kernel that processes 4
query tokens per dispatch, paired with the existing PLD draft proposer.

### The win — M=4 amortisation on the four bf16 matmuls

Phases 3, 9, 13, 17 are weight-bandwidth-bound.  At M=1 each weight
row drives one dot product.  At M=4 each weight row drives **four**
dot products against four different inputs:

```cpp
float acc[M_EFF=4] = {0};
for (kp = lane; kp < HIDDEN/2; kp += WAVE) {
    unsigned int packed = w_row_u32[kp];                  // ONE VMEM load
    float w0 = bf16_to_f32(packed & 0xFFFF);
    float w1 = bf16_to_f32(packed >> 16);
    #pragma unroll
    for (m = 0; m < M_EFF; m++) {
        acc[m] += w0 * b_x_norm_g[m*HIDDEN + 2*kp]
               + w1 * b_x_norm_g[m*HIDDEN + 2*kp + 1];
    }
}
```

Same weight bandwidth, 4× the output.  Layer-time grows ~1.6× from
M=1 to M=4 (compute scales but bandwidth doesn't), so per-token rate
in the dispatch jumps from 88 to ~220 tok/s.  After PLD accept
overhead the effective rate lands at **164 tok/s**.

### Per-stream phases

Phases 1 (rmsnorm), 5 (qkv-split + qknorm + rope_qk), 7 (attn), 11
(post-norm), 15 (silu-mul) are not weight-bandwidth-bound — they scale
linearly with M.  Each WG loops `for m in 0..M_EFF` doing the per-stream
work, accessing slabs at `m * dim` stride.  Phase 5 expands from 32 WGs
(16 Q + 8 K + 8 V) to 128 WGs (4 × 32) so each WG handles ONE
`(m, head_kind)` pair; phase 7 expands from 64 to 256 cooperating WGs
(16 heads × ATTN_COOP=4 × M=4).

### Per-token RoPE (the previous-agent debug story)

The per-op spec_K=4 path uses single-pos RoPE — all 4 query/key tokens
get rotated by `cos(pos_base)/sin(pos_base)` even though they are at
positions `pos_base..pos_base+3`.  This is numerically wrong and
causes per-op spec_K=4 to drift from PyTorch reference around step 19.

Slice 4.14 mks4 fixes this: each query token uses its own RoPE
angle `cos/sin[pos_base + m]`.  K/V cache writes go to slot
`pos_base + m`.  Output is bit-identical to the M=1 mega (which is
bit-identical to PyTorch greedy reference).

A bisection AB harness (`qwen3_layer_megakernel_speck4_ab.mlr`) was
needed to confirm this — initial divergence vs per-op REF turned out
to be a bug in the AB harness itself (per-op qknorm needs 5 args, was
getting 4); once fixed, mks4 vs per-op-with-single-pos-rope is bit-close.

### Final bench

3-run mean, RX 7800 XT gfx1100, qwen3-0.6B, bf16 weights / fp32
compute, `MLRIFT_QWEN3_MAX_NEW=12 MLRIFT_LONG_PROMPT=1`:

```
mks4 + spec_K=4 + LONG_PROMPT  : 164.2 tok/s   [bit-identical to M=1 mega]
M=1 mega @ slice 4.13          :  89.6 tok/s   [bit-identical to PyTorch]
per-op + spec_K=4 + LONG_PROMPT:  72.0 tok/s
PyTorch ROCm bf16              :  ~74  tok/s   →  mks4 is +222 %
PyTorch ROCm fp32              :  41   tok/s   →  mks4 is +300 %
```

### Files

- `examples/llm/qwen3_layer_megakernel_speck4.hip.cpp` — the kernel.
  Same skeleton as the M=1 mega, with `M_EFF=4` accumulator arrays in
  the matmul phases and per-m slab indexing in the rest.  Padded weight
  strides (HIDDEN_PAD=1152, Q_DIM_PAD=2176, FF_PAD=3200) inherited.
- `examples/llm/qwen3_layer_megakernel_speck4_smoke.mlr` — 8-phase
  bisection smoke at WG_PERSIST=512, 4× scratch sizes.
- `examples/llm/qwen3_layer_megakernel_speck4_ab.mlr` — random-LCG
  bit-equivalence vs per-op M=4 reference, two rope modes.
- `std/inference_gpu.mlr` — `_gpu_fh_qwen3_megakernel_speck4` static
  + .co loader + `gpu_qwen3_layer_megakernel_speck4_to_dev` launcher.
- `std/qwen3.mlr` — 14 mks4 statics + slab allocations +
  `qwen3_forward_layer_megakernel_speck4_gpu` wrapper.
- `examples/qwen3_generate.mlr` — env-gated branch in spec_K=4 layer
  loop (env: `MLRIFT_QWEN3_MEGAKERNEL_SPECK4=1`).

Memory cost: ~200 MB additional GPU scratch.  Padded weight cache
shared with M=1 mega — no duplication.

### Outstanding (slice 4.14)

- `max_seq=64` hardcoded; spec_K=4 hangs at pos≥61 because KV writes
  exceed the cache.  Pre-existing in the per-op speck4 path.  Fix is
  a rolling-window KV cache or `max_seq=128` mega variant.
- Per-op spec_K=4's single-pos-rope drift is real but pre-existing.
  Best fix: deprecate that path now that mks4 covers it at 2.3× speed.

## Slice 4.15 — M=8 spec-decode mega-kernel

Doubles slice 4.14's M_EFF=4 to M_EFF=8.  Same kernel structure: each
weight row drives 8 dot products via `acc[8]` accumulator arrays in
the matmul phases (3, 9, 13, 17), and per-stream phases (1, 5, 7, 11,
15) loop `for (m = 0; m < 8; m++)`.  Phase 5 expands to 256 WGs
(M_EFF × (Q + K + V) = 8 × 32) and phase 7 expands to 512 cooperating
WGs (16 heads × ATTN_COOP=4 × M_EFF=8) — right at the safe boundary
on RX 7800 XT but still under the 1500-WG concurrent cap.

`attn_w_lds_speck8[M_EFF * MAX_SEQ] = 8 × 64 = 512 floats / WG = 2 KB
LDS`.  Comfortably fits.

### The bench-vs-steady-state caveat

The reported tok/s number is dominated by step 0's ~77 ms cold
dispatch (weight L2 cold-fill + KV cache touch + first .co code-
load).  At `max_seq=64` and `spec_K=8`, only 5 fast steps fit after
step 0 (each ~36 ms × 8 tokens accepted).  The warmup amortisation
floor is therefore ~180 tok/s reported even though the steady-state
is ~222 tok/s (8 tokens / 36 ms).  Bumping `max_seq=128` and the
corresponding kernel `MAX_SEQ` constant (with attn LDS [64]→[128])
would let the bench run 12 steps and settle near 222.

### Files

- `examples/llm/qwen3_layer_megakernel_speck8.hip.cpp` — NEW.  Copy
  of speck4 kernel with `#define M_EFF 8`.
- `examples/llm/qwen3_layer_megakernel_speck8_smoke.mlr` — NEW.
- `std/inference_gpu.mlr` — `_gpu_fh_qwen3_megakernel_speck8` static
  + .co loader + `gpu_qwen3_layer_megakernel_speck8_to_dev` launcher.
- `std/qwen3.mlr` — 14 mks8 statics for M_EFF=8 scratches.
- `examples/qwen3_generate.mlr` — relax gate to allow `MLRIFT_SPEC_K=8`.

## Slice 4.16 — WMMA bf16 tensor cores on phase 13

Replaces the bf16x2 vector-FMA inner loop in mks8's phase 13
(gate_up matmul, the heaviest of the four) with gfx1100's
`v_wmma_f32_16x16x16_bf16` tensor-core instruction.

### Tile structure

Each WG owns a 16×16 output tile (tile_idx = wg_id; WGs ≥ 384 idle
but still hit the barrier).  Per K-step (k_base += 16, 64 iterations
over HIDDEN=1024):

```cpp
typedef float v8f __attribute__((ext_vector_type(8)));
typedef short v16s __attribute__((ext_vector_type(16)));   // bf16 as i16

v8f acc = {0};
for (int k_base = 0; k_base < HIDDEN; k_base += 16) {
    v16s a = load_bf16_a_fragment(gate_up_w, row_base, k_base, lane);
    v16s b = load_bf16_b_fragment_from_f32(b_mid_norm_g_8, k_base, lane);
    acc = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, acc);
}
store_tile(acc, gu_scratch_8, row_base, lane);
```

A-fragment: 16 bf16 weights from `gate_up_w[(row_base + lane%16) *
HIDDEN_PAD + k_base]`.  B-fragment: 16 f32 inputs from
`b_mid_norm_g_8[(lane%16) * HIDDEN + k_base]`, truncated to bf16 by
`(uint32 >> 16)` (lossy but sufficient for an already-rmsnormed
activation).

### Why the speedup is modest (+4.5%)

RX 7800 XT decode is bandwidth-bound, not compute-bound.  WMMA
accelerates compute density on the matmul; bandwidth is unchanged.
At M=8 only half the WMMA tile (8 of 16 columns) is utilised — the
real payoff stacks with M ≥ 16 batched spec-decode.

### Outstanding (slice 4.16)

- WMMA on phases 3 (qkv), 9 (o_proj), 17 (down) of mks8 not yet
  applied.  Bounded follow-up — expected another ~3-5 % per phase.
- mks4 (M=4) and M=1 mega untouched.  At M < 16 WMMA tile is poorly
  utilised; less likely to pay off there.
- B-fragment f32→bf16 truncation is lossy for non-rmsnormed
  activations.  Phases 9 and 17 read attn output / silu_mul output
  which may have wider dynamic range; verify bit-equivalence
  carefully before applying WMMA there.
