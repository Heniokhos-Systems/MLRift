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
