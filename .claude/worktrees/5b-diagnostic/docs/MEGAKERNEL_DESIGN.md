# Mega-kernel design: persistent per-layer chain (Qwen3-0.6B decode)

**Status:** design (no code yet).  Drafted 2026-05-01 after task #106
audit + #107 AQL-chain dead-end.

## Goal

Reduce the 17 ms / step at the bf16-batched + spec_K=4 + LONG config to
~10 ms by fusing the per-layer decode chain into a single kernel.
Projected ceiling: **~144 tok/s** (from current 92 tok/s).

## Why this is the only remaining lever

Audit (#106) found:

| Bucket | ms/step | Source |
|---|---|---|
| Sync wait | 3.3 | 32 syncs × 102 µs floor |
| Launch host overhead | 3.0 | 592 launches × 5 µs KFD shim |
| GPU compute | 10.7 | actual kernel time |
| **total** | **17.0** | |

AQL chain (#107) tried to collapse syncs/launches without changing
kernels; it serialized host-prep with GPU compute and made things
worse.  The per-layer flush turned out to be **load-bearing for
correctness** (skipping it crashes spec accept rate 2.05 → 1.35
tok/step from numerical drift across kernel boundaries).

So: launch reductions only count if the kernel does the per-layer
work as one atomic unit — i.e. mega-kernel.  Anything short of that
either hits the drift problem or doesn't actually save sync time.

## Per-layer work to fuse

For one layer at M_eff=4 (spec_K=4):

```
input_norm(x_in, in_norm_w)              →  x_norm
qkv_proj(x_norm, q_proj_w)               →  qkv          [  4 × 4096 ]
qkv_split(qkv)                           →  q,  k_slot, v_slot
qknorm_q(q, q_norm_w)                    →  q_norm
qknorm_k(k_slot, k_norm_w)               →  k_norm  (writes k_slot in place)
rope_q(q_norm, cos, sin)                 →  q_rope
rope_k(k_norm, cos, sin)                 →  k_slot  (in place)
attn_decode(q_rope, k_cache, v_cache)    →  attn
o_proj(attn, o_proj_w)                   →  o
resid_add(x_in, o)                       →  mid          [ 4 × 1024 ]
post_norm(mid, post_norm_w)              →  mid_norm
gate_up(mid_norm, gate_proj_w)           →  gu           [ 4 × 6144 ]
silu_mul(gu)                             →  ff           [ 4 × 3072 ]
down_proj(ff, down_proj_w)               →  ff_out       [ 4 × 1024 ]
resid_add(mid, ff_out)                   →  x_out        [ 4 × 1024 ]
                                                     (carry to next layer)
```

15 logical phases × 28 layers × 1 sync = 420 launches + 28 syncs
just for the layer sweep.  The audit's 592/step includes lm_head,
final_norm, embed, prefix-postfix bookkeeping.

## The shape problem

Each phase has a different ideal (block, grid) shape:

| Phase | Ideal block | Ideal grid |
|---|---|---|
| Matmul (cooperative gemv) | 32 lanes | N rows (1024..6144) |
| RMSNorm (hidden=1024) | 1024 lanes | M_eff streams |
| RMSNorm (head=128) | 128 lanes | M_eff × n_heads |
| RoPE | 128 lanes | M_eff × n_heads |
| QKV-split | 128 lanes | 8 × M_eff |
| Attn-decode (fused per-head) | 32 lanes | n_heads × M_eff |
| Silu-mul | 256 lanes | ff/256 × M_eff |
| Resid-add | 256 lanes | hidden/256 × M_eff |

A single kernel can have only ONE (block, grid).  Pick wrong → wasted
lanes during phases that don't need them, OR not enough lanes for
phases that do.

## Three architectural options

### Option A: persistent block=1024, grid=1, layer-loop inside kernel

One WG of 1024 lanes processes all 28 layers sequentially.  The 1024
lanes are repurposed each phase.

**Lane utilization across phases (for hidden=1024, M_eff=4):**

| Phase | Useful lanes | Total | Util |
|---|---|---|---|
| input_norm (each stream) | 1024 | 1024 | 100% |
| qkv_proj per row | 32 (cooperative) | 1024 | 3% |
| Attn 16 Q heads (32 per-head) | 16×32=512 | 1024 | 50% |
| silu_mul over 4 streams × 3072 | 1024 | 1024 | 100% (do 12288/iter)|

The matmul phases waste 30/32 lanes per WG.  But the lane-utilization
math is misleading — a "wasted" lane is still doing 1 fmac per cycle;
what matters is whether the GEMV compute is bandwidth- or
compute-bound.  At hidden=1024, K=1024 we're memory-bound; lanes
aren't the bottleneck.

**Pros:**  Eliminates ALL 27 inter-layer launches (one kernel runs
the whole sweep).  Eliminates the per-layer flush — there are no
inter-kernel boundaries for state to drift across.

**Cons:**  Single WG = single CU = ~1/60 of the GPU.  Throughput
bottleneck.

### Option B: persistent grid=N, block=32, one-WG-per-output-row

Each WG handles one *output element* of the longest fan-out matmul.
Grid = max(qkv_dim, gate_up_dim) = 6144.  Each WG does its row's
work for matmul, then *cooperates with other WGs via global memory*
for the cross-row phases (rmsnorm reduction, attn softmax).

**Pros:**  Massively parallel (6144 WGs spread across all 60 CUs).
Matches the existing gemv_coop kernels' block=32 cooperative reduce.

**Cons:**  Cross-WG communication via global memory means an
**implicit barrier** between phases, which on AMD GPUs requires...
either a separate kernel (defeats the point) or `s_barrier_global`
which only fences within a workgroup, not across.  AMD has no GPU-
wide CTA barrier short of kernel-end semantic.

**Verdict:** Option B doesn't work without GPU-wide sync, which AMD
RDNA3 doesn't provide.  Discarded.

### Option C: persistent grid=NUM_CUs, block=1024, work-stealing

NUM_CUs × 1 WG per CU, each WG persistent for the whole layer sweep.
Phases that have inherent cross-CU dependencies (rmsnorm reduction
over hidden=1024) require a barrier, which we get via... again, no
GPU-wide barrier.

**Verdict:** Same blocker as B.  Discarded.

## Conclusion: Option A is the only viable shape

block=1024, grid=1, persistent across all 28 layers.  Throughput is
limited by 1 CU (not full GPU), but *we already have that limit* —
the audit shows GPU compute is 10.7 ms / step at full GPU usage,
and Option A's compute scales with that.  The win is in collapsing
all the launches into one persistent kernel.

Wait — that doesn't add up.  If we currently use the full GPU and
get 10.7 ms compute, and Option A uses 1 CU, compute would balloon
to 60 × 10.7 = 642 ms, totally wrecking the win.

So Option A as written is wrong.  Re-think.

## The actual viable architecture: phase-fused per-WG

Don't try to fuse all 28 layers — fuse the **inter-phase boundaries
inside one layer's pipeline** while keeping the SAME parallelism
the existing kernels already have.

Specifically: keep `block=32, grid=N` (the cooperative-gemv shape)
for the matmul phases, but write a kernel that *doesn't return*
between phases.  Inter-phase data moves through LDS or VRAM as
today; the "return + relaunch" is what's removed.

Per-layer: launch ONE kernel with grid=qkv_dim that does:
1. qkv_proj cooperative matmul (block=32, grid=qkv_dim) → write qkv to VRAM
2. (kernel-end semantic: ALL WGs done with phase 1)
3. ... but step 2 IS a kernel boundary — no GPU-wide barrier exists.

So even within one layer, multi-phase fusion requires kernel re-launch
*or* picking ONE phase shape and stretching others to fit.

**This is the exact same problem as Option A/B/C, just at a smaller
scale.**

## What's actually achievable

Given AMD GFX11 has no GPU-wide barrier, fusion can only happen WITHIN
one workgroup's compute sphere.  Useful fusions:

1. **`resid_add + rmsnorm`**: same WG handles both for one stream's
   row.  Block=hidden=1024.  Saves one dispatch.
2. **`qknorm_q + rope_q`**: per-head WG does both.  Block=128.
   Already similar shape; saves one dispatch.
3. **`silu_mul + down_proj` tail**: gate_up writes to LDS, silu_mul
   computes from LDS, down_proj reads from LDS.  Saves two
   dispatches.
4. **`o_proj + resid_add + post_norm`**: matmul output goes
   directly into resid+norm in same WG.  Hard because matmul has
   block=32 grid=hidden, while resid+norm wants block=hidden grid=1.

## Realistic launch-count target

Current per layer (M_eff=4): 15 dispatches.
Best fusion within shape constraints:  ~9 dispatches per layer.
Reduction: 9/15 × 420 = 252 dispatches per token (vs 420).
Sync count probably stays at ~28/token (1 per layer flush).

Time savings:
- Launch host: -168 launches × 5 µs = **-0.84 ms / step**
- Sync wait: unchanged

Total win: 17 → 16 ms / step → 99 tok/s (+8 % over 92).

**This is much smaller than the audit's +52 % projection.**  The
audit assumed all 592 launches and 32 syncs were savable; in reality
only the *intra-WG* fusions are without GPU-wide barrier hardware,
and the per-layer flush stays for correctness.

## The right next step

(1) Confirm AMD GFX11 indeed has no GPU-wide barrier — search for
`s_barrier_global`, `s_dcache_inv_vol`, or any signal-based ring
barrier pattern that could synchronize across WGs without a
kernel boundary.  If found, the design opens up significantly.

(2) Otherwise, accept the +8 % ceiling and pick the cheapest fusion
to implement first as proof-of-pattern: **resid_add + rmsnorm fuse**
(slice 1 from the original memo).  Saves 2 dispatches × 28 layers
× 20 steps = 1120 dispatches per benchmark run = ~5.6 ms total
wallclock = +1 % perf.

(3) Or: focus on reducing per-launch host µs (KFD shim hot path)
and per-sync µs (signal floor).  Both are ~5 µs and ~102 µs
respectively today; halving either gives ~10 % perf.

## Open question for the user

The mega-kernel ceiling on this hardware is ~+8 %, not +50 %.  The
audit's projection was wrong because it assumed sync/launch overhead
was savable; it isn't unless we have GPU-wide sync hardware.

Three paths forward, ranked by margin:effort:
- **A. Profile the KFD signal-poll floor** (~3 days; could halve
  sync cost, ~+10 %)
- **B. Fuse resid_add + rmsnorm + silu_mul + qknorm/rope as
  intra-WG slices** (~1 week; +5–8 %)
- **C. Investigate GPU-wide ring barriers** (~1 week of research,
  may not exist; opens +50 % door if found)

Which path?
