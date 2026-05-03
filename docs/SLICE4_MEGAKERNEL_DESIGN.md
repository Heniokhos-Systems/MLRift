# Slice 4 — Mega-kernel design (route to bf16 single-stream win)

Status: design + measurement.  Implementation deferred to a dedicated
multi-session piece.

## Why slice 4 is the only path to honest single-stream bf16 win

After slices 1-3, qwen3-0.6B / RX 7800 XT decode is **launch-overhead
bound**, not ALU-bound and not bandwidth-bound:

```
$ MLRIFT_GPU_MATMUL=1 MLRIFT_GPU_FULL_FORWARD=1 MLRIFT_GPU_FLUSH_EVERY_N=28 /tmp/q3
  step N took 10 ms launches=421 syncs=2 sync_us=2300
  total_decode_ms=329  generated_tokens=20  tok/s=60.7
```

- **421 dispatches per token** (28 layers × 15 ops + 1 lm_head).
- **Only 2 syncs per token** (one trailing forward flush, one lm_head).
- 10 ms GPU compute / 421 = **23.8 us per dispatch** effective.
- 16.5 ms wall / token; ROCm bf16 ceiling = 13.6 ms / token.

We need to save ~3 ms / token to cross 73.7 tok/s.  Bandwidth-bound
matmuls (gemv_coop_bf16_f32) already run at the HBM ceiling for their
shape.  The 3 ms is dispatch overhead spent on the 11 NON-matmul ops
per layer (rmsnorm, qkv_split, qknorm, rope, attn, resid, silu).

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

Persistent-thread design: launch grid = max(qkv_dim, hidden, 2*ff)
WGs of 32 lanes.  Each WG owns a fixed slice of the OUTPUT dimension
across all phases.  Cross-WG sync via a per-layer global-memory
barrier (signal + spin-wait, similar to the KFD shim's signal-poll).

Phases inside the kernel (within one dispatch):

1. **rmsnorm input**: WG 0..ceil(hidden/32) computes b_x_norm = rmsnorm(in)
   into LDS.  Other WGs sleep via s_sleep.
2. **GLOBAL BARRIER 1** (cross-WG signal).
3. **qkv matmul**: every WG computes one row of b_qkv from b_x_norm
   (loaded back from LDS-shared region).  WG count = qkv_dim.
4. **GLOBAL BARRIER 2**.
5. **qkv_split + qknorm + rope** fused: per-head WG (16+8 WGs)
   rmsnorms head, applies rope, writes Q to b_q_attn or K/V to cache.
6. **BARRIER 3**.
7. **attn_decode**: per-head WG (16 WGs) computes attention output.
8. **BARRIER 4**.
9. **o_proj matmul**: per-row WG.  Add to in for residual b_mid.
10. **BARRIER 5**.
11. **post_norm**: 1 WG computes b_mid_norm in LDS.
12. **BARRIER 6**.
13. **gate_up matmul**: per-row WG of 2*ff.
14. **BARRIER 7**.
15. **silu_mul**: per-element WG over ff.
16. **BARRIER 8**.
17. **down matmul**: per-row WG of hidden.  Add to b_mid for final
    residual.  Write to `out`.
18. End kernel.

Per-token: 28 layer-mega-dispatches + 1 lm_head matmul = **29 launches**
(vs 421 today).  Saves ~9 ms of launch overhead per token at the 24
us-per-launch rate measured.  Step 16 ms - 9 ms = 7 ms = **143 tok/s**
projected.

## Hard parts

1. **Cross-WG global barriers.**  GFX11 has no ring-barrier instruction;
   we use signal-and-spin via a global-memory counter.  Each barrier:
   1 atomic_add + spin-load on a counter dword.  Tested empirically at
   ~5-10 us per barrier, 7 barriers/layer × 28 = 1.4-2.8 ms / token
   overhead — but still net win vs 9 ms saved.
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
