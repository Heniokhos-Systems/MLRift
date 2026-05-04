# MLRift

A systems language for machine-learning workloads — built on
[KernRift](https://github.com/Pantelis23/KernRift) at commit `6cf758b`
(v2.8.15). The compiler's own source is KernRift; MLRift extends the
KRIR backend with ML-specific primitives (tensors, event streams,
continuous-time dynamics, sparse CSR ops, plasticity rules) and will
introduce a `.mlr` frontend for user programs as the roadmap lands.

**Status:** day zero. The rename pass is done (product identity is
MLRift, binary is `mlrc`, bootstrap binary is `build/mlrc`), 436/436
tests pass, self-host fixed point holds. The MLRift-specific syntax
and IR extensions are not started yet — those follow the roadmap in
`~/Desktop/Projects/Work/ideas/MLRift.md`.

## What works today

Real LLM inference is already running end-to-end through the existing
KernRift compiler + a small ML stdlib (`std/qwen3.mlr`,
`std/matmul.mlr`, `std/tokenizer.mlr`, `std/gguf.mlr`) — no external
runtime, no Python.

| Model | Quant | tok/s (CPU) | tok/s (GPU matmul) | vs PyTorch BF16 | Peak RSS |
|---|---|---:|---:|---:|---:|
| Qwen3-0.6B | bf16 (HF safetensors) | 32.03 | ~33 | **1.24×** | 1.67 GB |
| Qwen3-0.6B | bf16 (GGUF) | 32.27 | ~33 | **1.25×** | 1.67 GB |
| **Qwen3-14B** | **Q8_0 (GGUF)** | **0.479** | — | **3.63×** | **14.81 GB** |

7900X / 16 threads / greedy decode / use_cache.  GPU rows: RX 7800 XT
gfx1100, native KFD shim, `MLRIFT_GPU_MATMUL=1`.  First 10 generated
tokens are bit-identical to HuggingFace `transformers.generate`
across both sizes (token-id check, not a fuzzy text match). Methodology
+ commit-by-commit perf history:

- `docs/BENCH_QWEN3.md` — Qwen3-0.6B (78× from scalar baseline, 3.27×
  vs PyTorch F32, full per-op breakdown).
- `docs/BENCH_QWEN3_14B.md` — Qwen3-14B Q8_0 vs PyTorch BF16 (3.63×
  decode, 1.37× less peak RSS, 5-token prompt → 20-token greedy
  continuation).
- `docs/bench_60m.md` — 60 M neuron / 240 M synapse spiking sim,
  74× over PyTorch CPU and 3.6× over PyTorch GPU on the same card,
  end-to-end via the native AMDGCN emitter (zero ROCm DSOs in the
  launcher binary).

## AMD GPU backend

The native AMDGCN emitter (`src/format_amdgpu.mlr`) compiles `@kernel`
functions directly to gfx1100 (RDNA 3) ELF code objects.  31 LLM
kernels are reachable today via the AST-walking lowerer (Phase 3).
Pass `--target-arch=gfx1030` and the same source emits RDNA 2 binaries
that disassemble cleanly under `llvm-objdump --mcpu=gfx1030` with zero
`.long` placeholders — Slice B (RDNA 2) is feature-complete for the
LLM kernel set.  Slice C (NVIDIA Blackwell / Ada / Ampere via PTX) is
the next target.

### Qwen3-0.6B on RX 7800 XT — destroy-PyTorch comparison

Greedy decode, 20 new tokens, seed token 14990, `attn_implementation="eager"`.
Median of 3 runs.  Goal: **beat PyTorch ROCm in both fp32 and bf16.**
Token-id output of every MLRift row is bit-identical to HuggingFace
`transformers.generate(do_sample=False)` across all 20 tokens.

| Stack | dtype (weights / compute) | tok/s | peak GPU MB | vs PyTorch (same dtype) |
|---|---|---:|---:|---:|
| PyTorch ROCm eager | fp32 / fp32 | 41.6 | 2 280 | 1.00× (baseline) |
| PyTorch ROCm eager | bf16 / bf16 | 73.7 | 1 140 | 1.00× (baseline) |
| MLRift `--target=amdgpu-native` (matmul only) | bf16 / f32 | 35.1 | 1 920 | 0.48× vs ROCm bf16 |
| MLRift `--target=amdgpu-native` (matmul only, `MLRIFT_GPU_MATMUL_BF16=0`) | f32 / f32 | 35.2 | 1 920 | 0.85× vs ROCm fp32 |
| **MLRift `--target=amdgpu-native` + `MLRIFT_GPU_FULL_FORWARD=1`** | bf16 / f32 | **55.4** | 1 920 | **1.33× vs ROCm fp32** |
| **+ `MLRIFT_GPU_FLUSH_EVERY_N=28`** (slice 2 — drop per-layer sync) | bf16 / f32 | **60.4** | 1 920 | **1.45× vs ROCm fp32** |
| **+ slice 2b** (fused `residual_rmsnorm` mid-layer) | bf16 / f32 | **60.7** | 1 920 | **1.46× vs ROCm fp32** |
| **+ `MLRIFT_GPU_MATMUL_BF16=0`** (slice 3 — pure fp32 weights) | f32 / f32 | **59.1** | 2 480 | **1.42× vs ROCm fp32** |
| **MLRift + `GPU_FULL_FORWARD` + `SPEC_K=4` + LONG-prompt (PLD)** | bf16 / f32 | **87.5** | 1 920 | **1.19× vs ROCm bf16** |

The matmul-only rows route only the matmul + lm_head through native
gfx1100 ISA; qknorm, rope, attn, residuals still run CPU.  The
**`GPU_FULL_FORWARD=1`** row keeps the entire 28-layer forward
on-device (one D2H per token, only at lm_head) and **beats PyTorch
ROCm fp32 by 33 % at honest single-stream decode**, bit-identical
output.  The PLD row uses a synthetic prefill that the prefix-lookup
draft proposer hits at ~2 tok/step accept; **beats ROCm bf16 by 19 %**
on that workload.

**Dtype clarification.** All MLRift rows above use `bf16 weights /
f32 compute` — weights stream from VRAM as bf16 and widen to f32
inside `gemv_coop_bf16_f32` (no dequant pass), with all
matmul/rmsnorm/attn accumulators in f32.  The `BF16=0` row swaps
in an f32-weight (dequanted-once-and-cached) variant; arithmetic is
still f32.  PyTorch ROCm bf16 in the table, by contrast, runs
**bf16 storage + bf16 GEMM accum** — strictly less numerical
headroom than ours; we beat their fp32 row using less than half its
weight memory.

`GPU_FULL_FORWARD=1` is opt-in for now (not default) for two
reasons: (1) the on-device chain depends on a `/tmp` `.co` cache
that is rebuilt from source by `mlrc`; if the cache is stale relative
to the current `build/mlrc` AST recogniser, the chain produces
silent wrong tokens (the matmul-only path fails loudly via threshold
+ CPU fallback), and (2) it allocates ~1.9 GB of GTT/VRAM that the
matmul-only path doesn't need on smaller cards.  Default-on after
the `.co` cache moves into the binary and an md5-vs-fresh-emit
verify gate lands.

The flag is **qwen3-specific** — it gates a transformer-shaped
chain in `examples/qwen3_generate.mlr`.  The 60 M neuron SNN bench
in `docs/bench_60m.md` already runs fully on-device through a
different path (`noesis_60m_gpu_launch.mlr` keeps state +
spike_mask + CSR resident, dispatches `decay_step → delivery_step →
lif_step` with no per-step D2H); its 26.6 s sim is bandwidth-bound
on the LIF state read, not launch-overhead bound, so a
forward-style flag wouldn't apply.

Roadmap to extend the lead, ranked by ceiling:

| Slice | unlock | tok/s | vs PyTorch (same dtype) |
|---|---|---:|---:|
| 2. ✅ Per-token flush throttle (`MLRIFT_GPU_FLUSH_EVERY_N=28`) | drop 27/28 per-layer syncs (~75 µs each) | **60.4** | **1.45× ROCm fp32** |
| 3. ✅ **Pure fp32 path** (`MLRIFT_GPU_MATMUL_BF16=0`) | f32 weight VRAM, f32 GEMM accum (apples-to-apples vs ROCm fp32) | **59.1** | **1.42× ROCm fp32** |
| 2b. Kernel-level fusion (`resid+rmsnorm`, `qknorm+rope`, qknorm-Q+K) | save ~5 launches/layer × 28 × ~10 µs ≈ 1.4 ms/token | 65–70 | 0.88–0.95× ROCm bf16 |
| 4. **Mega-kernel** (one dispatch per layer; collapses ~15 ops) — design + measurement complete, see [`docs/SLICE4_MEGAKERNEL_DESIGN.md`](docs/SLICE4_MEGAKERNEL_DESIGN.md) | 421 → ~29 dispatches/token; saves 9 ms of launch overhead | **143 (projected)** | **1.94× ROCm bf16** |
| 4b. WMMA bf16 GEMV through `gpu_matmul` (M ≥ 4) | 2× on prefill / spec_K matmuls (gfx1100 tensor cores) | 100–120 PLD | 1.4–1.6× ROCm bf16 |

WMMA at honest M=1 decode is dropped from the critical path: profile
shows we're now **launch-overhead bound**, not ALU-bound (421 launches
× ~24 µs ≈ 10 ms/step; only 2.3 ms is sync wait).  WMMA accelerates
ALU on a workload that's bandwidth-bound — no help at M=1.  It still
matters for prefill / `SPEC_K=4` mode where M_eff ≥ 4 hits the tensor
core efficiently.

Single-stream **bf16 win** has to come from launch-count reduction
(slice 4 mega-kernel).  The mega-kernel collapses the 15-op layer
into 1 dispatch — saves ~14 launches/layer × 28 layers × ~10 µs =
3.9 ms/token, taking step time from 10 ms → 6 ms ≈ 165 tok/s ceiling.

Memory roofline on this card (624 GB/s ÷ 600 MB bf16 weights) is
≈1 040 tok/s; we are at **6 % single-stream / 8 % with PLD** today,
ROCm bf16 is at 7 %.  Nothing here is blocked on hardware.  Tracking
as tasks #178–#181 with the full methodology and per-slice notes in
`project_destroy_pytorch_roadmap.md`.

### Pure fp32 win — fully apples-to-apples vs PyTorch ROCm fp32

The `MLRIFT_GPU_MATMUL_BF16=0` row above is the dtype-clean f32
comparison: f32 weights resident in VRAM (one-time dequant from
bf16 + cached), f32 GEMM compute, f32 accumulator — same dtype
profile as PyTorch ROCm fp32.  We hit **59.1 tok/s vs their 41.6
(1.42×)** at honest single-stream decode, bit-identical token output,
all on the same RX 7800 XT.

## Build

```
make build    # self-compiles build/mlrc (bootstrap committed)
make test     # 436/436
make bootstrap   # verify stage3 == stage4
mlrc --version
```

## Why "built on KernRift"

MLRift is explicitly a **layer on top of KernRift**, not a hard fork.
It shares the type system, the optimization pipeline, the codegen
backends (x86_64 + ARM64, Linux/macOS/Windows/Android), and all the
infrastructure KernRift spent the last year hardening. MLRift-specific
work lives in added passes, added IR ops, and a new frontend — not in
re-implementing the basics. When KernRift fixes a backend bug,
MLRift inherits it with a cherry-pick.

## License

Same as KernRift — see `LICENSE`.
