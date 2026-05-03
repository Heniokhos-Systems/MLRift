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

### Qwen3-0.6B on RX 7800 XT — `--target=amdgpu-native` vs PyTorch ROCm

Greedy decode, 20 new tokens, seed token 14990, `attn_implementation="eager"`.
Median of 3 runs.  Peak GPU memory from `rocm-smi` snapshot at decode end.

| Stack | weights / compute | tok/s | peak GPU MB | vs PyTorch (same dtype) |
|---|---|---:|---:|---:|
| **MLRift `--target=amdgpu-native`** (`MLRIFT_GPU_MATMUL=1`) | bf16 / f32 | **35.0** | 1 920 | **0.84× vs ROCm fp32** |
| PyTorch ROCm eager | fp32 / fp32 | 41.6 | 2 280 | 1.00× |
| PyTorch ROCm eager | bf16 / bf16 | 73.7 | 1 140 | 1.00× |

The current `--target=amdgpu-native` path routes only the matmul +
lm_head through native gfx1100 ISA; the rest of the layer (qknorm,
rope, attn region) still runs on the CPU.  Token-id output is
bit-identical to HuggingFace `transformers.generate(do_sample=False)`
across all 20 generated tokens.

A previously-measured full-pipeline path (`use_gpu_full=1` + spec_K=4)
ran the same model at **92.9 tok/s** — **1.26× over PyTorch ROCm bf16**
and **2.23× over ROCm fp32** — but is currently dormant pending a
queue-state fix; see `docs/destroy_pytorch.md` and the project memory
for the open follow-up.

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
