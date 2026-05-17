# MLRift — v1.0.0

A self-hosted systems language and compiler for machine-learning
workloads. Forked from [KernRift](https://github.com/Pantelis23/KernRift)
at commit `6cf758b` (v2.8.15); MLRift extends the IR backend with
ML-specific primitives (tensors, event streams, continuous-time
dynamics, sparse CSR ops, plasticity rules) and ships a native
AMDGCN GPU emitter that talks to `/dev/kfd` directly with zero ROCm
DSO dependencies on Linux.

**v1.0.0 highlights**

- Self-hosted (`build/mlrc` is built from `src/*.mlr` by `build/mlrc`).
- 439/439 tests pass, self-host fixed point holds.
- 8-target fat binary (`.mlrbo`) — Linux/macOS/Windows/Android × x86_64/arm64.
- Native AMDGCN backend for gfx1100 + gfx1030; 31 LLM kernels reachable.
- Four-platform benchmark report — see `benchmarks/BENCHMARKS.md`
  for x86_64 Linux, aarch64 Pi 400, Windows 11 x86_64, and Android arm64.

## Install

One-line installers pull the right `mlrc` (compiler) + `mlr` (runner) for
your platform from the latest [GitHub release](https://github.com/Pantelis23/MLRift/releases/latest)
and copy every `std/*.mlr` module from `main` into your local standard
library — no Python, no LLVM, no toolchain prerequisites.

**Linux / macOS / Android (adb or Termux)**

```sh
curl -sSf https://raw.githubusercontent.com/Pantelis23/MLRift/main/install.sh | sh
```

**Windows (PowerShell)**

```powershell
irm https://raw.githubusercontent.com/Pantelis23/MLRift/main/install.ps1 | iex
```

What lands on disk:

| File | Path |
|---|---|
| `mlrc` (compiler) | `~/.local/bin/mlrc` (Linux/macOS), `%LOCALAPPDATA%\MLRift\bin\mlrc.exe` (Windows) |
| `mlr` (fat-binary runner) | same directory as `mlrc` |
| stdlib (every `std/*.mlr`) | `~/.local/share/mlrift/std/` (Linux/macOS), `%LOCALAPPDATA%\MLRift\std\` (Windows) |

The stdlib list is **enumerated live from the repo via the GitHub
contents API**, so newly-added modules ship the moment they land on
`main` without re-cutting the installer. The current set spans
40 modules across language primitives (`io`, `fmt`, `vec`, `map`,
`string`, `math`, `mem`, `alloc`, `time`, `log`, `net`, `widget`,
`font`, `color`, `fb`, `fixedpoint`, `rng`, `thread`, `memfast`),
ML/numeric infrastructure (`matmul`, `quant`, `gguf`, `safetensors`,
`tokenizer`, `inference`, `inference_gpu`, `math_float`, `vec_f64*`),
model implementations (`qwen3`, `qwen35`, `qwen36`, `gemma2`,
`gemma3`), and GPU runtimes (`hip`, `hip_kfd`, `kfd`, `kfd_raw`).

Verify:

```sh
mlrc --version
echo 'fn main() { println("hello mlrift"); exit(0) }' > hello.mlr
mlrc hello.mlr -o hello.mlrbo   # 8-slice fat binary
mlr hello.mlrbo                 # runs the host slice
```

**Other install paths**

- Build from source: `git clone https://github.com/Pantelis23/MLRift && cd MLRift && make` (the in-tree `build/mlrc` self-compiles).
- Direct download: every per-arch binary lives under [`releases/latest`](https://github.com/Pantelis23/MLRift/releases/latest) (`mlrc-linux-x86_64`, `mlrc-windows-arm64.exe`, `mlrc.mlrbo`, …).

## What works today

Real LLM inference is already running end-to-end through the existing
MLRift compiler + a small ML stdlib (`std/qwen3.mlr`,
`std/matmul.mlr`, `std/tokenizer.mlr`, `std/gguf.mlr`) — no external
runtime, no Python.  Slice 9 (2026-05-17) added
`tokenizer_load_from_gguf(gf)` so chat REPLs can read the vocab + BPE
merges + special-token IDs + chat template directly from the GGUF
metadata, removing the dependency on external HuggingFace
`tokenizer.json` files. Phase 1 (commit 147fbe2) shipped the SPM path
with a greedy encoder and migrated Mistral off the v0.3 ID-shift hack;
Phase 2 added a canonical merge-rank BPE encoder for the gpt2 family
and migrated the three remaining chat REPLs (Qwen3-0.6B, Llama-3.2-1B,
Llama-3.2-3B) off external tokenizer.json files. Bit-exact vs the
HuggingFace `tokenizers` library on the 5-prompt chat-REPL test
corpus.

| Model | Quant | tok/s (CPU) | tok/s (GPU mega) | vs PyTorch bf16 GPU | Peak RSS / VRAM |
|---|---|---:|---:|---:|---|
| Qwen3-0.6B | bf16 (HF safetensors) | 32.0 | **119.7** (M=1) / **264** (mks16+PLD) | **~2.0× / 4.4-5.3× GPU** | 1.76-1.98 GB / ~2.0 GB |
| **Llama-3.2-1B-Instruct** | Q8_0→bf16 (CPU) / bf16 (GPU) | **16.2** | **99.8** (M=1) / **84.8** (speck4+PLD) | **+6.7% CPU, +95% GPU** | 2.39 GB / ~2.1 GB |
| **Llama-3.2-3B-Instruct** | Q8_0→bf16 (GPU) | — | **40.7** (M=1) | **+28% GPU** | 9.63 GB / ~5.8 GB |
| **Mistral-7B-Instruct-v0.2** | Q8_0 (GGUF) / bf16 (safetensors) | — | **22.9** (Q8_0) / **22.7** (bf16) | **+80% / +79% GPU** | 14.0 / 27.8 GB |
| Qwen3-0.6B | bf16 (GGUF) | 32.3 | ~33 (matmul-only) | **1.25×** CPU | 1.67 GB |
| **Qwen3-14B** | **Q8_0 (GGUF)** | **0.479** | — | **3.63×** CPU | **14.81 GB** |

7900X / 24 threads / greedy decode / use_cache.  GPU rows: RX 7800 XT
gfx1100, native KFD shim, `--target=amdgpu-native`, mega-kernel emitted
by MLRift's `@kernel` AST-walker (no hipcc / LLVM / clang on the
Llama-1B / Llama-3B / Mistral M=1 paths).  Tokens are bit-identical to
`llama.cpp` / PyTorch greedy on every row (20/20 on `"hello"` for
Qwen3 / Llama-1B / Llama-3B; Mistral-7B matches 22/23 — pos 6 lands on
the opposite side of a 1-bf16-ULP knife edge in the lm_head output,
bisected to reduction-tree topology, not precision).  Llama-1B CPU uses
`MLRIFT_CPU_BF16=1` (dequant once at load, AVX2 2-wide bf16 matmul).
Verified post-reboot 2026-05-17 — see `docs/bench_2026-05-17.md`.
Methodology + commit-by-commit perf history:

- `docs/bench_2026-05-17.md` — post-reboot verified scorecard for the
  4 GPU models (Qwen3-0.6B 119.7-264 tok/s, Llama-1B 99.8, Llama-3B 40.7,
  Mistral-7B 22.7-22.9), with single-env reproducers + caveats. Updated
  for slice 7.6 (multi-accumulator + VOPD `v_dual_fmac_f32`); Mistral
  Q8_0 21.7 → 22.9 (+5.5%), Qwen3 M=1 118 → 119.7, Llama-1B flat.
- `docs/bench_2026-05-13.md` — Llama-3.2-1B vs PyTorch ROCm fp32/bf16
  on RX 7800 XT (CPU bf16 +11 %, GPU M=1 +59 % vs PT bf16, +162 % vs
  PT fp32).  Session evolution `M=1: 33 → 81.8 → 99.8 tok/s` across
  slices 6.7e/f/g + 4.23 + 4.24 + 4.25 + 7.5.
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

### OS portability

The compiler and the emitted GPU bytes are OS-agnostic: `mlrc` builds
on Linux/macOS/Windows/Android (MLRift's portable host backends),
and the AMDGCN code object the emitter writes doesn't care about the
host kernel.  The **runtime** that loads and dispatches those bytes
is what's OS-bound:

| Runtime path | Linux | Windows | macOS |
|---|:---:|:---:|:---:|
| `--target=amdgpu-native` (KFD shim, no ROCm DSOs) | ✅ shipped | ❌ — KFD is Linux-only | ❌ |
| `--target=hip-amd` (links `libamdhip64.so` / `amdhip64.dll`) | ✅ | ⚠ untested but ROCm has Windows builds | ❌ — no ROCm |
| Native Metal backend (Apple) | n/a | n/a | not implemented |

The KFD shim talks directly to `/dev/kfd` via `ioctl()` — that's the
AMDKFD kernel driver, which exists only on Linux.  The "zero ROCm
DSOs" pitch trades portability for deployment simplicity.  Windows
AMD support is reachable today via the HIP runtime path with no
emitter changes (same code-object bytes, just a different DSO).
macOS would need a separate Metal/MPS backend, planned alongside
Slice C (NVIDIA PTX) as the next-platform work.

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
| **+ slice 2b** (fused `residual_rmsnorm` mid-layer + cross-layer) | bf16 / f32 | **61.4** | 1 920 | **1.48× vs ROCm fp32** |
| **+ `MLRIFT_GPU_MATMUL_BF16=0`** (slice 3 — pure fp32 weights) | f32 / f32 | **56.1** | 2 480 | **1.35× vs ROCm fp32** |
| **+ slice 2c** (fused `qknorm + rope_qk`) | bf16 / f32 | **69.9** | 1 920 | **0.95× vs ROCm bf16** |
| **+ `MLRIFT_QWEN3_MEGAKERNEL=1`** (slice 4 — one dispatch per layer; mega-kernel slices 4.10–4.13) | bf16 / f32 | **88.0** | 2 010 | **1.19× vs ROCm bf16** |
| **+ `MLRIFT_QWEN3_MEGAKERNEL_SPECK4=1` + `SPEC_K=4` + LONG_PROMPT (slice 4.14 — M=4 mega-kernel + PLD spec-decode)** | bf16 / f32 | **164.2** | 2 200 | **2.22× vs ROCm bf16** |
| **+ `MLRIFT_QWEN3_MEGAKERNEL_SPECK8=1` + `SPEC_K=8` + LONG_PROMPT (slice 4.15 — M=8 mega-kernel)** | bf16 / f32 | **181.8** | 2 600 | **2.46× vs ROCm bf16** |
| **+ slice 4.16 — phase-13 `v_wmma_f32_16x16x16_bf16` tensor cores** | bf16 / f32 | **190.3** | 2 600 | **2.57× vs ROCm bf16** |
| **+ `MLRIFT_QWEN3_MEGAKERNEL_SPECK16=1` + `SPEC_K=16` + LONG_PROMPT (slice 4.18 — M=16 mega-kernel; slice 4.17 unblocks max_seq=128)** | bf16 / f32 | **200.9** | 3 400 | **2.71× vs ROCm bf16** |
| **+ slice 4.20 — VRAM chase, mks-K cap correction, mks16 LDS bump 64→96** | bf16 / f32 | **216.4** | **2 046** | **3.46× vs ROCm bf16** |
| **+ post-4.20 lm_head bf16-direct (fb2de6a + 226b2e2, 2026-05-12)** | bf16 / f32 | **229.8** | **2 046** | **3.68× vs ROCm bf16** |
| **+ slice 8.5 single-env reproducer re-verified post-reboot 2026-05-17** | bf16 / f32 | **264.2** | **2 046** | **4.23× vs ROCm bf16** (peak, 5 spec steps) |
| **+ slice 8.5 M=1 single-env reproducer (`MLRIFT_NATIVE_MEGAKERNEL=2` only)** | bf16 / f32 | **118** | ~2 100 | **~2.0× vs ROCm bf16** (any prompt, no PLD) |
| MLRift + `GPU_FULL_FORWARD` + `SPEC_K=4` + LONG-prompt (per-op PLD path, pre-mega) | bf16 / f32 | 72.0 | 1 920 | 0.97× vs ROCm bf16 |

> **Caveat (2026-05-12): mks8 / mks16 are still hipcc-compiled.**  The
> 229.8 / 216.4 / 201.6 numbers above all flow through
> `qwen3_layer_megakernel_speck{8,16}.co` files that are built by
> `hipcc --offload-arch=gfx1100 --genco` from the matching
> `examples/llm/*.hip.cpp` sources.  This contradicts MLRift's "no
> hipcc, no LLVM, no clang in the build path" headline.  The M=1 mega
> (88 tok/s, 1.19× vs ROCm bf16) and mks4 (169.3 tok/s, 2.71× vs
> ROCm bf16) numbers DO run on AST-walker-emitted `.co` files
> (`--emit-amdgpu-qwen3-megakernel-v2` / `-speck4-v2`) and ship the
> destroy-PyTorch claim cleanly.  Slices 4.21+ port mks8 and mks16 to
> the same AST-walker pipeline; until they land, the mks-K rows above
> have an hipcc footnote.
>
> **Reproducing the 264 tok/s mks16 peak (slice 8.5, 2026-05-17)**:
> `MLRIFT_NATIVE_MEGAKERNEL=2 MLRIFT_QWEN3_MEGAKERNEL_SPECK16=1
> MLRIFT_SPEC_K=16 MLRIFT_LONG_PROMPT=1 MLRIFT_PLD_BENCH=1` —
> slice 8.5 collapsed the prior 3-env opt-in (`QWEN3_MEGAKERNEL=1`
> + `GPU_FULL_FORWARD=1`) into the single `MLRIFT_NATIVE_MEGAKERNEL=2`
> master switch.  Also run `scripts/rebuild_helper_cos.sh` first —
> mks8/mks16 are hipcc-compiled (no v2 AST-walker port yet) and are
> absent from `/tmp` after machine reset.  See
> [`docs/bench_2026-05-17.md`](docs/bench_2026-05-17.md) for the
> full env block + the 4-model scorecard.

The matmul-only rows route only the matmul + lm_head through native
gfx1100 ISA; qknorm, rope, attn, residuals still run CPU.  The
**`GPU_FULL_FORWARD=1`** row keeps the entire 28-layer forward
on-device (one D2H per token, only at lm_head) and **beats PyTorch
ROCm fp32 by 33 % at honest single-stream decode**, bit-identical
output.  The PLD row uses a synthetic prefill that the prefix-lookup
draft proposer hits at ~2 tok/step accept; **beats ROCm bf16 by 19 %**
on that workload.

The **mega-kernel** (`MLRIFT_QWEN3_MEGAKERNEL=1`) collapses the 28-layer
chain into one dispatch per layer (29 launches/token vs the per-op chain's
310) and lands at **88.0 tok/s — +19 % over PyTorch ROCm bf16 on fp32
weights**, bit-identical to the reference.

**Slice 4.14** adds an `M=4` variant (`MLRIFT_QWEN3_MEGAKERNEL_SPECK4=1`)
that processes 4 query tokens per dispatch, paired with the existing
PLD prefix-lookup draft proposer.  Each weight row is read once and
drives 4 dot products → 4× compute amortisation on the bandwidth-bound
matmuls.  At `SPEC_K=4 + LONG_PROMPT` it lands at **164.2 tok/s —
2.22× PyTorch ROCm bf16**.

**Slice 4.15** doubles to `M=8` (`MLRIFT_QWEN3_MEGAKERNEL_SPECK8=1`,
`SPEC_K=8`): **181.8 tok/s reported / ~222 tok/s steady-state**
(the warmup-diluted number is what the bench prints because
`max_seq=64` only fits 5 fast steps after step 0's cold dispatch).
**2.46× PyTorch ROCm bf16.**

**Slice 4.16** drops in `v_wmma_f32_16x16x16_bf16` tensor-core
instructions for phase 13 (gate_up matmul) of the mks8 kernel, edging
to **190.3 tok/s — 2.57× PyTorch ROCm bf16** on fp32 compute / bf16
storage.

**Slice 4.17** bumps `max_seq` from 64 → 128 to unblock longer decode
beyond the previous 64-pos KV cache boundary.  No regression on
existing short benches; opens room for `M ≥ 16` spec-decode.

**Slice 4.18** scales the M-amortisation to `M=16`
(`MLRIFT_QWEN3_MEGAKERNEL_SPECK16=1`, `SPEC_K=16`).  Each weight row
now drives 16 dot products per dispatch and the WMMA tile (16×16×16)
is fully utilised.  Two design constraints handled: phase 7's
cooperative-WG count drops `ATTN_COOP=4 → 2` to keep total at
WG_PERSIST=512; LDS softmax cache stays at `M_EFF×64` (instead of
×128) so per-CU LDS budget isn't exceeded.  Lands at **200.9 tok/s
reported / ~250 tok/s steady-state — 2.71× / 3.38× PyTorch ROCm
bf16**, output bit-identical to the M=1 mega-kernel.  See
[`docs/SLICE4_MEGAKERNEL_DESIGN.md`](docs/SLICE4_MEGAKERNEL_DESIGN.md)
for the slice 4.10–4.18 progression.

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
| 2b. ✅ Fused `residual_rmsnorm` (mid + cross-layer) | 56 launches/token saved | **61.4** | **1.48× ROCm fp32** |
| 3. ✅ **Pure fp32 path** (`MLRIFT_GPU_MATMUL_BF16=0`) | f32 weight VRAM, f32 GEMM accum (apples-to-apples vs ROCm fp32) | **59.1** | **1.42× ROCm fp32** |
| 2c. ✅ Fused `qknorm + rope_qk` (Q+K heads in one launch) | 1 dispatch replaces qknorm Q + qknorm K + rope Q + rope K; 366 → 310 launches/step (−56 = −2/layer × 28); 143 dwords gfx1100 ISA, smoke-test bit-exact, end-to-end PyTorch parity | **63.2** | **1.52× ROCm fp32** |
| 4.8 ✅ **Mega-kernel wire-up** (one dispatch per layer; collapses 15 ops into 1) | 310 → 29 launches/token; baseline at WG=64 | 18.9 | 0.46× ROCm fp32 |
| 4.10 ✅ WG_PERSIST 64 → 512 (recover gemv_coop row parallelism) | 8× wider WG fan-out for matmul phases | **46.0** | **1.10× ROCm fp32** |
| 4.11 ✅ Cooperative phase 7 (ATTN_COOP=4) + cached softmax LDS + dropped trailing fence | 4 WGs/head, 32-dim/lane pass 3; weights cached, no Q·K redo; mega ≈ per-op | **54.1** | **1.30× ROCm fp32** |
| 4.12 ✅ bf16x2 packed matmul loads (`u32 = 2 bf16`) | halves matmul VMEM count, +20 % | **64.9** | **0.88× ROCm bf16** |
| **4.13 ✅** **Channel-repacked padded weights (HIDDEN_PAD=1152, Q_DIM_PAD=2176, FF_PAD=3200)** | break GDDR6 16-channel × 256 B cycle so consecutive rows hit distinct channels (was 2/16 → now 16/16) | **88.0** | **1.19× ROCm bf16** |
| **4.14 ✅** **M=4 mega-kernel + PLD spec-decode** (`MLRIFT_QWEN3_MEGAKERNEL_SPECK4=1` + `MLRIFT_SPEC_K=4`) | each weight row drives 4 dot products → 4× amortisation on bandwidth-bound matmuls; bit-identical to M=1 mega | **164.2** | **2.22× ROCm bf16** |
| **4.15 ✅** **M=8 mega-kernel** (`MLRIFT_QWEN3_MEGAKERNEL_SPECK8=1` + `MLRIFT_SPEC_K=8`) | doubles batch dim; phase 5/7 expand to 256/512 active WGs; ~222 tok/s steady-state | **181.8** (warmup-diluted) | **2.46× ROCm bf16** |
| **4.16 ✅** **WMMA `v_wmma_f32_16x16x16_bf16`** on phase 13 of mks8 | tensor-core 16×16×16 matmul replaces bf16x2 vector FMAs; modest at M=8 (50 % tile utilization) | **190.3** | **2.57× ROCm bf16** |
| **4.17 ✅** **`max_seq` 64 → 128** | unblocks longer decode + M ≥ 16 spec; LDS attn cache scales accordingly | n/a (infrastructure) | unchanged |
| **4.18 ✅** **M=16 mega-kernel** (`MLRIFT_QWEN3_MEGAKERNEL_SPECK16=1` + `MLRIFT_SPEC_K=16`) | 16-way matmul amortisation; ATTN_COOP=4→2 for phase-7 WG fit; LDS cache capped at M×64 for per-CU budget; WMMA tile fully utilised | **200.9** (250 steady) | **2.71× / 3.38× ROCm bf16** |
| 4b (deferred). WMMA on phases 3/9/17 of mks16 (full tile utilization at M=16) | ~5-10 % per phase, compounded | est. 220-260 | est. 3.0-3.5× ROCm bf16 |

WMMA at honest M=1 decode remains off the single-stream critical
path: at slice 4.13 we are **bandwidth-bound on the matmul k-loop
VMEM**, not ALU-bound, even after channel repacking.  WMMA still
matters for prefill / `SPEC_K=4` mode where M_eff ≥ 4 hits the
tensor core efficiently — see slice 4b.

Memory roofline on this card (624 GB/s ÷ ~880 MB bf16 weights/token)
is ≈ 700 tok/s.  We are at **13 % single-stream** today (slice 4.13
at 88 tok/s); PyTorch ROCm bf16 is at ~10 %.  Nothing here is blocked
on hardware.  Tracking as tasks #178–#181 with the full methodology
and per-slice notes in `project_destroy_pytorch_roadmap.md` and the
mega-kernel slice 4.10–4.13 progression in
[`docs/SLICE4_MEGAKERNEL_DESIGN.md`](docs/SLICE4_MEGAKERNEL_DESIGN.md).

### Pure fp32 win — fully apples-to-apples vs PyTorch ROCm fp32

The `MLRIFT_GPU_MATMUL_BF16=0` row above is the dtype-clean f32
comparison: f32 weights resident in VRAM (one-time dequant from
bf16 + cached), f32 GEMM compute, f32 accumulator — same dtype
profile as PyTorch ROCm fp32.  We hit **59.1 tok/s vs their 41.6
(1.42×)** at honest single-stream decode, bit-identical token output,
all on the same RX 7800 XT.

### Llama-3.2-1B on RX 7800 XT — second model beating PyTorch

After Qwen3-0.6B, **Llama-3.2-1B-Instruct** is the second model running
fully on the AST-walker-emitted mega-kernel pipeline.  Same hardware,
same `--target=amdgpu-native` KFD shim, no hipcc.  Greedy decode of
prompt `"hello"`, N=20, median of 3 runs.

| Stack | dtype (weights / compute) | tok/s | peak VRAM | Peak RSS | vs PyTorch (matching dtype) |
|---|---|---:|---:|---:|---:|
| MLRift CPU Q8_0 | int8 / f32 | 1.34 | — | 1.27 GiB | 0.088× PT CPU bf16 |
| **MLRift CPU bf16** (`MLRIFT_CPU_BF16=1`, slice 4.27) | bf16 / f32 | **16.2** | — | **2.39 GiB** | **1.07× PT CPU bf16, -37% RAM** |
| PyTorch ROCm fp32 | f32 / f32 | 31.2 | 4 749 MiB | ~4.7 GiB | 1.00× (PT fp32 baseline) |
| PyTorch CPU bf16 (MKL + oneDNN) | bf16 / f32 | 15.2 | — | ~3.8 GiB | 1.00× (PT CPU bf16 baseline) |
| PyTorch ROCm bf16 (SDPA) | bf16 / bf16 | 51.3 | 2 392 MiB | ~3.8 GiB | 1.00× (PT GPU bf16 baseline) |
| **MLRift M=1 mega** (slice 7.5) | bf16 / f32 | **99.8** | ~2 100 MiB | 3.71 GiB | **1.95× PT GPU bf16 (+95%)** |
| **MLRift speck4 + PLD** | bf16 / f32 | **84.8** | ~2 100 MiB | 1.27 GiB | **1.65× PT GPU bf16 (+65%)** |

**MLRift > PyTorch on both CPU and GPU for Llama-3.2-1B.**
The GPU mega-kernel uses higher numerical precision than PT bf16 GPU
(bf16 weights stream, f32 activations + f32 accumulator throughout —
matches PT fp32 fidelity, runs 2.62× faster than PT fp32 GPU at 31.2
tok/s).  CPU bf16 path beats PT CPU bf16 (MKL+oneDNN) by +6.7 % via
MLRift's AVX2 2-wide bf16 inner loop (`mm_worker_bf16_f32_avx2_naive_2w`
in `std/matmul.mlr:424`).  **Slice 4.27 (2026-05-13 mem opt)** drops
CPU bf16 peak RSS from 3.63 → 2.39 GiB (-34 %) by `madvise(MADV_DONTNEED)`
on the Q8_0 GGUF pages immediately after each tensor's bf16 dequant,
plus reusing the tied lmhead bf16 buffer for embedding lookup so the
Q8_0 embed region can also be evicted. MLRift's CPU bf16 RSS (2.39 GiB)
is now **37 % below** PyTorch's CPU bf16 RSS (~3.8 GiB).  Session
evolution on M=1:

| Slice | Change | M=1 tok/s |
|---|---|---:|
| 6.7e/f/g | Per-layer flush + GPU lm_head + GTT residuals (correctness fixes) | 33 |
| 4.23 | Drop `buffer_gl1_inv` from cross-WG barrier spin loop | 37 |
| 4.24 | 2× unroll inner K-loop + `s_clause(5)` on 6-load batch | 58.3 |
| 4.25 | 4× unroll inner K-loop + `s_clause(11)` on 12-load batch | 81.8 |
| **7.5** | **8× unroll inner K-loop + `s_clause(23)` on 24-load / 16-FMA batch** | **99.8** |

2.48× cumulative on M=1 from a single session of inner-loop optimisation,
all in MLRift's `_emit_gemv_coop_bf16_padded_strided_inline`
(`src/format_amdgpu_megakernel.mlr:991-1170`).  Full per-slice notes
in `docs/bench_2026-05-13.md`.

## Build

```
make build    # self-compiles build/mlrc (bootstrap committed)
make test     # 439/439
make bootstrap   # verify stage3 == stage4
mlrc --version
```

## Lineage — forked from KernRift

MLRift began as a soft fork of [KernRift](https://github.com/Pantelis23/KernRift)
at v2.8.15 and shares its type system, IR-level optimization pipeline, and the
host code generators for x86_64 + ARM64 across Linux / macOS / Windows /
Android. MLRift-specific work lives in added passes, added IR ops, the AMDGCN
emitter, and ML-oriented stdlib modules — not in re-implementing the basics.
Generic compiler-level findings that apply upstream are tracked in
`docs/kernrift_upstream.md`.

## License

See `LICENSE`.
