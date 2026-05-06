# Qwen3-0.6B CPU Inference Bench

MLRift vs PyTorch, single-machine CPU, greedy decode. Kept here so we
don't have to re-derive it every time we revisit the perf work.

> **Looking for the GPU numbers?** Qwen3-0.6B on RX 7800 XT lands at
> **88 tok/s single-stream → 164 (M=4) → 190 (M=8 + WMMA) → 200 (M=16)
> — +271 % over PyTorch ROCm bf16, +390 % over fp32, on fp32 compute /
> bf16 weights, bit-identical** (steady-state ~250 tok/s at M=16).
> Setup, methodology, and the full slice 4.10 → 4.18 mega-kernel
> progression are in
> [`SLICE4_MEGAKERNEL_DESIGN.md`](SLICE4_MEGAKERNEL_DESIGN.md), with the
> destroy-PyTorch comparison table in the top-level
> [`README.md`](../README.md#qwen3-06b-on-rx-7800-xt--destroy-pytorch-comparison).

## Hardware / environment

| Item | Value |
|---|---|
| CPU | AMD Ryzen 9 7900X — 12 cores / 24 SMT threads, Zen 4 |
| SIMD path | AVX2 (AVX-512 present on Zen 4 but microcoded; we don't use it) |
| RAM | DDR5 |
| OS | Linux 6.17 |
| Worker threads | 16 (best wall-clock; 12 and 24 are ~5 % slower) |
| Model | Qwen3-0.6B, bfloat16 weights from the HF release |
| Input | seed token 14990 ("hello"), `max_new_tokens=20`, greedy |
| Correctness gate | all 20 generated token ids bit-identical to HuggingFace `transformers.generate(do_sample=False)` |

## Arithmetic used by each config

Both MLRift and PyTorch BF16 on this CPU do **bf16 weights streamed
into f32 lanes at load time** — neither uses native bf16 FMA intrinsics
(the 7900X doesn't expose them). The GEMM arithmetic in both is f32
multiply-add, accumulator f32. So "PyTorch BF16" on this CPU is
essentially a memory-bandwidth variant of PyTorch F32, not a different
arithmetic.

| Config | Weight storage | Matmul compute |
|---|---|---|
| MLRift safetensors | bf16 on disk, decoded per-element inside AVX2 matmul | f32 FMA |
| MLRift GGUF | bf16 on disk (via our converter), same decode | f32 FMA |
| PyTorch F32 | f32 in memory | f32 FMA |
| PyTorch BF16 | bf16 in memory, streamed to f32 per matmul tile | f32 FMA |

## Results (20-token greedy, same seed, same CPU)

Wall numbers are the decode-only segment (excluding model load).
Memory is the kernel `RUSAGE_SELF.ru_maxrss` peak over the whole
process (load + decode). Run-to-run variance is ±3 % on the wall
numbers; memory is deterministic.

| Config | Wall | tok/s | Peak RSS | vs PyTorch F32 | vs PyTorch BF16 |
|---|---:|---:|---:|---:|---:|
| **MLRift safetensors** | **624 ms** | **32.03** | **1.67 GB** | **3.27×** | **1.24×** |
| **MLRift GGUF** | **619 ms** | **32.27** | **1.67 GB** | **3.30×** | **1.25×** |
| PyTorch BF16 (bf16 weights, f32 GEMM) | 774 ms | 25.83 | 4.44 GB | 2.64× | 1.00× |
| PyTorch F32 (f32 weights, f32 GEMM) | 2 043 ms | 9.79 | 7.23 GB | 1.00× | 0.38× |

Reference frame that makes the most sense arithmetically is **MLRift
vs PyTorch F32**: same FMA dtype, same accumulator dtype, same CPU.
We're **3.27× faster** and use **4.3× less memory**. Against PyTorch
BF16 on the same CPU (which also runs f32 GEMM): **1.24× faster** and
**2.6× less memory**.

Memory story: MLRift mmap's bf16 weights once (1.4 GB resident for
the touched pages) plus ~250 MB of fused Q/K/V + gate/up stacked
copies and per-step scratch. PyTorch loads the whole checkpoint into
torch tensors and keeps both a bf16 master copy and a f32 GEMM staging
buffer, which is why bf16-on-CPU still uses ~3× what MLRift does for
the same weights.

## Per-op breakdown (MLRift safetensors, 20-token run at 684 ms)

Per-run totals (ms), captured by `qwen3_profile_dump()` in
`std/qwen3.mlr`. Adds to more than wall because some of the "matmul
submit" cost overlaps with MLRift's synchronous `thread_pool_wait`.

| Op | ms | % of layer_sweep |
|---|---:|---:|
| qkv_proj (FUSED matmul, 1 per layer since `ef27a56`) | 103 | 15 |
| gate_up (FUSED matmul, 1 per layer since `ef27a56`) | 150 | 22 |
| down_res2 (matmul + residual, 1 per layer) | 86 | 13 |
| oproj_res1 (matmul + residual, 1 per layer) | 67 | 10 |
| qknorm_rope (scalar, no more eps mmap) | 31 | 5 |
| attn (AVX2 dot + AVX2 axpy) | 24 | 4 |
| post_norm (scalar) | 14 | 2 |
| input_norm (scalar) | 13 | 2 |
| silu (AVX2) | 7 | 1 |

Plus `lm_head` = 118 ms and argmax = 7 ms outside `layer_sweep`.

**Matmul accounts for ~80 % of wall now.** Everything else was dragged
down to the low single digits once the AVX2 kernels landed and the
RMSNorm mmap leak was fixed.

## How we got here

Each row correctness-verified bit-identical to PyTorch before commit:

| Commit | Change | Wall | tok/s | Mul |
|---|---|---:|---:|---:|
| (baseline) | Scalar Kahan matmul, ST `down_proj` | 48 697 ms | 0.41 | 1.0× |
| `1ea14db` | MT `down_proj` + timers | 22 860 ms | 0.87 | 2.1× |
| `7898b11` | `std/matmul.mlr` extract, persistent ctx, Kahan-off flag | 19 142 ms | 1.04 | 2.5× |
| `7018028` | AVX2 BF16×F32 matmul kernel | 7 676 ms | 2.61 | 6.3× |
| `4c40e58` | Static `bf16_to_f32` scratch (no alloc per call) | 1 032 ms | 19.4 | 47× |
| `55af77e` | Rope cache + polynomial `exp_f32_fast` + profiling | 996 ms | 20.1 | 49× |
| `7e24197` | AVX2 SiLU (mul+add poly, dual clamp) | 876 ms | 22.8 | 55.6× |
| `d61f0d7` | AVX2 attention (dot + axpy with static scratch) | 770 ms | 25.96 | 63× |
| `68d517b` | Static RMSNorm eps (no `uint64[1]` mmap per call) | 729 ms | 27.41 | 66.8× |
| `7b25eb5` | 2-wide AVX2 matmul (independent FMA accumulators) | 703 ms | 28.44 | 69.3× |
| `ef27a56` | Fuse Q/K/V + gate/up matmuls at load time | 684 ms | 29.24 | 71.2× |
| `e114133` | AVX2 rmsnorm (bf16-weight + f32-weight) | 642 ms | 31.15 | 75.8× |
| `312bc74` | Main thread runs its own matmul slice inline | 632 ms | 31.65 | 77.0× |
| `211a8d3` | (opt) unfused path for large-model memory savings | **624 ms** | **32.03** | **78.0×** |

## Reproduce

```bash
# MLRift (from repo root):
./build/mlrc --emit=elfexe examples/qwen3_generate.mlr -o /tmp/qw3_generate
/usr/bin/time -v /tmp/qw3_generate

# MLRift GGUF variant (needs the BF16 GGUF from the converter):
python MLRift-experimental/qwen3_safetensors_to_gguf.py   # one-time, ~1.4 GiB
./build/mlrc --emit=elfexe examples/qwen3_generate_gguf.mlr -o /tmp/qw3_generate_gguf
/usr/bin/time -v /tmp/qw3_generate_gguf

# PyTorch side (from MLRift-experimental/):
source venv/bin/activate
python bench_qwen3_pytorch.py bf16 16
python bench_qwen3_pytorch.py f32  16
```

## Things that worked, in order of unexpected leverage

1. **Static BF16 decode scratch** — a per-call `alloc(4)` inside
   `bf16_to_f32` looked harmless until profiling showed 1.3 M calls per
   run and 2.6 M mmap/munmap syscalls. Replacing with a module-level
   4-byte buffer went 7 676 ms → 1 032 ms. Single biggest single-step
   win of the session.

2. **AVX2 matmul kernel** with inline `vpmovzxwd` + `vpslld 16` to
   decode 8 bf16 weights per lane per iter, then `vfmadd231ps`. 8-wide
   FMA + integrated decode. 48 697 → 7 676 ms when combined with the
   Kahan-off flag for non-residual matmuls.

3. **Fused `silu × up`** at 8-wide with an AVX2 polynomial `exp`. The
   scalar exp was the bottleneck of the non-matmul path.

4. **AVX2 attention** as two tiny helpers (`qwen3_dot_avx2`,
   `qwen3_axpy_avx2`) over a static scratch. The scratch matters a lot:
   first attempt used per-call `alloc`/`dealloc` and made attention
   *slower* than scalar before I moved it to a static.

## Gotchas worth remembering

- **FMA vs mul+add matters**. A `vfmadd213ps` in a poly is ~1 ULP more
  accurate per step than `vmulps + vaddps`; over 28 Qwen3 layers the
  accumulated ULP drift flips argmax. The AVX2 SiLU uses mul+add to
  stay bit-identical with the scalar reference (which doesn't have
  FMA). If we ever switch MLRift's scalar codegen to emit FMA, flip
  both sides together.

- **AVX2 poly needs both clamps.** The `2^n` bit-injection path
  (`(n + 127) << 23` as f32 bits) breaks if `n + 127` goes negative.
  A max clamp alone is not enough; `x ≥ −87` must be enforced too.

- **`unsafe { *(p as f32) = v }` was emitting an 8-byte store.** The
  upstream MLRift parser bug is fixed (commit `6fa19f2`), but be aware
  for historical branches.

- **Short-jump reach**. The original AVX2 kernel tried to put the
  entire K-loop inside one `asm {}` block; it overflowed rel8. Keeping
  the 8-step increment in MLRift and making the asm block straight-line
  code is simpler and not measurably slower.

## Next push ideas, ordered by expected win

1. **Thread-pool submit overhead.** ~78 µs × 1 800 matmul submits ≈
   140 ms of the 770 ms wall. Candidates: reusable "group submit" (all
   workers for one matmul, one futex wake), busy-wait-with-backoff on
   the main thread side when the submit latency dominates compute.

2. **Fuse Q/K/V into a single matmul.** Stacked weight matrix
   `[q_dim + 2*kv_dim, hidden]` means 1 submit instead of 3 per layer.
   Removes ~25 ms of overhead; also improves weight-streaming locality.

3. **Vectorise `qknorm_rope`.** 62 ms currently, mostly scalar f32.
   Same AVX2 shape as SiLU/attn helpers — should drop to ~10 ms.

4. **Fuse residual add into `o_proj` / `down_proj`.** Each currently
   does a separate scalar pass over 1 024 f32s that we just wrote.
   Reads+writes the residual with SIMD for one fused pass.

5. **Tile the matmul 2- or 4-wide on N.** The current kernel does 1
   output column per thread at a time; keeping 2–4 accumulators live
   reuses the loaded `x_vec` across more weight columns and pushes
   FMA throughput closer to peak. Expected ~1.5–2×.

6. **Native BF16 FMA on AVX-512 BF16 CPUs.** Not applicable to the
   7900X as configured; would halve bandwidth pressure on e.g. Xeon
   Sapphire Rapids or Granite Rapids.
