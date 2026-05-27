# Qwen3-14B CPU Inference Bench

MLRift Q8_0 vs PyTorch BF16, single-machine CPU, greedy decode, real
text prompt. Counterpart to `BENCH_QWEN3.md` (0.6B bench).

## Hardware / environment

| Item | Value |
|---|---|
| CPU | AMD Ryzen 9 7900X — 12 cores / 24 SMT threads, Zen 4 |
| SIMD path | AVX2 (no AVX-512 BF16/FP16 on Zen 4) |
| RAM | 30 GiB DDR5 |
| Swap | 43 GiB (~14 GiB in use during the run) |
| OS | Linux 6.17 |
| Worker threads | 16 (best wall-clock; same as the 0.6B bench) |
| Model | Qwen3-14B |
| MLRift weights | `Qwen3-14B-Q8_0.gguf` (~15 GiB on disk, mmap'd) |
| PyTorch weights | `Qwen3-14B` HF safetensors, BF16 (~28 GiB on disk) |
| Tokenizer | shared `tokenizer.json` (Qwen3 GPT-2 byte-level BPE) |
| Prompt | `"The capital of France is"` (5 tokens) |
| Generation | greedy, `max_new_tokens=20`, `use_cache=True` |

## Arithmetic used by each config

The two configurations are not the same dtype — this is an
**end-to-end CPU inference comparison** (real model, real prompt,
realistic memory budget) rather than an arithmetic-parity comparison.
The 0.6B bench has the parity numbers; at 14B the BF16 checkpoint
barely fits in 30 GiB RAM and the F32 checkpoint doesn't fit at all,
so we don't run F32.

| Config | Weight storage | Matmul compute |
|---|---|---|
| **MLRift GGUF Q8_0** | int8 + f16 scale per 32-element block, mmap'd | f32 FMA (AVX2 fused decode + matmul) |
| **PyTorch BF16** | bf16 in heap memory, streamed to f32 per matmul tile | f32 FMA |

## Results (20-token greedy, same prompt, same CPU)

Wall numbers below are the **decode-only** segment (20 generated
tokens after prompt prefill). Memory is the kernel
`RUSAGE_SELF.ru_maxrss` peak (matches `/usr/bin/time -v`).

| Config | Prefill (5 tok) | Decode (20 tok) | tok/s | Mean step | Peak RSS | vs PyTorch BF16 |
|---|---:|---:|---:|---:|---:|---:|
| **MLRift GGUF Q8_0** | **8 050 ms** | **41 712 ms** | **0.479** | **2 086 ms** | **14.81 GB** | **3.63×** |
| PyTorch BF16 (bf16 weights, f32 GEMM) | 12 016 ms | 151 366 ms | 0.132 | 7 568 ms | 20.32 GB | 1.00× |

Decode is **3.63× faster** at **1.37× less peak RSS**. Prefill is
also faster (1.49×) despite MLRift running prompt tokens
sequentially (one forward per token) while PyTorch batches them — the
Q8_0 bandwidth advantage outweighs the missing prefill batching.

### Correctness cross-check

The first **10 generated token ids are bit-identical** between MLRift
Q8_0 and PyTorch BF16:

```
12095, 13, 3555, 374, 279, 6722, 315, 279, 3639, 4180
```

(`Paris. What is the capital of the United States`)

Sequences diverge at step 10 — expected for Q8_0 vs BF16. Both
continuations are coherent English:

```
MLRift Q8_0  : The capital of France is Paris. What is the capital
               of the United States of America? The capital of the
               United States of
PyTorch BF16 : The capital of France is Paris. What is the capital
               of the United States? The capital of the United
               States is Washington,
```

The 10-token prefix match is a strong end-to-end correctness signal:
prompt encoding, prefill, KV cache layout, attention with GQA, RoPE,
QK-Norm, RMSNorm, lm_head, and argmax all line up across two
independent implementations on the same model.

## Why the speedup is bigger than at 0.6B

At 0.6B, MLRift was 1.24× over PyTorch BF16. At 14B it's 3.63×.
Two things change with size on this 30 GiB box:

1. **PyTorch's BF16 weights crowd the working set.** 14B BF16 = ~28 GiB
   resident — the OS must page-in/out around the small headroom and
   the 14 GiB of swap already in use. The decode loop touches all
   weights once per token; on a tight box that's death by minor
   page faults.

2. **MLRift's Q8_0 reads half the per-matmul bandwidth.** Q8_0 is
   34 bytes per 32-element block (1 byte/weight + 2 bytes scale per
   32 weights ≈ 1.06 B/wt). BF16 is 2 B/wt. The CPU is bandwidth-bound
   for matmul at this size, so halving the read is close to halving
   the time.

3. **mmap vs. heap-resident.** MLRift's 15 GiB GGUF lives on disk
   pages that the kernel can evict cheaply between tokens; PyTorch's
   28 GiB tensor is private heap that swaps to disk if the kernel
   can't keep it.

So this isn't a same-arithmetic claim — it's "MLRift's Q8_0 path is
3.6× faster than PyTorch's BF16 path on this CPU at 14B, and uses
1.37× less peak RSS."

## Per-step breakdown (MLRift, 20-token decode at 41.7 s)

Per-step wall time is dominated by the 7 matmuls per layer:
3 (Q/K/V) + 1 (O) + 1 (gate) + 1 (up) + 1 (down) = 7 per layer × 40
layers = 280 matmul submits per token. The `q8_14b_matmul_probe`
shows a single 5120×5120 Q8_0 matmul at ~5 ms hot, so the matmul
budget per token is ~280 × 5 ms / dispatch overhead ≈ ~2 s,
matching the observed 2.1 s/token.

The `gguf_f16_to_f32_exact` mmap-per-call bug landed in the same
session as this bench — fixing it dropped the matmul from 2 660 ms
to 5 ms (532×) and the per-layer time from 62 700 ms to ~50 ms
steady state. See `gguf: static scratch for f16_to_f32_exact` in
the git log for the writeup.

## Reproduce

```bash
# MLRift (from MLRift repo root):
./build/mlrc --emit=elfexe examples/qwen3_14b_q8_generate.mlr \
    -o /tmp/qwen3_14b
/usr/bin/time -v /tmp/qwen3_14b > /tmp/mlrift_14b.log 2>&1

# PyTorch (from MLRift-experimental/):
source venv/bin/activate
python bench_qwen3_pytorch.py bf16 16 \
    --model-dir /home/pantelis/Desktop/Projects/Work/MLRift-experimental/Qwen3-14b \
    --prompt 'The capital of France is' --max-new 20
```

## Things that worked at this size

Almost the entire stack from the 0.6B path generalized cleanly:

- **AVX2 matmul kernels** (Q8_0 in this run; BF16 in 0.6B) — same
  thread-pool, same per-worker scratch, same 2-wide N-tiling.
- **AVX2 SiLU + AVX2 attention + AVX2 RMSNorm** ported unchanged.
- **On-demand Q8_0 embed-row decode** — the embed table at 14B is
  3.1 GB if eagerly decoded to f32; decoding one row per token keeps
  RSS under 16 GB.
- **No QKV / gate-up fusion at 14B** — the 0.6B path fuses these
  three pairs of weights into stacked matrices, saving one matmul
  submit per pair. At 14B that would heap-resident a few extra GB
  of weight copies and OOM the box. The unfused path (`qwen3_set_fuse_qkv(0)`)
  trades two extra submits per layer for no duplication.
- **GPT-2 byte-level tokenizer** — Qwen3 needs the bytes_to_unicode
  bijection rather than SentencePiece's `space → ▁`. The MLRift
  tokenizer landed a `tokenizer_set_kind_gpt2(tk)` switch in the
  same session.

## Next push ideas

1. **Batched prompt prefill.** MLRift currently runs prefill tokens
   sequentially (`forward_layer` per token). A batched prefill
   (M-wide matmul instead of M=1) would beat PyTorch's prefill too.
2. **2-wide N-tiling on Q8_0.** The BF16 kernel got a ~5 % win at
   0.6B from running 2 output columns concurrently with independent
   FMA accumulators (commit `7b25eb5`). The Q8_0 worker is still
   1-wide; same trick should apply.
3. **Native AVX-512 BF16 / VNNI int8 FMA on supported CPUs.** Not on
   the 7900X but a Sapphire Rapids / Granite Rapids would benefit
   directly without changing the kernel shape.
4. **llama.cpp Q8_0 reference.** Same-quant fair comparison would
   answer "is MLRift's Q8_0 path competitive with the most-tuned
   public Q8_0 implementation" — distinct question from this doc's
   "MLRift Q8_0 vs PyTorch BF16 end-to-end".
