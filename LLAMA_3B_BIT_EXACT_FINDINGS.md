# Llama-3.2-3B GPU Mega-Kernel — Bit-Exact Investigation Findings

> ## ⚠️ RESOLVED 2026-05-14 — THE 11/20 METRIC WAS INVERTED
>
> **MLRift GPU is bit-exact with PyTorch.** PyTorch bf16 CPU and PyTorch fp32 CPU (transformers 5.8.1, unsloth/Llama-3.2-3B-Instruct mirror) both produce byte-identical 20-token greedy output to MLRift GPU mega-kernel — same token stream as all four GPU variants (bf16, fp32, bf16+swexp, fp32+swexp).
>
> The "11/20 vs CPU bf16 reference" framing measured MLRift CPU's AVX2 vhaddps reduction-tree against MLRift GPU's XOR-butterfly. PyTorch's reduction order happens to round-equal the GPU tree. MLRift CPU diverges. The "hello" prompt sits on a probability knife-edge at pos 11; either reduction order is arithmetically legitimate.
>
> Truth table for `prompt="hello"`, greedy, 20 new tokens:
>
> | Source | Output (continuation) |
> |---|---|
> | **PyTorch bf16 CPU** | `concept of "dark matter" and its implications for our` |
> | **PyTorch fp32 CPU** | `concept of "dark matter" and its implications for our` |
> | **MLRift GPU (all 4 variants)** | `concept of "dark matter" and its implications for our` |
> | MLRift CPU bf16 driver | `topic of "sustainable agriculture" and its impact on` |
> | Ollama Q8_0 (llama.cpp) | `topic of "sustainable agriculture" and its impact on` |
> | Ollama Q4_K_M | `new user here. I have a few questions about the website` |
>
> Token streams (PT bf16 == PT fp32 == MLRift GPU): `128000, 15339, 11, 358, 2846, 8173, 304, 6975, 810, 922, 279, 7434, 315, 330, 23449, 5030, 1, 323, 1202, 25127, 369, 1057`
>
> **Implications:**
> - Slice 8.1 (commit `4958215`, phase-7 non-pow-2 GQA libdivide fix) is the actual bug fix and is still load-bearing.
> - Slice 8.2-B Option A (6 commits `9e89233`..`3630a5a`, cputree gemv + IEEE-correct scalar pipeline) targeted the wrong reference. Benign no-op on tokens; gated to `MEGA_HIDDEN == 3072` so no impact on other archs. Possible future cleanup: revert and delete the gates.
> - Slice 8.3 fp32 path (CPU `matmul_f32_weights_f32_avx2` + GPU `_emit_gemv_cputree_f32_padded_strided_inline` + 5 CLI flags + uncommitted driver env-var matrix selector) was built to escape the false 11/20 — also produces byte-identical tokens to bf16. Don't ship as user-facing default.
> - The "Software exp poly port" (B path) was also built on the false premise. Same conclusion: byte-identical tokens.
>
> **PyTorch reference reproducer:**
> ```bash
> python3 -m venv /tmp/pt_venv --system-site-packages
> /tmp/pt_venv/bin/pip install transformers
> /tmp/pt_venv/bin/python /tmp/pt_ref.py bfloat16   # ~4 s
> /tmp/pt_venv/bin/python /tmp/pt_ref.py float32    # ~20 s
> # /tmp/pt_ref.py uses unsloth/Llama-3.2-3B-Instruct (gateless mirror), prompt="hello", greedy, max_new_tokens=20
> ```
>
> The historical narrative below is preserved for posterity; treat it as the journey, not the destination.

---

**Date range:** 2026-05-13 → 2026-05-14
**bf16 path state (HISTORICAL):** Llama-3B GPU was tracked at "11/20 token-match vs CPU bf16 reference" — the metric was later proven inverted (see banner above). bf16 GPU is bit-exact with PyTorch bf16 *and* PyTorch fp32. 41 tok/s, +86% over PT bf16 GPU baseline.
**Other archs (bf16):** Mistral 20/20, Llama-1B 20/20, Qwen3 20/20 — all GPU mega-kernel.
**Planned default (NO LONGER NEEDED):** fp32 weight path was being built to escape the 11/20 metric. Reframe collapsed the rationale.

---

## TL;DR — Slice 8.2-B journey (2026-05-14)

Six commits chased the bf16 reduction-tree drift from 19 ULP → fixable structural state:

| Commit | What | Llama-3B effect |
|---|---|---|
| `9e89233` | Phase 1: standalone bf16 gemv with vhaddps tree (validator) | 0 mismatches at K=3072/1024/4096 |
| `45a8c9c` | Phase 2: cputree gemv in mega-kernel phase 3 | `b_attn_q_g` per-element bit-exact (given input) |
| `d18448b` | Phase 2.5: cputree rmsnorm SSQ phases 1+11 | SSQ sum bit-exact |
| `d83bd3a` | Phase 2.6: IEEE-correct div+sqrt-rcp + mul-order | **`b_x_norm_g` fully bit-exact** |
| `e388636` | Phase 2.7: IEEE-correct inv_sum in phase-7 softmax | Structural improvement, no token change |
| `3630a5a` | Phase 3: cputree gemv phases 9 + 13 | `b_mid_g` and `gu_scratch` bit-exact |

**End-to-end Llama-3B: still 11/20** because phase 7 attention softmax has a structural floor:
- `v_exp_f32` (RDNA3 ~1 ULP approximate hardware) vs CPU's `exp_f32_fast` (custom 5-term minimax polynomial) — DIFFERENT approximations entirely. Porting CPU's polynomial to ISA is research-level work.
- Sum-reduce parallelism: CPU sequential fold over t-positions vs GPU XOR-butterfly over wave32 — no CPU-tree shape matches a sequential fold without destroying GPU parallelism.
- FMA-vs-mul+add semantics in Q·K dot, fused scale-adjust, V-axpy.

**Sister archs (Qwen3, Llama-1B, Mistral) remain 20/20 throughout** — verified via MD5-identical `.co` files. All cputree changes are gated on `MEGA_HIDDEN == 3072` (Llama-3B's non-pow-2 hidden).

**Why pow-2 archs escape the floor:** their reduction trees happen to round-equal CPU vhaddps for the specific weight values, and the softmax 1 ULP drift falls below their argmax-flip threshold for the tested token streams. Probabilistic, not structural.

## Planned next: fp32 weight path (not yet implemented)

The plan is to make Llama-3B GPU default to fp32 weight precision instead of bf16:
- Build a new CPU fp32-weights matmul (greenfield; only bf16-weights variant exists today)
- Build a parallel fp32 mega-kernel emit (mirror existing bf16 emit; replace `global_load_u16` + bf16-widen with `global_load_b32`; stride doubles)
- Driver uploads weights as f32 (2× VRAM: ~5.7 → ~11.4 GB, fits 16 GB card)
- Default: fp32 (hopefully 20/20). Opt-in bf16 via env var with documented 11/20 limitation.
- Caveat: phase-7 softmax floor is precision-independent — fp32 weights alone may not flip pos-11 argmax. Won't know until we try.

Reusable cputree+IEEE-div helpers from this session (`_emit_gemv_cputree_bf16_padded_strided_inline`, `asm_v_div_f32_ieee`, etc.) will be straightforward to port to fp32 weight loads (drop the bf16-widen step).

---

---

## Per-Model Exact Commands (copy-paste ready)

### Common setup (run once per session)

```bash
cd /home/pantelis/Desktop/Projects/Work/MLRift

# Rebuild mlrc (compiler) — only needed if you change src/*.mlr
make build
# expect: "0 error(s)" + "build/mlrc.new" → "build/mlrc"
```

### Common decoder env (run once)

```bash
export MLRIFT_LLAMA_N=20         # decode 20 tokens (same for all models)
# DO NOT set MLRIFT_MEGAK_DUMP_* unless you want per-phase diagnostics
```

---

### Model 1: Qwen3-0.6B (BIT-EXACT reference — the "gold standard")

**Arch:** HIDDEN=1024 (pow-2), N_HEADS=16 (pow-2), GQA=2, HEAD_DIM=128. All pow-2 → bit-exact by tree-luck.

**GGUF:** `MLRIFT_QWEN3_0_6B_DIR` env var (points to a DIRECTORY containing GGUF, NOT a file path). Default path resolved by `resolve_model_path()`.

**Driver:** `examples/qwen3_generate.mlr`

**Build .co + driver:**
```bash
./build/mlrc --emit-amdgpu-qwen3-megakernel-v2=/tmp/qwen3_layer_megakernel_v2.co examples/llm/qwen3_layer_megakernel.mlr
./build/mlrc --arch=x86_64 --target=amdgpu-native --emit=elfexe -o /tmp/qwen3_gen examples/qwen3_generate.mlr
# expect: "0 error(s)" + "/tmp/qwen3_gen" produced
```

**Run CPU baseline + GPU mega-kernel:**
```bash
echo "=== CPU ===";       /tmp/qwen3_gen 2>&1 | grep -A 2 "GENERATED IDs" | head -3
echo "=== GPU MK ===";    MLRIFT_NATIVE_MEGAKERNEL=1 /tmp/qwen3_gen 2>&1 | grep -A 2 "GENERATED IDs" | head -3
# expect: BOTH outputs identical (20/20 bit-exact match)
```

**Interpretation:** If GPU != CPU, you broke Qwen3 — that's the canary for "I damaged the bit-exact baseline." Revert and reconsider.

---

### Model 2: Llama-3.2-1B (BIT-EXACT — 22/22 tokens)

**Arch:** HIDDEN=2048 (pow-2), N_HEADS=32 (pow-2), GQA=4 (pow-2), HEAD_DIM=64 (pow-2). All pow-2. Required ATTN_COOP=2 refactor (slice 5.10) + 4 phase-7 bug fixes (slices 6.7-6.7e) before reaching bit-exact.

**GGUF:** `MLRIFT_LLAMA3_1B_GGUF` env var (or default `models/llama-3.2-1b-instruct-q8_0.gguf`).

**Driver:** `examples/llama3_1b_gpu_generate.mlr`

**Build .co + driver:**
```bash
./build/mlrc --emit-amdgpu-llama-megakernel-v2=/tmp/llama_layer_megakernel_v2.co examples/llm/llama_layer_megakernel.mlr
./build/mlrc --arch=x86_64 --target=amdgpu-native --emit=elfexe -o /tmp/llama32_1b_gpu examples/llama3_1b_gpu_generate.mlr
```

**Run CPU vs GPU:**
```bash
echo "=== CPU ===";    /tmp/llama32_1b_gpu 2>&1 | grep -A 2 "GENERATED IDs" | head -3
echo "=== GPU MK ==="; MLRIFT_NATIVE_MEGAKERNEL=1 /tmp/llama32_1b_gpu 2>&1 | grep -A 2 "GENERATED IDs" | head -3
# expect: BOTH identical, prompt "hello" → tokens starting 128000, 15339, 11, 1268, 649, ...
```

**Interpretation:**
- Bit-exact match → Llama-1B baseline preserved
- Any divergence → you broke Llama-1B; this would surface tree-luck-sensitive changes

---

### Model 3: Mistral-7B (BIT-EXACT — 20/20 tokens)

**Arch:** HIDDEN=4096 (pow-2), N_HEADS=32 (pow-2), GQA=4 (pow-2), HEAD_DIM=128. All pow-2. Required 2 emit fixes (lit32 patch in slice 7.2b at commit 40671db; phase-15 WG cap in slice 7.2c at 113f750).

**GGUF:** `MLRIFT_MISTRAL_7B_GGUF` env var. The ollama blob path is:
```bash
export MLRIFT_MISTRAL_7B_GGUF=/usr/share/ollama/.ollama/models/blobs/sha256-0624ba75c3cebd8c75b53cc7bcc3f344ad5a410ef74bfc68d757a9ac6764495a
```

**Driver:** `examples/mistral_7b_gpu_generate.mlr`

**Build .co + driver:**
```bash
./build/mlrc --emit-amdgpu-mistral-megakernel-v2=/tmp/mistral_layer_megakernel_v2.co examples/llm/mistral_layer_megakernel.mlr
./build/mlrc --arch=x86_64 --target=amdgpu-native --emit=elfexe -o /tmp/mistral_7b_gpu examples/mistral_7b_gpu_generate.mlr
```

**Run CPU vs GPU:**
```bash
echo "=== CPU ===";    /tmp/mistral_7b_gpu 2>&1 | grep -A 2 "GENERATED IDs" | head -3
echo "=== GPU MK ==="; MLRIFT_NATIVE_MEGAKERNEL=1 /tmp/mistral_7b_gpu 2>&1 | grep -A 2 "GENERATED IDs" | head -3
# expect: BOTH identical, prompt "hello" → tokens 1, 6312, 28709, 28725, 13, 13, 28710, 506, 264, 2996, ...
```

**Interpretation:**
- Bit-exact match → Mistral baseline preserved
- Any divergence → you broke the largest pow-2 K test case

---

### Model 4: Llama-3.2-3B (THE BROKEN ONE — 11/20 tokens)

**Arch:** HIDDEN=3072 (NON-pow-2), N_HEADS=24 (NON-pow-2), N_KV_HEADS=8, GQA=3 (NON-pow-2), HEAD_DIM=128, FF=8192, 28 layers, vocab=128256. The ONLY arch with non-pow-2 K.

**GGUF:** `MLRIFT_LLAMA3_3B_GGUF` env var. Ollama blob:
```bash
export MLRIFT_LLAMA3_3B_GGUF=/usr/share/ollama/.ollama/models/blobs/sha256-ed5cd7dbde6e2b5fb2d9926857ecf0f73ae3509ac1efd119ee54584d7a724688
```

**Driver:** `examples/llama3_3b_gpu_generate.mlr`

**Build .co + driver:**
```bash
./build/mlrc --emit-amdgpu-llama-3b-megakernel-v2=/tmp/llama_3b_layer_megakernel_v2.co examples/llm/llama_3b_layer_megakernel.mlr
./build/mlrc --arch=x86_64 --target=amdgpu-native --emit=elfexe -o /tmp/llama32_3b_gpu examples/llama3_3b_gpu_generate.mlr
```

**Run CPU vs GPU:**
```bash
echo "=== CPU ===";    /tmp/llama32_3b_gpu 2>&1 | grep -A 2 "GENERATED IDs" | head -3
echo "=== GPU MK ==="; MLRIFT_NATIVE_MEGAKERNEL=1 /tmp/llama32_3b_gpu 2>&1 | grep -A 2 "GENERATED IDs" | head -3
```

**Current expected output:**
```
CPU: 128000, 15339, 11, 358, 2846, 8173, 304, 6975, 810, 922, 279, 8712, 315, 330, 82,    42341, 30029, 1, 323, 1202, 5536, 389
GPU: 128000, 15339, 11, 358, 2846, 8173, 304, 6975, 810, 922, 279, 7434, 315, 330, 23449, 5030,  1,     323, 1202, 25127, 369, 1057
```
Match: 0-10 (11 tokens). Diverge: pos 11 onwards.

**Interpretation:**
- ≥11 tokens match → no regression (current baseline)
- Fewer tokens match → you regressed Llama-3B (rare but possible)
- ALL 20 tokens match → **you fixed it.** Verify other 3 archs still bit-exact, then celebrate.

---

## Sanity sweep (run all 4 in one go)

```bash
# Set GGUFs
export MLRIFT_MISTRAL_7B_GGUF=/usr/share/ollama/.ollama/models/blobs/sha256-0624ba75c3cebd8c75b53cc7bcc3f344ad5a410ef74bfc68d757a9ac6764495a
export MLRIFT_LLAMA3_3B_GGUF=/usr/share/ollama/.ollama/models/blobs/sha256-ed5cd7dbde6e2b5fb2d9926857ecf0f73ae3509ac1efd119ee54584d7a724688
export MLRIFT_LLAMA_N=20

# Rebuild all .co's
./build/mlrc --emit-amdgpu-qwen3-megakernel-v2=/tmp/qwen3_layer_megakernel_v2.co examples/llm/qwen3_layer_megakernel.mlr
./build/mlrc --emit-amdgpu-llama-megakernel-v2=/tmp/llama_layer_megakernel_v2.co examples/llm/llama_layer_megakernel.mlr
./build/mlrc --emit-amdgpu-mistral-megakernel-v2=/tmp/mistral_layer_megakernel_v2.co examples/llm/mistral_layer_megakernel.mlr
./build/mlrc --emit-amdgpu-llama-3b-megakernel-v2=/tmp/llama_3b_layer_megakernel_v2.co examples/llm/llama_3b_layer_megakernel.mlr

# Rebuild all drivers
./build/mlrc --arch=x86_64 --target=amdgpu-native --emit=elfexe -o /tmp/qwen3_gen        examples/qwen3_generate.mlr        2>&1 | tail -1
./build/mlrc --arch=x86_64 --target=amdgpu-native --emit=elfexe -o /tmp/llama32_1b_gpu  examples/llama3_1b_gpu_generate.mlr 2>&1 | tail -1
./build/mlrc --arch=x86_64 --target=amdgpu-native --emit=elfexe -o /tmp/mistral_7b_gpu  examples/mistral_7b_gpu_generate.mlr 2>&1 | tail -1
./build/mlrc --arch=x86_64 --target=amdgpu-native --emit=elfexe -o /tmp/llama32_3b_gpu  examples/llama3_3b_gpu_generate.mlr 2>&1 | tail -1

# Run all 4 (CPU + GPU)
for m in qwen3_gen llama32_1b_gpu mistral_7b_gpu llama32_3b_gpu; do
    echo "=========== $m ==========="
    cpu=$(/tmp/$m 2>&1 | grep -A 1 "GENERATED IDs" | tail -1)
    gpu=$(MLRIFT_NATIVE_MEGAKERNEL=1 /tmp/$m 2>&1 | grep -A 1 "GENERATED IDs" | tail -1)
    echo "CPU: $cpu"
    echo "GPU: $gpu"
    if [ "$cpu" = "$gpu" ]; then echo "  MATCH ✓"; else echo "  DIVERGE ✗"; fi
done
```

**Expected:**
- qwen3_gen: MATCH ✓
- llama32_1b_gpu: MATCH ✓
- mistral_7b_gpu: MATCH ✓
- llama32_3b_gpu: DIVERGE ✗ (the open issue)

---

## How to treat results (decision tree)

After any change to `src/format_amdgpu_megakernel.mlr`:

```
Run sanity sweep ↑
│
├─ All 4 match?
│  └─ ✓ You're golden — Llama-3B FIXED, others preserved. Commit, ship.
│
├─ Qwen3 ✓ + Llama-1B ✓ + Mistral ✓ + Llama-3B closer (>11/20)?
│  └─ Progress! Investigate the Llama-3B drift further, but don't break others.
│
├─ Qwen3 ✗ OR Llama-1B ✗ OR Mistral ✗?
│  └─ REGRESSION — revert: `git reset --hard pre-wmma-revert-point`
│     The pow-2 luck broke. Your change interacted with their tree alignment.
│
└─ Compile error / NaN output / hang?
   └─ Bug in your emit. Disassemble .co to check (llvm-objdump --mcpu=gfx1100).
     Common: missing s_delay_alu, wrong VGPR alloc, kernarg offset typo.
```

---

## Per-Phase Diagnostic Reproducer (for bisection)

When you suspect a specific phase has a bug, dump CPU vs GPU intermediates at the divergence step:

```bash
export MLRIFT_LLAMA_N=9                # decode 9 tokens (enough to reach divergence)
export MLRIFT_MEGAK_DUMP_LAYER=0       # dump LAYER 0
export MLRIFT_MEGAK_DUMP_POS=9         # dump at decode POSITION 9 (where Llama-3B drift starts)
export MLRIFT_MEGAK_DUMP_STEP=8        # GPU-side step index (= pos - prompt_count = 9 - 1 = 8)

echo "=== CPU ==="
/tmp/llama32_3b_gpu 2>&1 | sed -n '/step 8 pos/,/step 9 pos/p' | grep -E "\[|sum="

echo "=== GPU ==="
MLRIFT_NATIVE_MEGAKERNEL=1 /tmp/llama32_3b_gpu 2>&1 | sed -n '/step 8 pos/,/step 9 pos/p' | grep -E "\[|sum="
```

**Expected dump fields:** `x_in_PRE`, `b_x_norm_g` (post phase 1), `b_qkv_g` (post phase 3 — GPU only), `b_attn_q_g` (post phase 5), `b_attn_out_g` (post phase 7), `b_mid_g` (post phase 9), `b_mid_norm_g` (post phase 11), `gu_scratch` (post phase 15), `out_resid` (post phase 17 = layer 0 final).

**Current diagnosis at step 8 layer 0:**
- `b_x_norm_g` bit-exact CPU=GPU ✓ (phase 1 is clean)
- `b_attn_q_g` diverges 19 ULP (phase 3 QKV gemv is the source)
- All downstream phases inherit the drift

The same diagnostic infrastructure works for any model — change the driver binary path. To bisect Llama-1B if it ever regresses:
```bash
export MLRIFT_LLAMA_N=22 MLRIFT_MEGAK_DUMP_LAYER=0 MLRIFT_MEGAK_DUMP_POS=N MLRIFT_MEGAK_DUMP_STEP=N-2
/tmp/llama32_1b_gpu ...
MLRIFT_NATIVE_MEGAKERNEL=1 /tmp/llama32_1b_gpu ...
# Find first phase where sums diverge.
```

---

## Baseline Reproducer

```
export MLRIFT_LLAMA3_3B_GGUF=/usr/share/ollama/.ollama/models/blobs/sha256-ed5cd7dbde6e2b5fb2d9926857ecf0f73ae3509ac1efd119ee54584d7a724688
./build/mlrc --emit-amdgpu-llama-3b-megakernel-v2=/tmp/llama_3b_layer_megakernel_v2.co examples/llm/llama_3b_layer_megakernel.mlr
./build/mlrc --arch=x86_64 --target=amdgpu-native --emit=elfexe -o /tmp/llama32_3b_gpu examples/llama3_3b_gpu_generate.mlr
MLRIFT_NATIVE_MEGAKERNEL=1 MLRIFT_LLAMA_N=20 /tmp/llama32_3b_gpu
```

**Current tokens (CPU vs GPU, prompt "hello"):**
```
CPU: 128000, 15339, 11, 358, 2846, 8173, 304, 6975, 810, 922, 279, 8712, 315, 330, 82,    42341, 30029, 1, 323, 1202, 5536, 389
GPU: 128000, 15339, 11, 358, 2846, 8173, 304, 6975, 810, 922, 279, 7434, 315, 330, 23449, 5030,  1,     323, 1202, 25127, 369, 1057
                                                                       ^^^ first divergence at pos 11
```

Match: positions 0-10 bit-exact (11 tokens). Pos 11 onwards drifts.

**Perf:** 41.07 tok/s decode (+86% over PT bf16 GPU baseline ~22).

---

## Per-Step Layer-0 Bit-Exact Bisection

| Step | Pos | Match? | Notes |
|---|---|---|---|
| 0 | 1 | ✅ BIT-EXACT | All phase outputs identical |
| 1-5 | 2-6 | ✅ BIT-EXACT | |
| 6 | 7 | ✅ BIT-EXACT | 16/16 head values identical |
| 7 | 8 | ✅ BIT-EXACT | |
| 8 | 9 | ❌ FIRST DRIFT | 1 ULP at one head element (0.000200 vs 0.000199) |
| 9-10 | 10-11 | ❌ DRIFT | accumulates |
| 11 | 12 | ❌ ARGMAX FLIP | tokens diverge |

---

## Per-Phase Divergence at Step 8 Layer 0 (the canary)

| Phase | CPU sum | GPU sum | Status |
|---|---|---|---|
| x_in_PRE (input) | -0.482374 | (match) | ✓ |
| **b_x_norm_g** (post phase 1 rmsnorm) | -3.564891 | -3.564891 | ✅ BIT-EXACT |
| **b_attn_q_g** (post phase 3+5 qkv+rope) | -126.695312 | -126.695293 | ❌ **19 ULP DIVERGE** |
| b_attn_out_g (post phase 7 attn) | 0.910227 | 0.910227 | ~match |
| b_mid_g (post phase 9 o_proj+resid) | 0.392827 | 0.392828 | 1 ULP |
| b_mid_norm_g (post phase 11 rmsnorm) | -1.282853 | -1.282848 | ~5 ULP |
| gu_scratch (post phase 15 silu_mul) | 1.297695 | 1.297693 | 2 ULP |
| out_resid (layer-0 final) | -0.084937 | -0.084935 | 2 ULP |

**Conclusion:** Divergence originates at phase 3 (QKV gemv) where `b_attn_q_g` diverges by ~19 ULP at the sum level (small per-element diff, but accumulates).

---

---

## How the OTHER 3 Archs Got to Bit-Exact (Replication Recipe)

**Critical pattern:** Every prior mega-kernel arch had MULTIPLE structural bugs found via per-phase bisection. After all structural bugs fixed, the resulting code happened to be bit-exact "by luck" of pow-2 K alignment with CPU vhaddps tree. Llama-3B has had ONE round of bisection (the libdivide fix); it may have more structural bugs hiding under the precision drift.

### Qwen3-0.6B (the original — first model to reach bit-exact)
Slices `4.19.0` → `4.19.11` (9-phase mega-kernel built phase by phase):
- 4.19.0: scaffold (mega_barrier + multi-phase recognizer + A/B harness)
- 4.19.1: phase 1 rmsnorm-wg-gated
- 4.19.2: phase 3 gemv-coop-bf16-padded-strided
- 4.19.3: phase 5 head-per-wg-qkv-fused
- 4.19.4: phase 7 attn-decode-coop
- 4.19.5: phase 9 gemv-residual-fused
- 4.19.6-9: phases 11, 13, 15, 17
- 4.19.10: **e2e 20-tok bit-exact + 3-run deterministic**
- 4.19.11: cleanup byte-wrapper artifacts (hipcc dependency removal)

Final state: Qwen3 HIDDEN=1024 (pow-2), N_HEADS=16 (pow-2), GQA=2 (pow-2), HEAD_DIM=128 — every dimension pow-2. Bit-exact came naturally.

### Llama-3.2-1B (parametric refactor + 4 bug fixes)
Slices `5.0` → `5.10` (parametrize Qwen3-only emit) + `6.7*` (bisect bugs):
- 5.0: arch globals + setters + name-dispatch
- 5.1-5.6: parametrize each phase emitter (no behavior change)
- 5.7-5.9: Llama M=1 body + driver wiring; e2e BLOCKED at phase 7
- **5.10 (commit 30ca5bd)**: phase 7 ATTN_COOP_arch = HEAD_DIM/WAVE — fixed OUT_PER_WG=WAVE invariant break for HEAD_DIM=64 (Qwen3=128 → ATTN_COOP=4; Llama-1B=64 → ATTN_COOP=2; lanes 16..31 were corrupting next coop region)

Then bisection found 4 bugs in Llama-1B phase-7 / mega-kernel:
- 6.7 (commit 380b3ec): phase 5 RoPE lane-stride fix (necessary but insufficient)
- 6.7b (commit 8f7e129): dump rig found root cause #2 — norm-weight bf16/f32 mismatch
- 6.7c (commit 4f69a79): gate norm-weight upload by qwen3_norm_kind
- 6.7d (commit 1fb4074): per-WG weight base offset fix (bug #3 — every-8 broadcast)
- 6.7e (commit 9203594): **phase-7 softmax scaling (bug #4)** → Llama-1B BIT-EXACT 22/22

Final state: Llama-1B HIDDEN=2048 (pow-2), N_HEADS=32 (pow-2), GQA=4 (pow-2), HEAD_DIM=64 (pow-2). After 4 bugs fixed, bit-exact came from pow-2 luck.

### Mistral-7B (4 bugs across slice 7.x)
Slices `7.0` → `7.5`:
- 7.0: arch setter + dispatcher + body + emit flag (scaffold)
- 7.1: driver clone of llama3_1b with Mistral arch knobs
- 7.2: wire hidden=4096 to mega-kernel + vocab=32000
- 7.2a: madvise pattern for host-RSS hardening
- **7.2b (commit 40671db)**: phase-7 outer-gate lit32 patch bug — outer_cap>64 (Mistral 32*4=128) triggered bug where `s_cbranch_scc1` lived at `outer_gate_pos+2` (not +1) because lit32 cmp takes 2 dwords vs 1 dword inline-imm. **Mistral hang resolved.**
- **7.2c/d (commit 113f750)**: phase-15 hardcoded WG cap of 256 → `(MEGA_FF+31)/32`. Mistral FF=14336 needs 448 WGs; previous code only processed first 8192 of 14336. **Mistral BIT-EXACT 20/20.**
- 7.3: bench vs PT bf16 GPU + ship
- 7.4: host-RSS −6.2 GiB via post-fuse dealloc
- 7.5: 8× inner-gemv unroll → +37% Mistral perf

Final state: Mistral HIDDEN=4096 (pow-2), N_HEADS=32 (pow-2), GQA=4 (pow-2), HEAD_DIM=128 (pow-2). Same pattern: 2 bugs fixed, then bit-exact from pow-2 luck.

### Bit-Exact Replication Recipe (general)

Each model's bit-exact journey followed this pattern:
1. **Scaffold:** arch setter, name-dispatch, driver, .co emitter (no behavior change)
2. **Run e2e:** observe token output. Garbage / coherent-but-wrong / partial-match.
3. **Per-phase dump diagnostic:** add CPU + GPU dump of each phase intermediate at a specific step/layer/pos
4. **Bisect:** find the first phase whose output diverges from CPU. That's where the bug is.
5. **Fix:** patch the emit for that phase
6. **Repeat steps 2-5 until 20/20**
7. **If all structural bugs fixed and STILL drift:** likely bf16-precision sensitivity (e.g., Llama-3B current state)

**The diagnostic infrastructure (already wired in `qwen3_forward_layer` + GPU drivers):**

```bash
# Compare ANY phase intermediate between CPU and GPU at any step/layer/pos:
export MLRIFT_MEGAK_DUMP_LAYER=N    # which layer (0..n_layers-1)
export MLRIFT_MEGAK_DUMP_POS=K       # decode position (1..max_seq)
export MLRIFT_MEGAK_DUMP_STEP=S      # GPU-only: decode step

# CPU side dumps phases x_in_PRE, b_x_norm_g, b_qkv_g, b_attn_q_g, b_attn_out_g,
# b_mid_g, b_mid_norm_g, gu_scratch, out_resid (in qwen3.mlr forward layer)
# GPU side dumps same names via D2H copy from device buffers (in driver)
```

### What this means for Llama-3B specifically

Llama-3B is the ONLY arch with non-pow-2 K (HIDDEN=3072). Slice 8.1 fixed ONE structural bug (the non-pow-2 GQA libdivide). **It may have additional structural bugs that the precision drift is masking.**

Things to look for if more bugs exist:
- A phase emit using N_HEADS=24 as if it were pow-2 somewhere we haven't checked
- A specific (N_HEADS, HIDDEN) combination triggering an emit edge case
- Speck4 path (M=4 multi-token) potentially has bugs — but isn't exercised in M=1 driver, so wouldn't show up

**Recommended next bisection round for Llama-3B:** add a per-phase dump at step 8 layer 0 (already done — see `Per-Phase Divergence at Step 8 Layer 0` section above). The dumps already show phase 3 b_attn_q_g is the divergence origin. The question is whether the ~19 ULP drift at phase 3 is genuinely just bf16-tree-topology (Hypothesis H1, current belief) or whether there's a residual structural bug in phase 3 emit for K=3072.

**Untested verification of H1:** Run the SAME forward pass through ALL FOUR archs' phase 3 emitter on the exact same input (random weights + activations), compare outputs. If Llama-3B is the ONLY one where CPU vs GPU diverges by ULPs, H1 is confirmed (it's the K=3072 unlucky alignment). If MULTIPLE archs diverge but only Llama-3B's drift accumulates enough to flip tokens, H1 is still confirmed (just more nuanced).

If you find evidence of a structural bug (e.g., a specific K-position where ALL archs give the same wrong answer), that's a smoking gun.

---

## Slice 8.1 Fix (Shipped, commit 4958215)

**The structural fix:** Phase 7 attn_decode_coop's `kv_head = q_head / gqa_ratio` was computed via `s_lshr_b32 s19, s16, _arch_log2_pow2(N_HEADS/N_KV_HEADS)`. For Llama-3B GQA ratio = 24/8 = 3 (non-pow-2), `_arch_log2_pow2(3)` silently returns 1 (treats 3 as 2 by truncation). Pre-fix: all Q heads routed to wrong KV slots, q_head≥16 read OOB → token gibberish (220, 220, 220, ...).

**Fix:** `_emit_div_kv_head` helper in `src/format_amdgpu_megakernel.mlr`:
- pow-2 ratios (Qwen3 2, Llama-1B 4, Mistral 4): unchanged `s_lshr_b32` byte
- ratio==3: `s_mul_hi_u32 sdst, ssrc, 0xAAAAAAAB` + `s_lshr_b32 sdst, sdst, 1`
  (libdivide magic — 3·0xAAAAAAAB = 2^33+1, so high_u32(q·M) >> 1 == q/3)

Verified in disassembly at offset 0x24FC: `9693FF10 AAAAAAAB` + `85138113`.

Result: 0/20 garbage → 11/20 bit-exact + coherent text + 41 tok/s.

---

## Root Cause of Remaining 11/20 Drift

**Definitive answer (verified by slice 8.2a precision test):** bf16 reduction-tree topology mismatch between CPU and GPU.

**CPU `qwen3_dot_avx2`** (`std/qwen3.mlr:639-684`):
- ymm0 holds 8 parallel f32 lanes
- `vfmadd231ps ymm0, w_chunk, x_chunk` per K-iter of 8: ymm0[j] accumulates K-indices {j, j+8, j+16, ...} (stride-8 distribution)
- After K-loop:
  - `vextractf128 xmm1, ymm0, 1` + `vaddps xmm0, xmm0, xmm1`: 8→4 (pair i with i+4)
  - `vhaddps xmm0, xmm0, xmm0` × 2: 4→1 via tree
  - Final topology: `((a+e)+(b+f)) + ((c+g)+(d+h))` where letters = ymm0 lanes

**GPU `_emit_gemv_coop_bf16_padded_strided_inline`** (`src/format_amdgpu_megakernel.mlr:1110-1388`):
- 32 cooperative HW lanes per output row
- Each lane sequentially accumulates K/32 elements (stride-1 distribution per lane)
- Per outer iter (8× unrolled): 16 fmacs into single accumulator v2
- Cross-lane reduce: 5-level XOR-butterfly (offsets 16, 8, 4, 2, 1)
- Final topology: different from CPU's vhaddps tree

For pow-2 K (Qwen3 1024, Llama-1B 2048, Mistral 4096), both trees happen to round-equal at the f32 level for the specific weight/activation values → 20/20 tokens match (lucky alignment).

For non-pow-2 K=3072 (Llama-3B), trees don't align → ~19 ULP diff at gemv output → compounds through 28 layers × 4 gemvs/layer = 112 gemvs per token → argmax flips at pos 11.

**This is NOT a compiler bug.** Four parallel agent audits confirmed no codegen bug for Llama-3B. The drift is mathematical fp32 non-associativity.

---

## Things We Tried

### ✓ Verified working (no improvement, no regression)
1. **Slice 8.1 libdivide fix** for q_to_kv_shift — fixed the structural bug, got from 0/20 to 11/20.
2. **CPU rmsnorm mul-by-reciprocal change** — switched `mean_sq = sum_sq / int_to_f32(dim)` to `mean_sq = sum_sq * (1.0f / int_to_f32(dim))`. CPU output unchanged → confirmed rmsnorm scale precision is NOT the bottleneck. (Reverted.)

### ✓ Audits that ruled OUT compiler bugs (Agents A/B/C/D)
1. **gpu_get_or_upload_bf16_weight_padded** — verified `alloc()` returns zeroed memory; padding bytes correctly zero; no power-of-2 assumptions.
2. **q8_0_to_bf16_alloc** — no power-of-2 assumptions; weights dequant correctly.
3. **FFMA peephole, `_arch_inv_n_f32_bits`, s_clause encoding, inline-imm masking** — all verified safe.
4. **RoPE cos/sin table for HEAD_DIM=128 + N_HEADS=24** — Mistral uses identical path bit-exact, so RoPE isn't the issue.
5. **MEGA_N_HEADS pow-2 audit** — only speck4 emits use log2(N_HEADS) (line 3230); speck4 isn't exercised in Llama-3B M=1 path.
6. **Disassembly verification of Llama-3B .co** — `s_mul_hi_u32 s19, s16, 0xAAAAAAAB` correctly emitted (0x9693FF10 + 0xAAAAAAAB); MEGA_INV_HIDDEN_BITS 0x39AAAAAB correctly in .text.

### ✓ Slice 8.2a: WMMA precision validation
Built `examples/llm/gemv_bf16f32_wmma_megalayout_launch.mlr` (REVERTED from main 2026-05-14). Used Llama-3B QKV shape K=3072, N=5120, K_PAD=3200.

| Test | What it measured | Result |
|---|---|---|
| [A] CPU(f32 x) vs WMMA | End-to-end with f32→bf16 cast | 0/5120 bit-exact, 87% drift >1e-3 |
| [B] CPU(bf16 x) vs WMMA | Pure reduction-tree, same bf16 inputs | **99.98% within 1 bf16-ULP** |

**Key insight:** WMMA's hardware reduction tree is EFFECTIVELY EQUIVALENT to CPU AVX2 vhaddps tree when given identical bf16 inputs. The 87% drift in [A] is from the f32→bf16 activation cast WMMA forces, NOT from reduction order.

This means: replacing CPU vhaddps with GPU XOR-butterfly is what causes Llama-3B's drift (the GPU mega-kernel does XOR-butterfly, NOT vhaddps). If GPU did vhaddps-equivalent (like WMMA), Llama-3B might bit-match — but WMMA also forces activation truncation which is WORSE than the current drift.

### ✗ Things tried that didn't pan out
1. **Agent attempt to monolithically port phase 3 to WMMA** — agent correctly refused after analysis; would have introduced activation truncation drift across all 4 archs.
2. **Naive WMMA implementation** — `[K, N]` layout vs `[N, K_PAD]` mismatch means significant emit restructuring needed (swap A↔B roles).

---

## Hypotheses (Ranked by Plausibility)

### H1 (CONFIRMED): bf16 reduction tree topology mismatch
- CPU vhaddps: `((a+e)+(b+f)) + ((c+g)+(d+h))`, offsets [4,1,2]
- GPU XOR-butterfly: different topology, offsets [16,8,4,2,1] across 32 lanes
- For pow-2 K, accidents of f32 rounding align both trees
- For K=3072, they don't align
- **Status:** Confirmed via per-phase bisection + WMMA precision test. NOT a compiler bug.

### H2 (PLAUSIBLE BUT INCONCLUSIVE): CPU forward might have non-obvious branch
- Agent audit found qwen3_forward_layer has NO dimension-specific branches for Llama-3B vs Mistral
- All matmuls dispatch on `qwen3_fuse_qkv` (fused vs unfused), not on K/N dims
- AVX2 vs scalar fallbacks only for unaligned tails, not dim-triggered
- **Conclusion:** No CPU branch is unique to Llama-3B
- **But:** could there be something in driver setup (e.g., a missing `qwen3_set_*` call in the Llama-3B driver vs Mistral driver)? Worth a 10-min diff before final ruling.

### H3 (RULED OUT): Compiler bug specific to Llama-3B values
- Four parallel agents found no codegen bug for HIDDEN=3072, N_HEADS=24, FF=8192, K_PAD=3200, qkv_dim=5120
- All hardcoded literals verified
- Disassembly confirms correct emission
- **Status:** Ruled out

### H4 (RULED OUT): Memory bug (uninitialized padding leaking)
- Agent 2 hypothesized that `q8_0_to_bf16_alloc`'s heap block might leak stale data through `gpu_get_or_upload_bf16_weight_padded`'s padding
- Agent 1 confirmed `alloc()` returns zeroed memory; padding always zero
- **Status:** Ruled out

### H5 (UNTESTED): Step-specific input pattern
- Step 0-7 bit-exact, step 8 first 1-ULP diff
- This is value-dependent rounding (NOT structural)
- Maybe step 8's specific x_in vector happens to trigger a value where both trees round differently
- **Status:** Untested. Could verify by running with a SHORTER prompt (e.g., longer prompts might shift the divergence to a different step, which would tell us drift is per-token not per-step-count).

### H6 (PARTIALLY TESTED): Maybe drift compounds DIFFERENTLY for Llama-3B because of HIDDEN=3072
- Mistral HIDDEN=4096, 32 layers
- Llama-1B HIDDEN=2048, 16 layers
- Llama-3B HIDDEN=3072, 28 layers
- Llama-3B has more layers than Llama-1B but fewer than Mistral, yet only Llama-3B drifts
- Suggests it's the HIDDEN value (3072 non-pow-2) more than layer count
- **Status:** Supported by data, no fix derived from it.

---

## Untested Experiments (Low-Hanging)

These were NOT tried; worth ~30 min each:

### E1: Run Llama-3B GPU with a DIFFERENT prompt
Currently testing "hello" (2 tokens). Try a 10-token prompt and see if drift starts at step 0+10=10 (consistent with "drift starts at decode pos≈11") or stays at "11 tokens from start" (consistent with "drift compounds per gemv from prompt start").

### E2: Run with Q8_0 CPU reference instead of bf16 CPU reference
Memory notes say "Q8_0×Q8_0 INT8 matmul for bit-exact match with llama.cpp" — Llama-3B's CPU reference might be Q8_0×Q8_0 (more precise) while GPU is bf16. Verify CPU is using bf16 (qwen3_forward_layer) NOT Q8_0×Q8_0 (qwen3_main_matmul with q8_0_kind).

### E3: Diff Llama-3B driver vs Mistral driver for `qwen3_set_*` calls
A missing setting (norm_kind, fuse_qkv, etc.) could cause divergence in CPU vs GPU dispatch.

### E4: Run Llama-3B GPU with MLRIFT_LLAMA_N=8 and check tokens
If tokens 0-7 still bit-match, the drift is truly at pos 11 (regardless of total length).

### E5: Try Kahan summation per-lane in GPU gemv inner loop
Add error-compensation term (4 vregs: y, t, c, p per accumulator). Cost: 3-5× slower. But would tell us definitively whether scalar-tree precision is the issue (if Kahan gives 20/20 → confirmed; if not → some other issue exists).

### E6: fp32 weights for Llama-3B GPU (only)
Force Llama-3B to use fp32 weights instead of bf16. Tests whether weight precision is the issue. Cost: ~3× more VRAM for this model. Implementation: would need new gemv emit OR convert all bf16 to fp32 at upload time.

### E7: Use BOTH CPU and GPU as bf16 (force CPU to truncate activations)
If we modify CPU to also truncate activations to bf16 (matching GPU's behavior), and if GPU was using WMMA, they'd match. Tests whether activation precision is the dominant issue (we'd expect them to match closely).

### E8: Compare to llama.cpp's GPU output (not CPU)
Currently we compare against MLRift's CPU bf16. What does llama.cpp's ROCm GPU produce? If llama.cpp also drifts at pos 11 vs its CPU, then MLRift's behavior is normal and matches the broader ecosystem.

---

## Strategic Options Going Forward

### Option A: Ship Llama-3B at 11/20 (recommended in this session)
- Cost: 0
- Functional output, coherent text, 41 tok/s (+86% PT)
- Move on to Gemma3-1B / Gemma2-2B / Qwen3-14B
- Llama-3B documented as "bf16-precision-sensitive" model

### Option B: Kahan summation on GPU gemv scalar accumulator
- Cost: 3-5× slower per gemv (~14 tok/s instead of 41)
- Single emit-site change (`_emit_gemv_coop_bf16_padded_strided_inline`)
- Mathematically near-bit-exact regardless of tree
- Probably gives Llama-3B 20/20
- Probably preserves other archs' 20/20 (since Kahan is order-independent)
- ~1-2 days of work

### Option C: Reduction-tree match (vhaddps emulation on GPU)
- Cost: 8 emit sites (phases 3, 9, 13, 17 × base + speck4)
- Multi-day implementation
- May or may not preserve other archs' 20/20 (depends on whether they were lucky on XOR-butterfly OR vhaddps)
- Worst-case scenario: all archs regress

### Option D: WMMA + double-decomposition + Kahan
- Cost: 2-3 weeks
- Tensor-core perf win (4-8× faster gemv)
- Near-f32 precision via emulated f32 (2 bf16 mults per f32)
- Probably gives Llama-3B 20/20 AND speeds everyone up
- Highest impact, highest effort

### Option E: bf16-ify entire mega-kernel pipeline (then WMMA works clean)
- Cost: 2-3 weeks
- Cleaner architecture
- Breaks CPU vs llama.cpp bit-exact baseline (CPU also needs bf16 activations to match)
- Net: tensor-core perf + bit-exact GPU-vs-MLRift-CPU, but loses GPU-vs-llama.cpp bit-exactness

---

## Key Files

| File | Role |
|---|---|
| `src/format_amdgpu_megakernel.mlr` | Mega-kernel emit; phase 3 gemv at lines 1110-1388 |
| `examples/llm/llama_3b_layer_megakernel.mlr` | Llama-3B kernel source |
| `examples/llama3_3b_gpu_generate.mlr` | Driver with diagnostic dump rig |
| `std/qwen3.mlr` | CPU forward layer; `qwen3_dot_avx2` at line 639 |
| `std/matmul.mlr` | AVX2 matmul reference |
| `std/inference_gpu.mlr` | `gpu_get_or_upload_bf16_weight_padded` at line 570 |

## Key Memory Files

| File | Contents |
|---|---|
| `~/.claude/projects/-home-pantelis-Desktop-Projects-Work-MLRift/memory/project_llama_3b_gpu.md` | Full slice 8.1 detail + bisection |
| `~/.claude/projects/.../memory/project_wmma_megakernel_plan.md` | WMMA design + slice 8.2a results |

## Reproducible Diagnostic Commands

```bash
# Per-phase intermediate dump (compares CPU vs GPU at step 8 layer 0):
export MLRIFT_LLAMA3_3B_GGUF=/usr/share/ollama/.ollama/models/blobs/sha256-ed5cd7dbde6e2b5fb2d9926857ecf0f73ae3509ac1efd119ee54584d7a724688
export MLRIFT_LLAMA_N=9 MLRIFT_MEGAK_DUMP_LAYER=0 MLRIFT_MEGAK_DUMP_POS=9 MLRIFT_MEGAK_DUMP_STEP=8

# CPU side dumps phase intermediates:
/tmp/llama32_3b_gpu 2>&1 | sed -n '/step 8 pos/,/step 9 pos/p' | grep -E "\[|sum="

# GPU side dumps phase intermediates:
MLRIFT_NATIVE_MEGAKERNEL=1 /tmp/llama32_3b_gpu 2>&1 | sed -n '/step 8 pos/,/step 9 pos/p' | grep -E "\[|sum="
```

```bash
# Token-match test (full 20 decode):
unset MLRIFT_MEGAK_DUMP_LAYER MLRIFT_MEGAK_DUMP_POS MLRIFT_MEGAK_DUMP_STEP
export MLRIFT_LLAMA_N=20
echo "CPU:"; /tmp/llama32_3b_gpu 2>&1 | grep -A 2 "GENERATED IDs" | head -3
echo "GPU:"; MLRIFT_NATIVE_MEGAKERNEL=1 /tmp/llama32_3b_gpu 2>&1 | grep -A 2 "GENERATED IDs" | head -3
```

---

## Summary One-Liner

Llama-3B GPU 11/20 drift is mathematically real bf16 reduction-tree non-associativity for non-pow-2 K=3072 — not a compiler bug, not fixable by WMMA (which makes it worse via activation truncation), but FIXABLE by either Kahan-summation in the gemv inner loop (3-5× slowdown) or a full WMMA+double-decomp+Kahan rewrite (multi-week). Current state ships and works functionally.
