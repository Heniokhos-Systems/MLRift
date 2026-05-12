#!/bin/bash
# rebuild_helper_cos.sh — rebuild every `/tmp/*.co` the GPU runtime expects.
#
# Two emit paths exist in mlrc, with different invocation conventions:
#
#   1. Single-flag emit (`--emit-amdgpu-<kernel>=<path>`):
#        Produces a `.co` directly from the AST-walker emitters.
#        Used for the ~25 kernels with a dedicated flag in src/main.mlr.
#
#   2. Compile a source as a HIP target (`--arch=x86_64 --target=hip-amd src.mlr -o stub`):
#        Writes <stub>.hip, forks hipcc to compile it into <stub>.co.
#        Requires hipcc on PATH. Used for the 4 legacy kernels that
#        predate the AST-walker recognizers (bf16_to_f32, gemv_f32,
#        gemm_f32, residual_add_f32).
#
#   3. The single-stream `silu_mul_f32` kernel needs path #1's AST-walker
#      recognizer (`amdgpu_lower_silu_mul_3a`), invoked via
#      `--target=amdgpu-native` against the source file (NOT a flag).
#      format_hip cannot lower the `silu_f32(g)` call inside its body.
#
# Usage:
#   scripts/rebuild_helper_cos.sh [path/to/mlrc]
#
# Defaults to ./build/mlrc if no path given.
set -e
MLRC="${1:-${MLRC:-./build/mlrc}}"
[ -x "$MLRC" ] || { echo "rebuild_helper_cos: mlrc not executable: $MLRC" >&2; exit 1; }

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

mkdir -p /tmp
echo 'fn main() {}' > /tmp/empty.mlr

emit_flag() {
    # $1 = flag=/tmp/foo.co  (single-flag emit)
    local fp="$1"
    if "$MLRC" --arch=x86_64 "$fp" /tmp/empty.mlr > /dev/null 2>&1; then
        echo "  ok  $fp"
    else
        echo "  FAIL $fp" >&2
        return 1
    fi
}

emit_hip_amd() {
    # $1 = examples/llm/X.mlr → /tmp/X.co (via hipcc side product)
    local src="$1"
    local stub_path="$2"
    if "$MLRC" --arch=x86_64 --target=hip-amd "$src" -o "$stub_path" > /dev/null 2>&1; then
        echo "  ok  $src → ${stub_path}.co"
    else
        echo "  FAIL $src" >&2
        return 1
    fi
}

emit_amdgpu_native() {
    # $1 = examples/llm/X.mlr → /tmp/X.co (via AST-walker)
    local src="$1"
    local stub_path="$2"
    if "$MLRC" --target=amdgpu-native "$src" -o "$stub_path" > /dev/null 2>&1; then
        echo "  ok  $src → ${stub_path}.co"
    else
        echo "  FAIL $src" >&2
        return 1
    fi
}

echo "=== path 1: single-flag AST-walker emits ==="
emit_flag '--emit-amdgpu-gemv-coop-f32=/tmp/gemv_coop_f32.co'
emit_flag '--emit-amdgpu-gemv-coop-bf16-f32=/tmp/gemv_coop_bf16_f32.co'
emit_flag '--emit-amdgpu-gemv-coop-f32-batched=/tmp/gemv_coop_f32_batched.co'
emit_flag '--emit-amdgpu-gemv-coop-bf16-f32-batched=/tmp/gemv_coop_bf16_f32_batched.co'
emit_flag '--emit-amdgpu-gemv-coop-q4-0-f32-batched=/tmp/gemv_coop_q4_0_f32_batched.co'
emit_flag '--emit-amdgpu-rope-qwen3=/tmp/rope_qwen3_f32.co'
emit_flag '--emit-amdgpu-qkv-split-f32=/tmp/qkv_split_f32.co'
emit_flag '--emit-amdgpu-qkv-split-f32-batched=/tmp/qkv_split_f32_batched.co'
emit_flag '--emit-amdgpu-qkv-split-f32-14b=/tmp/qkv_split_f32_14b.co'
emit_flag '--emit-amdgpu-qkv-split-f32-batched-14b=/tmp/qkv_split_f32_batched_14b.co'
emit_flag '--emit-amdgpu-qkv-split-f32-speck4=/tmp/qkv_split_f32_speck4.co'
emit_flag '--emit-amdgpu-attn-decode-f32=/tmp/attn_decode_f32.co'
emit_flag '--emit-amdgpu-attn-decode-f32-14b=/tmp/attn_decode_f32_14b.co'
emit_flag '--emit-amdgpu-attn-decode-f32-speck4=/tmp/attn_decode_f32_speck4.co'
emit_flag '--emit-amdgpu-extract-q-qwen3-f32=/tmp/extract_q_qwen3_f32.co'
emit_flag '--emit-amdgpu-extract-k-qwen3-f32=/tmp/extract_k_qwen3_f32.co'
emit_flag '--emit-amdgpu-extract-k-qwen3-f32-speck4=/tmp/extract_k_qwen3_f32_speck4.co'
emit_flag '--emit-amdgpu-insert-k-qwen3-f32=/tmp/insert_k_qwen3_f32.co'
emit_flag '--emit-amdgpu-insert-k-qwen3-f32-speck4=/tmp/insert_k_qwen3_f32_speck4.co'
emit_flag '--emit-amdgpu-head-extract-f32=/tmp/head_extract_f32.co'
emit_flag '--emit-amdgpu-head-insert-f32=/tmp/head_insert_f32.co'
emit_flag '--emit-amdgpu-transpose-f32=/tmp/transpose_f32.co'
emit_flag '--emit-amdgpu-kv-broadcast-f32=/tmp/kv_broadcast.co'
emit_flag '--emit-amdgpu-embedding-lookup-f32=/tmp/embedding_lookup_f32.co'
emit_flag '--emit-amdgpu-argmax-logits-f32=/tmp/argmax_logits_f32.co'
emit_flag '--emit-amdgpu-qknorm-f32=/tmp/qknorm_f32.co'
emit_flag '--emit-amdgpu-silu-mul-f32-batched=/tmp/silu_mul_f32_batched.co'
emit_flag '--emit-amdgpu-rmsnorm-f32-N=1024:/tmp/rmsnorm_f32_1024.co'
emit_flag '--emit-amdgpu-rmsnorm-f32-N=5120:/tmp/rmsnorm_f32_5120.co'

echo "=== path 2: hipcc compile of legacy @kernel sources ==="
emit_hip_amd 'examples/llm/bf16_to_f32.mlr'      /tmp/bf16_to_f32
emit_hip_amd 'examples/llm/gemv_f32.mlr'         /tmp/gemv_f32
emit_hip_amd 'examples/llm/gemm_f32.mlr'         /tmp/gemm_f32
emit_hip_amd 'examples/llm/residual_add_f32.mlr' /tmp/residual_add_f32

echo "=== path 3: AST-walker recogniser for single-stream silu_mul_f32 ==="
emit_amdgpu_native 'examples/llm/silu_mul_f32.mlr' /tmp/silu_mul_f32

echo "=== mega-kernel .cos ==="
"$MLRC" --emit-amdgpu-qwen3-megakernel-v2=/tmp/qwen3_layer_megakernel_v2.co examples/llm/qwen3_layer_megakernel.mlr > /dev/null 2>&1 \
    && echo "  ok  qwen3_layer_megakernel_v2.co"
"$MLRC" --emit-amdgpu-qwen3-megakernel-speck4-v2=/tmp/qwen3_layer_megakernel_speck4_v2.co examples/llm/qwen3_layer_megakernel_speck4.mlr > /dev/null 2>&1 \
    && echo "  ok  qwen3_layer_megakernel_speck4_v2.co"
"$MLRC" --emit-amdgpu-llama-megakernel-v2=/tmp/llama_layer_megakernel_v2.co examples/llm/llama_layer_megakernel.mlr > /dev/null 2>&1 \
    && echo "  ok  llama_layer_megakernel_v2.co"
"$MLRC" --emit-amdgpu-llama-megakernel-speck4-v2=/tmp/llama_layer_megakernel_speck4_v2.co examples/llm/llama_layer_megakernel_speck4.mlr > /dev/null 2>&1 \
    && echo "  ok  llama_layer_megakernel_speck4_v2.co"

# mks8 / mks16 mega-kernels — hipcc-only (no v2 AST-walker port yet).
# Required by the qwen3-0.6B PLD speculative-decode path that reaches
# 200+ tok/s.  Without these the driver falls back to per-op M_eff=16
# chain at ~20 tok/s.
echo "=== path 4: hipcc compile of mks8 / mks16 mega-kernels ==="
if command -v hipcc > /dev/null 2>&1; then
    hipcc --offload-arch=gfx1100 --genco -O3 examples/llm/qwen3_layer_megakernel_speck8.hip.cpp -o /tmp/qwen3_layer_megakernel_speck8.co > /dev/null 2>&1 \
        && echo "  ok  qwen3_layer_megakernel_speck8.co" \
        || echo "  FAIL qwen3_layer_megakernel_speck8.co"
    hipcc --offload-arch=gfx1100 --genco -O3 examples/llm/qwen3_layer_megakernel_speck16.hip.cpp -o /tmp/qwen3_layer_megakernel_speck16.co > /dev/null 2>&1 \
        && echo "  ok  qwen3_layer_megakernel_speck16.co" \
        || echo "  FAIL qwen3_layer_megakernel_speck16.co"
else
    echo "  skip mks8/mks16 — hipcc not on PATH; spec_K=8/16 falls back to per-op chain"
fi

echo
n=$(ls /tmp/*.co 2>/dev/null | wc -l)
echo "Done. /tmp/*.co count = $n"
