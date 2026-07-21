#!/bin/bash
# rebuild_wzma_cos.sh — rebuild EXACTLY the 22 `/tmp/*.co` kernels that the
# WZMA training launcher (examples/wzma_train_md.mlr) loads at startup.
#
# CPU-only compilation. This script NEVER runs a GPU launcher, never touches
# the GPU, and never invokes hipcc/ROCm/clang. Everything comes out of
# MLRift's own AST-walker / native AMDGPU backend (./build/mlrc).
#
# Two emit mechanisms are used:
#
#   1. Single-flag AST-walker emit (`--arch=x86_64 --emit-amdgpu-<k>=<path>`):
#        Produces a `.co` directly from a dedicated emitter in src/main.mlr.
#        NOTE: the emitted filename must match the launcher's expected name,
#        e.g. the `gemm-f32-grouped-v2` flag writes /tmp/gemm_f32_grouped.co
#        (NOT ..._v2.co).
#
#   2. Native source-compile (`--target=amdgpu-native <src>.mlr -o <stub>`):
#        Writes <stub>.co by name-routing the @kernel in the source file.
#        gemm_f32_native.co comes from examples/llm/gemm_f32.mlr (symbol
#        `gemm_f32`); its plain `--emit-amdgpu-gemm-f32=` flag was deleted
#        (Phase 3c-3 — the AST path owns it), so it MUST go through path 2.
#
# Usage:
#   scripts/rebuild_wzma_cos.sh [path/to/mlrc]
# Defaults to ./build/mlrc if no path given.
set -e
MLRC="${1:-${MLRC:-./build/mlrc}}"
[ -x "$MLRC" ] || { echo "rebuild_wzma_cos: mlrc not executable: $MLRC" >&2; exit 1; }

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

mkdir -p /tmp
echo 'fn main() {}' > /tmp/empty.mlr

fail=0

emit_flag() {
    # $1 = '--emit-amdgpu-<k>=/tmp/foo.co'  (single-flag AST-walker emit)
    local fp="$1"
    if "$MLRC" --arch=x86_64 "$fp" /tmp/empty.mlr > /dev/null 2>&1; then
        echo "  ok   $fp"
    else
        echo "  FAIL $fp" >&2
        fail=1
    fi
}

emit_native() {
    # $1 = examples/llm/X.mlr   $2 = /tmp/stub  → /tmp/stub.co (AST-walker)
    local src="$1"; local stub="$2"
    if "$MLRC" --target=amdgpu-native "$src" -o "$stub" > /dev/null 2>&1; then
        echo "  ok   $src -> ${stub}.co"
    else
        echo "  FAIL $src" >&2
        fail=1
    fi
}

echo "=== path 1: single-flag AST-walker emits (8) ==="
emit_flag '--emit-amdgpu-adamw-fused-v2=/tmp/adamw_fused.co'
emit_flag '--emit-amdgpu-embedding-lookup-f32=/tmp/embedding_lookup_f32.co'
emit_flag '--emit-amdgpu-fill-step=/tmp/fill_step.co'
emit_flag '--emit-amdgpu-gemm-bf16-grouped-v2=/tmp/gemm_bf16_grouped.co'
emit_flag '--emit-amdgpu-gemm-f32-grouped-v2=/tmp/gemm_f32_grouped.co'
emit_flag '--emit-amdgpu-gemm-f32-grouped-rb2=/tmp/gemm_f32_grouped_rb2.co'
emit_flag '--emit-amdgpu-pack-f32-to-bf16=/tmp/pack_f32_to_bf16.co'
emit_flag '--emit-amdgpu-transpose-f32=/tmp/transpose_f32.co'
emit_flag '--emit-amdgpu-wzma-fwd-mega-v2=/tmp/wzma_fwd_mega.co'

echo "=== path 2: native source-compile of @kernel sources (14) ==="
emit_native examples/llm/adamw_step_f32.mlr          /tmp/adamw_step_f32
emit_native examples/llm/gate_bwd_bs_f32.mlr         /tmp/gate_bwd_bs_f32
emit_native examples/llm/gate_bwd_dgrad_f32.mlr      /tmp/gate_bwd_dgrad_f32
emit_native examples/llm/gemm_f32_accum.mlr          /tmp/gemm_f32_accum
emit_native examples/llm/gemm_f32.mlr                /tmp/gemm_f32_native
emit_native examples/llm/repack_abw_to_baw_f32.mlr   /tmp/repack_abw_to_baw_f32
emit_native examples/llm/row_scale_f32.mlr           /tmp/row_scale_f32
emit_native examples/llm/wzma_bwd_embed_kernel.mlr   /tmp/wzma_bwd_embed
emit_native examples/llm/wzma_bwd_pre_grad_kernel.mlr   /tmp/wzma_bwd_pre_grad
emit_native examples/llm/wzma_bwd_pre_logits_kernel.mlr /tmp/wzma_bwd_pre_logits
emit_native examples/llm/wzma_bwd_reduce_kernel.mlr  /tmp/wzma_bwd_reduce
emit_native examples/llm/wzma_bwd_uv_kernel.mlr      /tmp/wzma_bwd_uv
emit_native examples/llm/wzma_fwd_tail.mlr           /tmp/wzma_fwd_tail

echo
echo "=== verify all 22 exist and are non-empty ==="
WZMA_COS="gemm_f32_grouped_rb2 adamw_fused adamw_step_f32 embedding_lookup_f32 fill_step \
gate_bwd_bs_f32 gate_bwd_dgrad_f32 gemm_bf16_grouped gemm_f32_accum \
gemm_f32_grouped gemm_f32_native pack_f32_to_bf16 repack_abw_to_baw_f32 \
row_scale_f32 transpose_f32 wzma_bwd_embed wzma_bwd_pre_grad \
wzma_bwd_pre_logits wzma_bwd_reduce wzma_bwd_uv wzma_fwd_mega wzma_fwd_tail"
n=0
for k in $WZMA_COS; do
    if [ -s "/tmp/${k}.co" ]; then
        n=$((n + 1))
    else
        echo "  MISSING/EMPTY /tmp/${k}.co" >&2
        fail=1
    fi
done

echo
echo "Done. WZMA .co present = ${n}/22"
[ "$fail" -eq 0 ] || { echo "rebuild_wzma_cos: one or more kernels FAILED" >&2; exit 1; }
