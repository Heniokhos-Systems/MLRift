#!/usr/bin/env bash
# Slice 4.18 — extract gfx1100 .text bytes for the native mega-kernel emit.
#
# This is a one-shot offline step: it builds the HIP source via hipcc
# once, then unbundles + dumps the .text section that MLRift's own
# .co packager stamps into a fresh ELF.  After this runs, MLRift's
# --emit-amdgpu-qwen3-megakernel=PATH flag produces a working .co
# without invoking hipcc.
#
# Usage:
#   ./scripts/build_native_megakernel.sh
#
# Writes:
#   build/native_megakernel/qwen3_layer_megakernel.gfx1100.bin

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

ROCM_LLVM=/opt/rocm-7.2.0/llvm/bin
HIP_SRC=examples/llm/qwen3_layer_megakernel.hip.cpp
WORK=build/native_megakernel
mkdir -p "$WORK"

CO="$WORK/qwen3_layer_megakernel.hipcc.co"
ELF="$WORK/qwen3_layer_megakernel.gfx1100.elf"
BIN="$WORK/qwen3_layer_megakernel.gfx1100.bin"

echo "[1/3] hipcc -> $CO"
hipcc --offload-arch=gfx1100 --genco -O3 "$HIP_SRC" -o "$CO"

echo "[2/3] clang-offload-bundler --unbundle -> $ELF"
"$ROCM_LLVM/clang-offload-bundler" \
    --type=o --unbundle \
    --input="$CO" \
    --targets=hipv4-amdgcn-amd-amdhsa--gfx1100 \
    --output="$ELF"

echo "[3/3] llvm-objcopy --dump-section .text -> $BIN"
"$ROCM_LLVM/llvm-objcopy" --dump-section=.text="$BIN" "$ELF"

echo "ok: $BIN ($(stat -c %s "$BIN") bytes)"
