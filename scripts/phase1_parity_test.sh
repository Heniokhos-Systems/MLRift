#!/bin/bash
# Phase 1 AQL-dispatch token-parity smoke for Llama-3.2-1B.
#
# Builds the GPU mega-kernel driver, then runs it twice — once on the
# canonical hipModuleLaunchKernel path and once with MLRIFT_PHASE1_DISPATCH=1
# routing through the AQL builders shipped in slices 5b-compute phase1 1..5.
# Tokens must match byte-for-byte; tok/s must be within noise.
#
# Usage:
#   ./scripts/phase1_parity_test.sh            # 8-token smoke
#   N_TOKENS=20 ./scripts/phase1_parity_test.sh  # longer run
set -uo pipefail

N_TOKENS="${N_TOKENS:-8}"
MODEL="${MLRIFT_LLAMA3_1B_GGUF:-/home/pantelis/Desktop/Projects/Work/MLRift/models/llama-3.2-1b-instruct-q8_0.gguf}"
DRIVER_SRC="examples/llama3_1b_gpu_generate.mlr"
DRIVER_BIN="/tmp/llama1b_p1"
OFF_LOG="/tmp/llama1b_off.log"
ON_LOG="/tmp/llama1b_on.log"

if [[ ! -f "$MODEL" && ! -L "$MODEL" ]]; then
  echo "ERROR: model not found at $MODEL" >&2
  exit 1
fi

echo "[phase1-parity] rebuilding helper .co files..."
./scripts/rebuild_helper_cos.sh >/dev/null 2>&1 || {
  echo "ERROR: rebuild_helper_cos.sh failed" >&2
  exit 1
}

echo "[phase1-parity] building $DRIVER_SRC ..."
./build/mlrc --target=amdgpu-native --emit=elfexe "$DRIVER_SRC" -o "$DRIVER_BIN" 2>&1 | tail -3
if [[ ! -x "$DRIVER_BIN" ]]; then
  echo "ERROR: driver build failed" >&2
  exit 1
fi

echo "[phase1-parity] running canonical path (MLRIFT_PHASE1_DISPATCH unset, N=$N_TOKENS) ..."
MLRIFT_NATIVE_MEGAKERNEL=2 \
MLRIFT_LLAMA3_1B_GGUF="$MODEL" \
MLRIFT_QWEN3_MAX_NEW="$N_TOKENS" \
  timeout 60 "$DRIVER_BIN" > "$OFF_LOG" 2>&1
OFF_RC=$?

echo "[phase1-parity] running phase1 path (MLRIFT_PHASE1_DISPATCH=1, N=$N_TOKENS) ..."
MLRIFT_NATIVE_MEGAKERNEL=2 \
MLRIFT_PHASE1_DISPATCH=1 \
MLRIFT_LLAMA3_1B_GGUF="$MODEL" \
MLRIFT_QWEN3_MAX_NEW="$N_TOKENS" \
  timeout 60 "$DRIVER_BIN" > "$ON_LOG" 2>&1
ON_RC=$?

echo "[phase1-parity] off rc=$OFF_RC  on rc=$ON_RC"

# Strip noisy timing / addr fields; keep deterministic content only.
# step_ms varies run-to-run; logit/next_id should not.
grep -E "next_id|GENERATED IDs|^[0-9]+, [0-9]+" "$OFF_LOG" \
  | sed 's/ step_ms=[0-9]*//' > /tmp/llama1b_off_tokens.txt
grep -E "next_id|GENERATED IDs|^[0-9]+, [0-9]+" "$ON_LOG" \
  | sed 's/ step_ms=[0-9]*//' > /tmp/llama1b_on_tokens.txt

echo "[phase1-parity] === token diff (empty = parity) ==="
if diff /tmp/llama1b_off_tokens.txt /tmp/llama1b_on_tokens.txt; then
  echo "[phase1-parity] PARITY OK ($(wc -l < /tmp/llama1b_off_tokens.txt) lines each)"
  PARITY=0
else
  echo "[phase1-parity] PARITY FAIL"
  PARITY=1
fi

echo "[phase1-parity] === tok/s ==="
grep tokens_per_sec_x1000 "$OFF_LOG" | sed 's/^/  off: /'
grep tokens_per_sec_x1000 "$ON_LOG"  | sed 's/^/  on:  /'

echo "[phase1-parity] === KFD-WEDGE check ==="
WEDGE_OFF=$(grep -c "KFD-WEDGE" "$OFF_LOG")
WEDGE_ON=$(grep -c "KFD-WEDGE" "$ON_LOG")
echo "  off: $WEDGE_OFF  on: $WEDGE_ON"
if [[ "$WEDGE_ON" -gt "$WEDGE_OFF" ]]; then
  echo "[phase1-parity] WEDGE REGRESSION — phase1 wedged but canonical did not" >&2
  exit 2
fi

exit $PARITY
