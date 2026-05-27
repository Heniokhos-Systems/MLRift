#!/usr/bin/env bash
# Task 4 (tiled dU/dx) validation — RUN ONLY AFTER A GPU COLD POWER CYCLE.
#   The first buggy run (kernarg 504-byte segment OOB) wedged MES; the fix
#   (kernarg segment 504->512) is in the tree but UNVALIDATED on GPU.
#
# Sequence:
#   1) NW=1: rebuild mlrc+kernel+launcher, run -> dumps the T4 NW=1 oracle
#      (dU sum-order changed in T4, so the oracle MUST be regenerated).
#   2) NW=8: rebuild, run x3 -> validate loss<=1e-2 + grads<=1e-3 vs oracle,
#      capture segment-B ms/step, confirm no wedge across 3 runs.
#
# If ANY run prints [KFD-WEDGE]: STOP. Do NOT reboot/rmmod. Cold-cycle only.
set -e
cd "$(dirname "$0")/.."
BIN=$HOME/mlrift_bin/wzma_train
MEGA=src/format_amdgpu_megakernel.mlr
LAUN=examples/wzma_train_mega.mlr
KSRC=examples/llm/wzma_train_mega_kernel.mlr

set_nw () {  # $1 = NW value
  sed -i "s/^static uint64 MEGA_TRAIN_NW = .*/static uint64 MEGA_TRAIN_NW = $1/" "$MEGA"
  sed -i "s/^static u64 WZMA_MEGA_NW = .*/static u64 WZMA_MEGA_NW = $1/" "$LAUN"
}
build_all () {
  make build >/dev/null 2>&1
  rm -f "$BIN/wzma_train_mega.co.co"
  ./build/mlrc --target=amdgpu-native "$KSRC" -o "$BIN/wzma_train_mega" 2>&1 | grep -E "error" || true
  ./build/mlrc --target=amdgpu-native "$LAUN" -o "$BIN/wzma_train_mega_launcher" 2>&1 | grep -E "^[1-9][0-9]* error" || true
  echo "  kernarg: $(/usr/bin/llvm-readobj --notes "$BIN/wzma_train_mega.co" 2>/dev/null | grep kernarg_segment_size | tr -d ' ')"
}

echo "===== STEP 1: NW=1 — regenerate T4 oracle ====="
set_nw 1; build_all
"$BIN/wzma_train_mega_launcher" 2>&1 | grep -E "WEDGE|ORACLE|loss\[0\]|loss\[49\]|TASK 7|embed PASS|max loss"

echo "===== STEP 2: NW=8 — validate (3 runs) ====="
set_nw 8; build_all
for r in 1 2 3; do
  echo "----- NW=8 run $r -----"
  "$BIN/wzma_train_mega_launcher" 2>&1 | grep -E "WEDGE|REDUCE vs oracle|loss\[0\]|loss\[49\]|TASK 7|max loss|segment B|megakernel:"
done
echo "===== DONE (compare segment B vs 6.6 ms prior) ====="
