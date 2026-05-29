#!/bin/bash
# Batch sweep: vary B (BS=B*16), rebuild ONLY the launcher (kernels are runtime-parametric),
# run with synthetic ids, capture ms/step + VRAM + RSS. Kernels are NOT re-emitted.
# NOTE: each B>8 is a NEW shape — run under operator dmesg supervision (kfd-safe-run).
set -e
REPO=/home/pantelis/Desktop/Projects/Work/MLRift
SRC=$REPO/examples/wzma_train_md.mlr
BIN=~/mlrift_bin/wzma_train/wzma_train_md
B=${1:?usage: wzma_batch_sweep.sh <B>}
BS=$((B*16))
# match the EXACT current declaration text (verify with grep before sed):
sed -i "s/^static u64 B_DIM     = .*/static u64 B_DIM     = $B/"  $SRC
sed -i "s/^static u64 BS_DIM    = .*/static u64 BS_DIM    = $BS/" $SRC
$REPO/build/mlrc --target=amdgpu-native $SRC -o $BIN
echo "=== B=$B BS=$BS ==="
/usr/bin/time -v env MLRIFT_NSTEPS=300 MLRIFT_SYNTH_IDS=1 $BIN > /tmp/sweep_B$B.out 2> /tmp/sweep_B$B.time || true
grep -E "ms / step|TOTAL" /tmp/sweep_B$B.out || true
grep -iE "VRAM|GTT" /tmp/sweep_B$B.time || true
grep "Maximum resident" /tmp/sweep_B$B.time || true
