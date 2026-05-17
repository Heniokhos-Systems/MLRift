#!/bin/bash
# Run Qwen3 decode while sampling GPU register state.
# Output: /tmp/probe_during/snapshot_<seq>.txt + /tmp/probe_during/qwen3.log
set -uo pipefail

if [ "$(id -u)" -ne 0 ]; then
    echo "Re-run as root (BAR5 mmap requires root)." >&2
    exec sudo "$0" "$@"
fi

OUT=/tmp/probe_during
mkdir -p "$OUT"
rm -f "$OUT"/snapshot_*.txt "$OUT"/qwen3.log

# Verify prerequisites
[ -x /tmp/qwen3_gen ] || { echo "Build /tmp/qwen3_gen first (./build/mlrc --target=amdgpu-native --emit=elfexe examples/qwen3_generate.mlr -o /tmp/qwen3_gen)" >&2; exit 1; }
[ -x /tmp/pci_register_probe_v2 ] || { echo "Build /tmp/pci_register_probe_v2 first (Task 5)" >&2; exit 1; }

# Start a 200-token Qwen3 to keep the GPU busy for ~2 s
MLRIFT_QWEN3_0_6B_DIR=/home/pantelis/Desktop/Projects/Work/MLRift-experimental/Qwen3-0.6B/model.safetensors \
MLRIFT_NATIVE_MEGAKERNEL=2 \
MLRIFT_QWEN3_MAX_NEW=200 \
/tmp/qwen3_gen > "$OUT/qwen3.log" 2>&1 &
QPID=$!

# Sample registers every 100 ms while qwen3 runs (cap at 50 samples = 5 s safety)
seq=0
while kill -0 $QPID 2>/dev/null && [ "$seq" -lt 50 ]; do
    /tmp/pci_register_probe_v2 > "$OUT/snapshot_$(printf '%03d' $seq).txt" 2>&1
    seq=$((seq + 1))
    sleep 0.1
done

wait $QPID 2>/dev/null
QWEN_EXIT=$?
echo "qwen3 exit=$QWEN_EXIT; captured $seq snapshots"
grep -E "tokens_per_sec_x1000|KFD-WEDGE" "$OUT/qwen3.log" | head -3
