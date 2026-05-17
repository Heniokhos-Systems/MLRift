#!/bin/bash
# Run Llama-3.2-1B GPU mega-kernel decode while sampling MEC compute-queue state
# via SRBM-selected amdgpu_regs2 probe. Produces a machine-checkable artifact
# (snapshots + md5 unique count + first-vs-last diff) so future sessions can
# investigate why amdgpu_regs2+SRBM does not surface live HQD state on this
# kernel/firmware combo (see canonical_snapshot.txt for the all-zero baseline).
#
# Workload note: Llama-1B is the canonical "GPU is firing" decode (~100 tok/s
# mega-kernel, ~2.5 GiB VRAM resident). Qwen3 default decode currently regresses
# to CPU fallback so it would not be a valid GPU-load workload.
#
# Output dir: /tmp/probe_during/
#   - llama1b.log                Llama driver stdout/stderr
#   - snapshot_NNN.txt           one capture from /tmp/pci_register_probe_v3 per iter
#   - diff_summary.txt           first-vs-last diff (only if state changed)
#   - canonical_snapshot.txt     single canonical content (only if all identical)
set -uo pipefail

if [ "$(id -u)" -ne 0 ]; then
    echo "Re-run as root (BAR5 mmap requires root)." >&2
    exec sudo "$0" "$@"
fi

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT=/tmp/probe_during
mkdir -p "$OUT"
rm -f "$OUT"/snapshot_*.txt "$OUT"/llama1b.log "$OUT"/diff_summary.txt "$OUT"/canonical_snapshot.txt

# Verify prerequisites
if [ ! -x /tmp/llama1b_gen ]; then
    echo "Build /tmp/llama1b_gen first:" >&2
    echo "  ./build/mlrc --target=amdgpu-native --emit=elfexe examples/llama3_1b_gpu_generate.mlr -o /tmp/llama1b_gen" >&2
    echo "  (the --target=amdgpu-native flag is critical: without it the KFD-shim path is" >&2
    echo "   not linked and the driver silently falls back to CPU bf16.)" >&2
    exit 1
fi
if [ ! -x /tmp/pci_register_probe_v3 ]; then
    echo "Build /tmp/pci_register_probe_v3 first:" >&2
    echo "  ./build/mlrc --emit=elfexe examples/pci_register_probe_v3.mlr -o /tmp/pci_register_probe_v3" >&2
    exit 1
fi

# Resolve GGUF path (default to repo-local models/ if not overridden)
GGUF_PATH="${MLRIFT_LLAMA3_1B_GGUF:-$REPO_ROOT/models/llama-3.2-1b-instruct-q8_0.gguf}"
if [ ! -f "$GGUF_PATH" ]; then
    echo "GGUF not found at $GGUF_PATH; set MLRIFT_LLAMA3_1B_GGUF or place file at default." >&2
    exit 1
fi

# Launch Llama-1B GPU mega-kernel decode (20 tokens ~ 200 ms at 100 tok/s).
# MLRIFT_NATIVE_MEGAKERNEL=2 = required for native mega-kernel path.
# MLRIFT_QWEN3_MAX_NEW is the shared decode-cap env var (capped at 20 inside driver).
MLRIFT_LLAMA3_1B_GGUF="$GGUF_PATH" \
MLRIFT_NATIVE_MEGAKERNEL=2 \
MLRIFT_QWEN3_MAX_NEW=20 \
/tmp/llama1b_gen > "$OUT/llama1b.log" 2>&1 &
LPID=$!

# Sample registers in a tight loop while Llama is alive.
# Cadence ~50 ms; cap at 200 samples (10 s safety) so we never run forever.
seq=0
while kill -0 $LPID 2>/dev/null && [ "$seq" -lt 200 ]; do
    /tmp/pci_register_probe_v3 > "$OUT/snapshot_$(printf '%03d' $seq).txt" 2>&1
    seq=$((seq + 1))
    sleep 0.05
done

wait $LPID 2>/dev/null
LEXIT=$?

echo "===== run summary ====="
echo "llama1b_gen exit=$LEXIT"
echo "snapshots captured: $seq"

# Surface Llama tok/s to prove GPU was actually firing during the probe window.
TOK_LINE="$(grep -E 'tokens_per_sec_x1000' "$OUT/llama1b.log" | head -1 || true)"
VRAM_LINE="$(grep -E 'KFD alloc summary' "$OUT/llama1b.log" | head -1 || true)"
echo "llama: $TOK_LINE"
echo "llama: $VRAM_LINE"

# Diff analysis: how many distinct snapshot contents did we see?
if [ "$seq" -lt 2 ]; then
    echo "✗ fewer than 2 snapshots captured ($seq) — decode finished too fast to sample" >&2
    exit 2
fi

UNIQUE=$(md5sum "$OUT"/snapshot_*.txt | awk '{print $1}' | sort -u | wc -l)
echo "unique snapshot contents (by md5): $UNIQUE / $seq"

FIRST="$OUT/snapshot_000.txt"
LAST="$OUT/snapshot_$(printf '%03d' $((seq - 1))).txt"

if [ "$UNIQUE" -ge 2 ]; then
    echo "✓ register state changed during decode — SRBM probe is observing live MEC state"
    {
        echo "===== first-vs-last diff ($FIRST  vs  $LAST) ====="
        diff -u "$FIRST" "$LAST" || true
    } > "$OUT/diff_summary.txt"
    echo "diff written to $OUT/diff_summary.txt"
else
    echo "✗ all $seq snapshots byte-identical — SRBM+amdgpu_regs2 path does not surface live MEC state on this kernel/firmware (see live_hqd_count in any snapshot)"
    cp "$FIRST" "$OUT/canonical_snapshot.txt"
    LIVE_HQD="$(grep -E 'live_hqd|HQD' "$OUT/canonical_snapshot.txt" | head -3 || true)"
    if [ -n "$LIVE_HQD" ]; then
        echo "canonical snapshot HQD lines:"
        echo "$LIVE_HQD" | sed 's/^/  /'
    fi
    echo "canonical snapshot written to $OUT/canonical_snapshot.txt"
fi
