#!/usr/bin/env bash
# η.3g — Capture amdgpu's MMIO sequence as it initializes the dGPU.
#
# Strategy: while the kernel mmiotrace tracer is active, unbind dGPU
# from vfio-pci and bind it to amdgpu. amdgpu does its complete PSP /
# SOS / SMU / GFX init under our trace. Resulting trace shows every
# MMIO read/write byte-for-byte — the working reference sequence.
#
# Compare against our η.3e-7 LOAD_SOSDRV sequence to find the missing
# writes that wake SOS.
#
# DANGER:
#   - Amdgpu binding to a previously-vfio'd dGPU MAY wedge if device
#     state is dirty. Cold cycle recovers.
#   - If amdgpu binds successfully, you will have amdgpu attached to
#     the dGPU until you reboot. This is fine; vfio-pci will rebind on
#     next boot via softdep + modprobe.d.
#
# Output: /tmp/amdgpu_mmio_trace.txt (gigabytes possible — be ready).

set -euo pipefail

if [[ $EUID -ne 0 ]]; then
    echo "Must run as root." >&2; exit 1
fi

DEVICE="0000:03:00.0"
TRACEFS="/sys/kernel/debug/tracing"
OUT="/tmp/amdgpu_mmio_trace.txt"
WAIT_SECONDS="${WAIT_SECONDS:-45}"

if [[ ! -d "$TRACEFS" ]]; then
    echo "tracefs not mounted at $TRACEFS — mount with: mount -t tracefs nodev /sys/kernel/tracing" >&2
    exit 2
fi

# Verify mmiotrace is available.
if ! grep -qw mmiotrace "$TRACEFS/available_tracers" 2>/dev/null; then
    echo "mmiotrace not listed in $TRACEFS/available_tracers:" >&2
    cat "$TRACEFS/available_tracers" 2>/dev/null || true
    exit 3
fi

echo "=== mmio_capture_amdgpu ==="
echo "Device:          $DEVICE"
echo "Output file:     $OUT"
echo "Wait after bind: ${WAIT_SECONDS}s"
echo ""

# Sanity: dGPU must currently be bound to vfio-pci.
CUR_DRV="$(readlink -f /sys/bus/pci/devices/$DEVICE/driver 2>/dev/null || true)"
case "$CUR_DRV" in
    */vfio-pci) echo "Pre-state OK — dGPU is on vfio-pci." ;;
    *)
        echo "Pre-state WRONG: dGPU is not on vfio-pci (current: $CUR_DRV)" >&2
        echo "Refusing to proceed — would not capture amdgpu init from cold." >&2
        exit 4
        ;;
esac
echo ""

# Wipe any previous trace.
echo "1) Reset trace buffer + select mmiotrace tracer"
echo nop > "$TRACEFS/current_tracer"
echo > "$TRACEFS/trace"
echo mmiotrace > "$TRACEFS/current_tracer"
echo 1 > "$TRACEFS/tracing_on"

# Optionally enlarge the per-cpu trace buffer (the default 1408 KB
# fills up FAST during amdgpu PSP load — kernel will silently drop
# events if it overflows). 64MB per CPU here.
echo 65536 > "$TRACEFS/buffer_size_kb" 2>/dev/null || true
echo "   buffer_size_kb = $(cat "$TRACEFS/buffer_size_kb")"
echo ""

# Unbind from vfio-pci.
echo "2) Unbind $DEVICE from vfio-pci"
echo "$DEVICE" > /sys/bus/pci/drivers/vfio-pci/unbind
sleep 1

# Bind to amdgpu. This triggers the full driver init sequence we want
# to capture. amdgpu is already loaded (it owns the iGPU).
echo "3) Bind $DEVICE to amdgpu (this triggers PSP/SOS/MEC init)"
echo "$DEVICE" > /sys/bus/pci/drivers/amdgpu/bind &
BIND_PID=$!

# Wait while amdgpu does its init work.
echo "4) Waiting ${WAIT_SECONDS}s for amdgpu init to complete..."
sleep "$WAIT_SECONDS"

# Stop tracing.
echo "5) Stop tracing + dump"
echo 0 > "$TRACEFS/tracing_on"

# Wait for the bind command to settle (don't kill it).
wait "$BIND_PID" 2>/dev/null || true

# Check final binding state.
echo ""
echo "=== Post-capture state ==="
echo "Final driver:    $(readlink -f /sys/bus/pci/devices/$DEVICE/driver 2>/dev/null || echo 'unbound')"
echo "Trace dump:      $OUT"
echo ""

# Dump trace.
cp "$TRACEFS/trace" "$OUT"
chmod 644 "$OUT"
echo "Trace size:      $(stat -c%s "$OUT") bytes ($(wc -l < "$OUT") lines)"
echo ""

# Tail summary.
echo "=== Last 30 trace lines (sanity check) ==="
tail -30 "$OUT"
echo ""

# Stats on amdgpu activity.
echo "=== Stats ==="
echo "MAP events:    $(grep -c '^MAP ' "$OUT" 2>/dev/null || true)"
echo "R   events:    $(grep -c '^R '   "$OUT" 2>/dev/null || true)"
echo "W   events:    $(grep -c '^W '   "$OUT" 2>/dev/null || true)"
echo "UNMAP events:  $(grep -c '^UNMAP ' "$OUT" 2>/dev/null || true)"

# Reset tracer so future operations don't pay the overhead.
echo nop > "$TRACEFS/current_tracer"
echo "Done. mmiotrace disabled."
echo ""
echo "Trace at:        $OUT"
echo ""
echo "Note: dGPU remains bound to amdgpu until you reboot."
echo "After reboot: vfio-pci will rebind via modprobe.d softdep."
