#!/bin/bash
# Copies the GPU's IP-discovery binary out of debugfs for offline parsing.
# Run as root once per kernel version (the discovery binary is per-hardware
# and stable across boots).
set -euo pipefail

BDF="${1:-0000:03:00.0}"
SRC="/sys/kernel/debug/dri/${BDF}/amdgpu_discovery"
DST="tools/discovery.bin"

if [ "$(id -u)" -ne 0 ]; then
    echo "Need root to read $SRC. Re-run with sudo." >&2
    exit 1
fi
if [ ! -r "$SRC" ]; then
    echo "Cannot read $SRC — wrong BDF? amdgpu not loaded?" >&2
    exit 1
fi
mkdir -p tools
cp "$SRC" "$DST"
chmod 644 "$DST"
echo "Wrote $DST ($(stat -c%s "$DST") bytes)"
