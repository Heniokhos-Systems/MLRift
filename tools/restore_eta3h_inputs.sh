#!/usr/bin/env bash
# Restore the η.3h replay inputs to /tmp after a reboot wipes them.
#
# Usage:
#   ./tools/restore_eta3h_inputs.sh
#
# What it does:
#   1. Decompresses /lib/firmware/amdgpu/psp_13_0_10_sos.bin.zst →
#      /tmp/mlrift_fw/psp_13_0_10_sos.bin (used by the BL chain).
#   2. Copies captures/amdgpu_replay.bin → /tmp/amdgpu_replay.bin
#      (consumed by phase3_eta3h_trace_replay).
#   3. Verifies sizes look right.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

mkdir -p /tmp/mlrift_fw
zstd -d -k -f -q -o /tmp/mlrift_fw/psp_13_0_10_sos.bin \
    /lib/firmware/amdgpu/psp_13_0_10_sos.bin.zst

cp "$ROOT/captures/amdgpu_replay.bin" /tmp/amdgpu_replay.bin

ls -l /tmp/mlrift_fw/psp_13_0_10_sos.bin /tmp/amdgpu_replay.bin
echo "η.3h inputs restored. Next:"
echo "  sudo ./build/phase3_eta3h_trace_replay"
