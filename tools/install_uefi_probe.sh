#!/usr/bin/env bash
# Install (or remove) the η.3f UEFI POST auto-probe systemd service.
#
# Usage:
#   sudo ./tools/install_uefi_probe.sh install   # add + enable
#   sudo ./tools/install_uefi_probe.sh remove    # disable + delete
#   sudo ./tools/install_uefi_probe.sh status    # check
#
# What it does (install):
#   1. Copies mlrift-uefi-probe.service to /etc/systemd/system/
#   2. systemctl daemon-reload
#   3. systemctl enable mlrift-uefi-probe.service
#   4. Touches /var/log/mlrift_uefi_probe.log so it exists with sane perms
#
# After install: next boot will run phase3_eta3f_post_uefi_state and
# append output to /var/log/mlrift_uefi_probe.log.

set -euo pipefail

if [[ $EUID -ne 0 ]]; then
    echo "Must run as root (or via sudo)." >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVICE_SRC="$SCRIPT_DIR/mlrift-uefi-probe.service"
SERVICE_DST="/etc/systemd/system/mlrift-uefi-probe.service"
LOG_PATH="/var/log/mlrift_uefi_probe.log"
BIN_PATH="$(cd "$SCRIPT_DIR/.." && pwd)/build/phase3_eta3f_post_uefi_state"

mode="${1:-status}"

case "$mode" in
    install)
        if [[ ! -x "$BIN_PATH" ]]; then
            echo "Binary not found or not executable: $BIN_PATH" >&2
            echo "Build first: ./build/mlrc --arch=x86_64 examples/phase3_eta3f_post_uefi_state.mlr -o build/phase3_eta3f_post_uefi_state" >&2
            exit 2
        fi
        if [[ ! -f "$SERVICE_SRC" ]]; then
            echo "Service unit not found: $SERVICE_SRC" >&2
            exit 3
        fi
        cp "$SERVICE_SRC" "$SERVICE_DST"
        touch "$LOG_PATH"
        chmod 644 "$LOG_PATH"
        systemctl daemon-reload
        systemctl enable mlrift-uefi-probe.service
        echo "Installed. Next boot will append to $LOG_PATH."
        echo "Tail with:   tail -f $LOG_PATH"
        echo "Remove with: sudo $0 remove"
        ;;
    remove)
        systemctl disable mlrift-uefi-probe.service 2>/dev/null || true
        rm -f "$SERVICE_DST"
        systemctl daemon-reload
        echo "Removed. (Log file $LOG_PATH left in place for inspection.)"
        ;;
    status)
        echo "--- service unit ---"
        systemctl status mlrift-uefi-probe.service --no-pager || true
        echo ""
        echo "--- log file ---"
        if [[ -f "$LOG_PATH" ]]; then
            echo "Path: $LOG_PATH"
            echo "Size: $(stat -c%s "$LOG_PATH") bytes"
            echo "Last 40 lines:"
            tail -40 "$LOG_PATH"
        else
            echo "No log file at $LOG_PATH yet."
        fi
        ;;
    *)
        echo "Usage: sudo $0 {install|remove|status}" >&2
        exit 1
        ;;
esac
