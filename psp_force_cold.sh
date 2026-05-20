#!/bin/bash
# psp_force_cold.sh — attempt software-only PSP "cold reset" on the GPU.
#
# Tries multiple methods, escalating in aggressiveness. After each
# attempt, reads PSP state via phase3_eta0_psp_state and reports.
#
# REALITY CHECK: AMD GPUs have a PSP aux-power-island that survives
# FLR, SBR, and (usually) D3cold. The ONLY reliable cold reset is a
# full machine power cycle. This script tries the longshot paths so
# we can confirm they don't work for *our* setup, and avoid burning
# 5-minute cold cycles when they're not strictly needed.
#
# Usage:  sudo ./psp_force_cold.sh
# Exit:   0 if PSP is back in COLD-BL, 2 if cold cycle needed, 1 on error.

set -u
PCI_DEV="0000:03:00.0"
SYSFS="/sys/bus/pci/devices/${PCI_DEV}"

if [[ "$EUID" -ne 0 ]]; then
    echo "ERROR: must run as root (sudo ./psp_force_cold.sh)"
    exit 1
fi

if [[ ! -d "$SYSFS" ]]; then
    echo "ERROR: device $PCI_DEV not found"
    exit 1
fi

PROBE="./build/phase3_eta0_psp_state"
if [[ ! -x "$PROBE" ]]; then
    echo "ERROR: $PROBE not built. Run from MLRift repo root."
    exit 1
fi

run_probe() {
    local label="$1"
    echo
    echo "==== probe: ${label} ===="
    "$PROBE"
    local rc=$?
    echo "==== probe exit code: ${rc} ===="
    return $rc
}

settle() { sleep "${1:-1}"; }

# ------------------------- baseline state -------------------------
echo "############################################################"
echo "# psp_force_cold.sh  device=${PCI_DEV}"
echo "# baseline PSP state before any reset attempt"
echo "############################################################"
run_probe "baseline"
BASELINE=$?

if [[ $BASELINE -eq 0 ]]; then
    echo
    echo ">>> Already in COLD-BL — no reset needed."
    exit 0
fi

# ------------------------- method 1: sysfs /reset (SBR) -------------------------
# This is what vfio_open already does. Unlikely to clear PSP, but
# document empirically.
echo
echo "############################################################"
echo "# method 1: sysfs SBR via /sys/.../reset"
echo "# (note: vfio_open already does this; should be a no-op)"
echo "############################################################"
echo 1 > "${SYSFS}/reset" 2>&1 || echo "(reset write failed — device probably busy with vfio FD)"
settle 1
run_probe "after sysfs SBR"
if [[ $? -eq 0 ]]; then
    echo
    echo ">>> SBR cleared PSP — unexpected but use it."
    exit 0
fi

# ------------------------- method 2: device remove + rescan -------------------------
echo
echo "############################################################"
echo "# method 2: PCI remove + rescan"
echo "############################################################"
echo "  unbind / remove device from PCI subsystem..."
echo 1 > "${SYSFS}/remove"
settle 2
echo "  triggering bus rescan..."
echo 1 > /sys/bus/pci/rescan
settle 3
if [[ ! -d "$SYSFS" ]]; then
    echo "ERROR: device did NOT re-enumerate after rescan."
    exit 1
fi
echo "  device re-enumerated; driver: $(readlink ${SYSFS}/driver 2>/dev/null | xargs -r basename || echo none)"
run_probe "after remove+rescan"
if [[ $? -eq 0 ]]; then
    echo
    echo ">>> remove+rescan cleared PSP!"
    exit 0
fi

# ------------------------- method 3: D3cold via runtime PM -------------------------
echo
echo "############################################################"
echo "# method 3: D3cold transition via runtime PM"
echo "############################################################"
if [[ "$(cat ${SYSFS}/d3cold_allowed 2>/dev/null)" != "1" ]]; then
    echo "  d3cold_allowed=0 — skipping"
else
    echo "  current power_state: $(cat ${SYSFS}/power_state)"
    echo "  setting power/control=auto..."
    echo auto > "${SYSFS}/power/control" 2>&1 || true
    echo "  waiting 5s for runtime suspend..."
    settle 5
    STATE=$(cat "${SYSFS}/power_state" 2>/dev/null)
    echo "  power_state after wait: ${STATE}"
    if [[ "${STATE}" == "D3cold" ]]; then
        echo "  in D3cold; bringing back to D0..."
        echo on > "${SYSFS}/power/control" 2>&1 || true
        settle 2
    else
        echo "  device did NOT enter D3cold (held by binding / no consumer)."
    fi
    run_probe "after D3cold cycle"
    if [[ $? -eq 0 ]]; then
        echo
        echo ">>> D3cold cycle cleared PSP!"
        exit 0
    fi
fi

# ------------------------- method 4: unbind vfio + D3cold attempt -------------------------
# Method 3 only reached D3hot because vfio-pci binding holds a power
# ref. Try unbinding first so the kernel can let device drop deeper.
echo
echo "############################################################"
echo "# method 4: unbind vfio-pci, runtime-PM idle, D3cold check"
echo "############################################################"
DRIVER_LINK="${SYSFS}/driver"
if [[ -L "$DRIVER_LINK" ]]; then
    CURRENT_DRIVER=$(readlink "$DRIVER_LINK" | xargs basename)
    echo "  current driver: ${CURRENT_DRIVER}"
    echo "  unbinding from ${CURRENT_DRIVER}..."
    echo "$PCI_DEV" > "/sys/bus/pci/drivers/${CURRENT_DRIVER}/unbind" 2>&1 || \
        echo "  (unbind failed — device may already be unbound)"
    settle 2
    echo "  power_state after unbind: $(cat ${SYSFS}/power_state)"
    echo "  setting power/control=auto, waiting 8s for D3cold..."
    echo auto > "${SYSFS}/power/control" 2>&1 || true
    settle 8
    STATE=$(cat "${SYSFS}/power_state" 2>/dev/null)
    echo "  power_state: ${STATE}"
    # Re-bind vfio-pci so the probe can re-open the device
    echo "  re-binding to vfio-pci..."
    echo "$PCI_DEV" > /sys/bus/pci/drivers/vfio-pci/bind 2>&1 || \
        echo "  (vfio-pci bind failed — may need driver_override)"
    settle 2
    run_probe "after unbind + D3cold attempt"
    if [[ $? -eq 0 ]]; then
        echo
        echo ">>> unbind + D3cold cleared PSP!"
        exit 0
    fi
else
    echo "  no driver bound — skipping"
fi

# ------------------------- verdict -------------------------
echo
echo "############################################################"
echo "# VERDICT: no software-only reset cleared PSP wedge"
echo "############################################################"
echo "PSP aux-power-island survived FLR, SBR, remove+rescan,"
echo "and (on this platform) D3cold. A full power cycle is required."
echo
echo "  sudo shutdown -h now  # then 15+ s with PSU off, then power on"
echo
exit 2
