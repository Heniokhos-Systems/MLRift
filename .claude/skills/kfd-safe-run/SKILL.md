---
name: kfd-safe-run
description: Use before running any GPU launcher (.co / amdgpu-native / KFD). Enforces the hardware safety rules so a run can't wedge MES or reboot the box.
---

Non-negotiable GPU run rules (violating these has rebooted the machine):
1. EVERY launcher calls hipkfd_teardown() before exit — drains all 4 KFD queues; skipping accumulates MES wedge. No exceptions.
2. Build GPU launchers with --target=amdgpu-native, else libamdhip64 DT_NEEDED → silent CPU bf16 fallback (fake "GPU" numbers).
3. NEVER blanket-scan BARs — only IP-discovery-published ranges; a BAR5 sweep → MCE → reboot.
4. cold-boot/warm-handoff driver work needs headless: no compositor; only ONE run per boot under live KWin. amdgpu inference (KFD shim) is the safe path — use it for bench/train.
5. NEVER FLR; cold power cycle is the only PSP unwedge. If GPU wedged: STOP, report, do not reboot or rmmod-loop.
Production .co = MLRift --emit-amdgpu-*-v2 only; no hipcc/ROCm/clang in the build.