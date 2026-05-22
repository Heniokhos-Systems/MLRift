#!/usr/bin/env python3
"""
η.3q-18 — test the theory that SYS_DRV = the PSP kernel/OS itself.

If true, SYS_DRV's load_addr = 0x00000000 and body_start = 0x1000,
which would put:
  - PSP SRAM 0x2ee0  (kernel main entry, called from SOS reset)
    → SYS_DRV file 0x3ee0
  - PSP SRAM 0x50c4  (SVC dispatcher, called from SOS SVC vector)
    → SYS_DRV file 0x60c4
  - PSP SRAM 0xcbc   (exception logger, called from common excep)
    → SYS_DRV file 0x1cbc

For each candidate addr, disasm SYS_DRV at the computed file offset
in Thumb mode. If we see plausible code (function prologues with
push, sensible mnemonics), the theory is confirmed.

Also test load_addr values: 0x0, 0x100, 0x1000, plus the PSP SRAM
0xC000 range (in case SYS_DRV loads to PT region — unlikely but check).
"""
from pathlib import Path
from capstone import Cs, CS_ARCH_ARM, CS_MODE_THUMB, CS_MODE_ARM

SYS_DRV = Path("/home/pantelis/Desktop/Projects/Work/MLRift/captures/sys_drv_subblob.bin").read_bytes()
md_thumb = Cs(CS_ARCH_ARM, CS_MODE_THUMB); md_thumb.skipdata = True
md_arm   = Cs(CS_ARCH_ARM, CS_MODE_ARM);   md_arm.skipdata = True

print(f"SYS_DRV size: {len(SYS_DRV)} bytes (0x{len(SYS_DRV):x})\n")

# PSP SRAM addresses we want to find code at
TARGETS = [
    (0x2ee0, "kernel main entry (called from SOS Reset)"),
    (0x50c4, "SVC dispatcher (called from SOS SVC handler)"),
    (0x0cbc, "exception logger (called from common logger)"),
    (0x0e2c, "IRQ user handler (called from IRQ vector via 0xe2d)"),
]

# Plausible load_addr / body_start combinations
LOAD_CONFIGS = [
    (0x0000_0000, 0x1000, "load=0, body=0x1000"),
    (0x0000_0000, 0x0100, "load=0, body=0x100"),
    (0x0000_0000, 0x0400, "load=0, body=0x400"),
    (0x0000_0000, 0x0000, "load=0, body=0  (direct)"),
]

for load_addr, body_start, label in LOAD_CONFIGS:
    print(f"\n{'=' * 70}")
    print(f"CONFIG: {label}")
    print(f"{'=' * 70}")
    for tgt_addr, tgt_desc in TARGETS:
        # Compute file offset: virt_addr - load_addr + body_start
        # For Thumb addrs (bit 0 set), strip bit 0
        actual_addr = tgt_addr & ~1
        if actual_addr < load_addr or actual_addr >= load_addr + (len(SYS_DRV) - body_start):
            print(f"\n  {tgt_desc} @ 0x{tgt_addr:x}: out of range for this config")
            continue
        file_off = (actual_addr - load_addr) + body_start
        if file_off + 32 > len(SYS_DRV):
            continue
        # Disasm Thumb at file_off
        chunk = SYS_DRV[file_off:file_off + 64]
        insns = list(md_thumb.disasm(chunk, actual_addr))[:16]
        plausible = False
        # Quality heuristic: first 2-3 insns include push/mov/etc., not nops or garbage
        if insns:
            first_mns = [ins.mnemonic for ins in insns[:5]]
            if any(m in ("push", "push.w") for m in first_mns):
                plausible = True
            if "movs r0, r0" not in [f"{i.mnemonic} {i.op_str}" for i in insns[:3]]:
                plausible = plausible or True
        marker = "  *** LOOKS LIKE CODE!" if plausible else ""
        print(f"\n  {tgt_desc}: SYS_DRV file 0x{file_off:x} ({len(insns)} Thumb insns){marker}")
        print(f"    bytes: {chunk[:16].hex()}")
        for ins in insns[:8]:
            print(f"      0x{ins.address:08x}: {ins.bytes.hex():<10} {ins.mnemonic:<10} {ins.op_str}")
