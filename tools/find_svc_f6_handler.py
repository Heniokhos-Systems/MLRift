#!/usr/bin/env python3
"""
η.3q-23 — find SVC #0xf6's kernel-side handler in SYS_DRV.

SOC_DRV issues SVC #0xf6 from its exception logger (SRAM 0xcbc).
SYS_DRV must have the kernel-side handler for it.

The SOS SVC vector calls SYS_DRV's SVC dispatcher at file 0x50c4
(= SRAM 0x50c4). That dispatcher does a crypto/hash lookup (not a
direct table) — so we can't directly read off "SVC 0xf6 → fn X"
from a table.

Strategy:
  1. Search SYS_DRV for any function that writes to MP0 C2PMSG_60/
     61/62 (PSP-internal SMN 0x032001f0-0x032001f8).
  2. Search SYS_DRV for any function that loads constants
     0x83f00f80 (fault addr) or 0x0d1102f4 (fault PC handler entry).
  3. Search SYS_DRV for code that handles "type composition" —
     specifically, any function that computes type=4 from inputs.
  4. Search for the value 4 being stored to MP0 regs.

Also: SVC #0xf6 sites in SOC_DRV/INTF_DRV/DBG_DRV — for cross-ref.
"""
import struct
from pathlib import Path
from capstone import Cs, CS_ARCH_ARM, CS_MODE_THUMB

md = Cs(CS_ARCH_ARM, CS_MODE_THUMB); md.skipdata = True

BLOBS = [
    ("sys_drv",  "sys_drv_subblob.bin"),
    ("soc_drv",  "soc_drv_subblob.bin"),
    ("intf_drv", "intf_drv_subblob.bin"),
    ("dbg_drv",  "dbg_drv_subblob.bin"),
]

# Search each blob for various leads
for name, fn in BLOBS:
    p = Path("/home/pantelis/Desktop/Projects/Work/MLRift/captures") / fn
    if not p.exists(): continue
    data = p.read_bytes()
    print(f"\n=== {name} ({len(data)} bytes) ===")

    # Decode entire blob as Thumb
    all_insns = list(md.disasm(data, 0))

    # 1. Find SVC #0xf6 sites
    svc_f6_sites = []
    for ins in all_insns:
        if ins.mnemonic == "svc":
            try:
                num = int(ins.op_str.lstrip("#"), 0)
                if num == 0xf6:
                    svc_f6_sites.append(ins.address)
            except: pass
    print(f"  SVC #0xf6 sites: {len(svc_f6_sites)} — {[hex(s) for s in svc_f6_sites[:10]]}")

    # 2. Constants in 0x032001xx range (MP0 C2PMSG)
    mp0_consts = []
    for o in range(0, len(data) - 3, 4):
        v = struct.unpack_from("<I", data, o)[0]
        if 0x03200180 <= v < 0x032002fc:
            mp0_consts.append((o, v))
    if mp0_consts:
        print(f"  MP0 C2PMSG constants: {len(mp0_consts)}")
        for o, v in mp0_consts[:10]:
            dword_idx = (v - 0x03200000) // 4
            c2_num = dword_idx - 0x40 if dword_idx >= 0x40 else "?"
            print(f"    file 0x{o:x}: 0x{v:08x}  (= C2PMSG_{c2_num})")
    else:
        print(f"  No MP0 C2PMSG constants (aligned 4-byte)")

    # 3. Search for fault constants
    for tgt, lbl in [(0x83f00f80, "fault_addr"),
                     (0x0d1102f4, "fault_PC"),
                     (0xbad1add7, "BAD breadcrumb")]:
        pat = struct.pack("<I", tgt)
        hits = []
        for o in range(0, len(data) - 3, 4):
            if data[o:o+4] == pat:
                hits.append(o)
        if hits:
            print(f"  {lbl} (0x{tgt:08x}): {len(hits)}× at {[hex(h) for h in hits[:5]]}")

# Specifically: for SYS_DRV, find code that stores #4 to a register
# loaded from a literal in 0x03200xxx range
print("\n\n" + "=" * 70)
print("SYS_DRV: scan for code that stores immediate #4 to MP0_C2PMSG_60/61/62")
print("=" * 70)
SYS_DRV = Path("/home/pantelis/Desktop/Projects/Work/MLRift/captures/sys_drv_subblob.bin").read_bytes()
all_insns = list(md.disasm(SYS_DRV, 0))

# Look for any sequence where r0 (or rN) is loaded with literal in 0x032001xx range
# followed within 10 insns by a store of #4 to [r0, #X]
mp0_reg_loads = []  # (addr, register, value)
for i, ins in enumerate(all_insns):
    if "[pc," in ins.op_str:
        try:
            imm = int(ins.op_str.split("[pc,")[1].split("]")[0].strip().lstrip("#"), 0)
            lit_virt = ((ins.address + 4) & ~3) + imm
            if lit_virt + 4 <= len(SYS_DRV):
                lv = int.from_bytes(SYS_DRV[lit_virt:lit_virt+4], "little")
                if 0x03200000 <= lv < 0x03200400:
                    # parse register from op_str
                    reg = ins.op_str.split(",")[0].strip()
                    mp0_reg_loads.append((i, ins.address, reg, lv))
        except: pass

print(f"  Found {len(mp0_reg_loads)} MP0 reg loads in SYS_DRV")
for i, addr, reg, val in mp0_reg_loads[:20]:
    print(f"\n  @ 0x{addr:x}: load {reg} = 0x{val:08x}")
    # Print 10 insns following
    for ins in all_insns[i+1:i+12]:
        tag = ""
        if ins.mnemonic.startswith("str") and reg in ins.op_str:
            tag = "  *** STORE to MP0 reg!"
        op_n = ins.op_str.replace(" ", "")
        if "#4" in op_n and any(f"r{j},#4" in op_n for j in range(8)):
            tag += "  (sets #4)"
        print(f"    0x{ins.address:08x}: {ins.bytes.hex():<10} {ins.mnemonic:<10} {ins.op_str}{tag}")
