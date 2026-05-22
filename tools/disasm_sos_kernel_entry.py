#!/usr/bin/env python3
"""
η.3q-17 — disasm SOS kernel entry points in plaintext:
  - Reset handler          @ virt 0x0d110088 (file 0x188)
  - Undefined Instruction  @ virt 0x0d1101f8 (file 0x2f8)
  - SVC handler            @ virt 0x0d110228 (file 0x328)
  - Common exception logger @ virt 0x0d110288 (file 0x388)
  - IRQ secondary handler  @ virt 0x00000e2d (= 0x0e2d in PSP SRAM)

The mission: find WHAT writes type=4 to C2PMSG_60 (since Data Abort
sets type=2 not 4). Candidates:
  (a) Common logger has additional logic before writing
  (b) SVC handler escalates certain SVCs to type-4 faults
  (c) A different fault entry we haven't decoded
"""
import struct
from pathlib import Path
from capstone import Cs, CS_ARCH_ARM, CS_MODE_ARM, CS_MODE_THUMB

SOS = Path("/home/pantelis/Desktop/Projects/Work/MLRift/captures/sos_subblob.bin").read_bytes()
md_arm   = Cs(CS_ARCH_ARM, CS_MODE_ARM);   md_arm.skipdata = True
md_thumb = Cs(CS_ARCH_ARM, CS_MODE_THUMB); md_thumb.skipdata = True

LOAD_ADDR = 0x0d110000
BODY_START = 0x100
VIRT_TO_FILE = lambda v: (v - LOAD_ADDR) + BODY_START

def disasm_region(virt_start, virt_end, mode_label, md, label):
    print(f"\n{'=' * 70}")
    print(f"{label} (virt 0x{virt_start:x}-0x{virt_end:x}, {mode_label})")
    print(f"{'=' * 70}")
    file_start = VIRT_TO_FILE(virt_start)
    file_end = VIRT_TO_FILE(virt_end)
    if file_end > len(SOS): file_end = len(SOS)
    chunk = SOS[file_start:file_end]
    insns = list(md.disasm(chunk, virt_start))
    for ins in insns:
        tag = ""
        # Resolve literal pool refs (ARM uses PC+8, Thumb uses (PC+4) & ~3)
        if ins.mnemonic in ("ldr", "ldr.w") and "[pc," in ins.op_str.replace(" ", ""):
            try:
                imm = int(ins.op_str.split("[pc,")[1].split("]")[0].strip().lstrip("#"), 0)
                if mode_label == "ARM":
                    lit_virt = (ins.address + 8) + imm
                else:
                    lit_virt = ((ins.address + 4) & ~3) + imm
                lit_file = VIRT_TO_FILE(lit_virt)
                if 0 <= lit_file + 4 <= len(SOS):
                    lv = int.from_bytes(SOS[lit_file:lit_file+4], "little")
                    note = ""
                    if 0x03200000 <= lv < 0x03200400: note = "  MP0_REG"
                    elif lv == 0x032000d8: note = "  DEBUG_STATUS"
                    elif 0x03010000 <= lv < 0x03020000: note = "  MP1_SMU"
                    elif 0x0d100000 <= lv < 0x0d200000: note = "  SOS_internal"
                    elif 0xc000 <= lv < 0xd000: note = "  PT_region"
                    elif lv < 0x40000: note = "  PSP_SRAM"
                    tag = f"  /* lit @ 0x{lit_virt:x} = 0x{lv:08x}{note} */"
            except: pass
        if ins.mnemonic in ("bl", "blx"):
            tag += "  → CALL"
        if ins.mnemonic == "svc":
            tag += f"  → SVC TRAP"
        # Look for the specific value 4 being moved
        op_norm = ins.op_str.replace(" ", "")
        if ("#4" in op_norm and "r0,#4" in op_norm) or "r0,#0x4," in op_norm:
            tag += "  *** writes 4 to r0 (= type 4?)"
        if "#0x32" in op_norm and ",#0x32" in op_norm:
            tag += "  *** 0x32 imm"
        # str to C2PMSG_60 area would be [smn_base, #0x1f0] — flag those
        if "str" in ins.mnemonic and "[" in ins.op_str:
            if "#0x1f0" in ins.op_str or "#0x1f4" in ins.op_str or "#0x1f8" in ins.op_str:
                tag += "  *** STORE TO C2PMSG_60/61/62 RANGE"
        print(f"  0x{ins.address:08x}: {ins.bytes.hex():<10} {ins.mnemonic:<10} {ins.op_str}{tag}")


# 1. Common exception logger
disasm_region(0x0d110288, 0x0d1102ac, "ARM", md_arm,
              "COMMON EXCEPTION LOGGER (called from PrefAbort/DataAbort)")

# 2. SVC handler
disasm_region(0x0d110228, 0x0d110288, "ARM", md_arm,
              "SVC HANDLER (kernel-call dispatcher)")

# 3. Reset / SOS entry
disasm_region(0x0d110088, 0x0d1101f8, "ARM", md_arm,
              "RESET HANDLER (SOS entry point)")

# 4. Undefined Instruction handler
disasm_region(0x0d1101f8, 0x0d110228, "ARM", md_arm,
              "UNDEFINED INSTRUCTION HANDLER")

# 5. Look up the IRQ user-handler at PSP SRAM 0xe2d (try Thumb)
print("\n" + "=" * 70)
print("IRQ user-mode handler @ PSP SRAM 0x0e2d (try Thumb, +0x500 offset)")
print("=" * 70)
# 0x0e2d at PSP SRAM — bit 0 = 1 means Thumb mode; actual addr = 0xe2c
# In our SOS sub-blob, plaintext R1 starts at 0x500. Virt SRAM 0xe2c
# might map to file 0xe2c if SOS is loaded at virt 0 in SRAM. Check.
# Actually 0x0e2d is in PSP SRAM, not SOS sub-blob.
# Skip this for now — would need PSP SRAM dump.
print("  (SRAM dump needed)")

# 6. Find any code path that writes #4 to r0 followed by a store to
#    [smn, #0x1f0] (C2PMSG_60)
print("\n" + "=" * 70)
print("Scan ENTIRE plaintext for: store of #4 followed by store to C2PMSG_60")
print("=" * 70)
# Decode all plaintext regions as both ARM and Thumb
regions = [(0x100, 0x500, md_arm, "ARM"),
           (0x500, 0x6D00, md_thumb, "Thumb"),
           (0x15140, 0x19180, md_thumb, "Thumb")]
all_insns = []
for s, e, md, lbl in regions:
    region_insns = list(md.disasm(SOS[s:e], (LOAD_ADDR + s - BODY_START) if s < 0x500 else s))
    all_insns.extend([(ins, lbl) for ins in region_insns])

# Find any `mov r0, #4` insn
count_4 = 0
for i, (ins, lbl) in enumerate(all_insns):
    op = ins.op_str.replace(" ", "")
    if ins.mnemonic in ("movs", "mov.w", "mov") and (op == "r0,#4" or op.startswith("r0,#4,")):
        count_4 += 1
        # Print context
        print(f"  [{lbl}] 0x{ins.address:08x}: {ins.mnemonic} {ins.op_str}  (#{count_4})")
        # Check next 8 insns for a store to PSP SRAM or C2PMSG range
        for j in range(i+1, min(len(all_insns), i+10)):
            nxt, _ = all_insns[j]
            if nxt.mnemonic.startswith("str") or nxt.mnemonic.startswith("stm"):
                print(f"      next store: 0x{nxt.address:08x}: {nxt.mnemonic} {nxt.op_str}")
                break
            if nxt.mnemonic in ("bl", "blx", "b", "b.w"):
                print(f"      next branch: 0x{nxt.address:08x}: {nxt.mnemonic} {nxt.op_str}")
                break
        if count_4 > 30: break
