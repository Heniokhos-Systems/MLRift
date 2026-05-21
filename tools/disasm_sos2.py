#!/usr/bin/env python3
"""
η.3j-RE-2 — walk ARM-mode init from the *actual* reset handler entry.

The vector-table B instruction at 0x100 (E A 00 00 20 LE) branches to:
    target = PC + 8 + offset*4 = 0x108 + 0x80 = 0x188
Not 0x190 as we used in disasm_sos.py — that skipped the first 2 insns.

This script disassembles a longer window from 0x188 onwards and flags:
  - MRC/MCR (CP15 coprocessor access) — system-register init
  - BX/BLX with bit-0 of target = 1 — ARM→Thumb mode switch
  - LDR Rd, [PC, #imm] — extract literal-pool constants
  - LDR Rd, =addr where Rd is a likely MMIO base register

Also follows direct branches (B/BL) to find the entry of the next
function/block (instead of stopping at end of linear range).
"""

import struct
from pathlib import Path
from capstone import Cs, CS_ARCH_ARM, CS_MODE_ARM, CS_MODE_THUMB

BLOB = Path("/home/pantelis/Desktop/Projects/Work/MLRift/captures/sos_subblob.bin")
data = BLOB.read_bytes()
print(f"loaded {BLOB} ({len(data)} bytes, 0x{len(data):x})")
print()

md_arm   = Cs(CS_ARCH_ARM, CS_MODE_ARM)
md_arm.detail = True
md_thumb = Cs(CS_ARCH_ARM, CS_MODE_THUMB)
md_thumb.detail = True

def disasm_arm(file_off, length, label=""):
    if label:
        print(f"=== {label} ===")
    chunk = data[file_off:file_off + length]
    for ins in md_arm.disasm(chunk, file_off):
        # Flag interesting instructions.
        flag = ""
        m = ins.mnemonic
        if m.startswith("mrc") or m.startswith("mcr"):
            flag = "  <-- CP15 sysreg"
        elif m.startswith("bx") or m.startswith("blx"):
            flag = "  <-- MODE SWITCH CANDIDATE"
        elif m == "ldr" and "[pc" in ins.op_str:
            # Extract literal-pool value.
            try:
                rd, src = ins.op_str.split(", ", 1)
                imm = int(src.split("#")[1].rstrip("]"), 0)
                lit_addr = ins.address + 8 + imm  # ARM PC = +8
                if 0 <= lit_addr < len(data) - 4:
                    val = struct.unpack_from("<I", data, lit_addr)[0]
                    flag = f"  -> lit@0x{lit_addr:x} = 0x{val:08x}"
            except Exception:
                pass
        print(f"  0x{ins.address:08x}: {ins.bytes.hex():<10} {ins.mnemonic:<8} {ins.op_str:<35}{flag}")

print("=== ACTUAL RESET HANDLER @ 0x188 (Vector table B at 0x100 → 0x108+0x80) ===")
disasm_arm(0x188, 0x100, "")  # 64 instructions

print()
print("=== After memcpy, look at branch targets from reset handler ===")
# The reset handler at 0x188 starts with what looks like detection +
# memcpy. After memcpy completes, it should jump elsewhere (to a setup
# function or to the relocated body). Look for unconditional B/BL.
unconditional_branches = []
for off in range(0x188, 0x500, 4):
    if off + 4 > len(data):
        break
    w = struct.unpack_from("<I", data, off)[0]
    cond = (w >> 28) & 0xF
    op   = (w >> 24) & 0xF
    if cond == 0xE and op in (0xA, 0xB):  # AL B (0xA) or BL (0xB)
        offset = w & 0xFFFFFF
        if offset & 0x800000:
            offset |= 0xFF000000
        offset = struct.unpack("<i", struct.pack("<I", offset))[0]
        target = off + 8 + offset * 4
        kind = "BL" if op == 0xB else "B"
        unconditional_branches.append((off, kind, target))
        if 0 <= target < len(data):
            tgt_w = struct.unpack_from("<I", data, target)[0]
            print(f"  0x{off:08x}: {kind} -> 0x{target:08x}  (target word = 0x{tgt_w:08x})")

print()
print("=== Search for BX/BLX (mode-switch candidates) in [0x188, 0x20000] ===")
# BX Rm = 0xE12FFF1x, BLX Rm = 0xE12FFF3x
for off in range(0x188, min(len(data) - 4, 0x20000), 4):
    w = struct.unpack_from("<I", data, off)[0]
    if (w & 0xFFFFFFF0) == 0xE12FFF10:  # BX Rm
        rm = w & 0xF
        print(f"  0x{off:08x}: BX r{rm}  ; could switch to Thumb if r{rm}[0]==1")
    elif (w & 0xFFFFFFF0) == 0xE12FFF30:  # BLX Rm
        rm = w & 0xF
        print(f"  0x{off:08x}: BLX r{rm}  ; could switch to Thumb if r{rm}[0]==1")
    # BLX <imm> (always switches mode): 0xFAxxxxxx (unconditional, cond=0xF)
    elif (w >> 24) == 0xFA or (w >> 24) == 0xFB:
        offset = w & 0xFFFFFF
        if offset & 0x800000:
            offset |= 0xFF000000
        offset = struct.unpack("<i", struct.pack("<I", offset))[0]
        h = (w >> 24) & 1  # H bit for halfword offset in BLX-imm
        target = off + 8 + offset * 4 + (h * 2)
        print(f"  0x{off:08x}: BLX 0x{target:08x}  ; ALWAYS switches mode (ARM <-> Thumb)")

print()
print(f"Done. Found {len(unconditional_branches)} unconditional branches in reset handler region.")
