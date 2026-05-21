#!/usr/bin/env python3
"""
η.3j-RE-2c — disassemble the first Thumb-2 functions called from the
ARM-mode reset handler.

Key Thumb entry points (from literal pool 0x490-0x4D4):
  0x3dc0  - first Thumb call (BLX from 0x28C, BEFORE MMU enable)
  0x2ee0  - second target (BX from 0x2F4, around MMU enable)
  0x50c4  - exception handler dispatch (BLX from 0x368)
  0x0cbc  - common exception handler core (BLX from 0x38C)

Disassemble each in Thumb-2 mode. Flag MMIO-looking accesses.
"""

import struct
from pathlib import Path
from capstone import Cs, CS_ARCH_ARM, CS_MODE_THUMB

BLOB = Path("/home/pantelis/Desktop/Projects/Work/MLRift/captures/sos_subblob.bin")
data = BLOB.read_bytes()

md = Cs(CS_ARCH_ARM, CS_MODE_THUMB)
md.detail = True

def disasm_t(off, length, label):
    print(f"=== {label} @ 0x{off:x} ===")
    chunk = data[off:off + length]
    for ins in md.disasm(chunk, off):
        flag = ""
        m = ins.mnemonic
        op = ins.op_str
        # Thumb literal load: ldr Rd, [pc, #imm]
        if m == "ldr" and "[pc" in op:
            try:
                imm = int(op.split("#")[1].rstrip("]"), 0)
                # Thumb PC = (current_pc + 4) & ~3
                la = (ins.address + 4 + imm) & ~3
                if 0 <= la < len(data) - 4:
                    v = struct.unpack_from("<I", data, la)[0]
                    flag = f"  -> *0x{la:x} = 0x{v:08x}"
            except Exception:
                pass
        elif m == "ldr.w" and "[pc" in op:
            try:
                imm = int(op.split("#")[1].rstrip("]"), 0)
                la = (ins.address + 4 + imm) & ~3
                if 0 <= la < len(data) - 4:
                    v = struct.unpack_from("<I", data, la)[0]
                    flag = f"  -> *0x{la:x} = 0x{v:08x}"
            except Exception:
                pass
        elif m in ("bx", "blx", "bl", "b"):
            flag = "  <-- BRANCH"
        elif m == "movw" and "#" in op:
            flag = "  <-- IMMEDIATE LOAD (low half)"
        elif m == "movt" and "#" in op:
            flag = "  <-- IMMEDIATE LOAD (high half)"
        elif m in ("str", "ldr", "str.w", "ldr.w") and "[" in op:
            flag = "  ← MEMORY ACCESS"
        print(f"  0x{ins.address:08x}: {ins.bytes.hex():<10} {m:<8} {op:<35}{flag}")
    print()

# First Thumb function entry points (subtract 1 for bit-0 Thumb marker).
disasm_t(0x3dc0, 0x100, "FIRST THUMB CALL (target of BLX r2 from 0x28C)")
disasm_t(0x2ee0, 0x80,  "Thumb branch target from 0x2F4")
disasm_t(0x0cbc, 0x80,  "Common exception handler (from 0x38C)")
