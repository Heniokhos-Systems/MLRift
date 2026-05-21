#!/usr/bin/env python3
"""
η.3j-RE-2b — find MMU-enable point + mode-switch boundary + literal pool.

Critical landmarks to locate:
  1. SCTLR enable: a write to c1,c0,#0 where the value has bit 0 (M) set.
     The instruction *before* the write loads SCTLR, ORs in 1, writes back.
     After this point, addresses are VIRTUAL.
  2. Real ARM→Thumb mode switch: any BX/BLX where the target address has
     bit 0 set. The instruction before usually `ORR Rm, Rm, #1` or `ADD Rm, #1`.
  3. Literal pool boundary: contiguous block of LDR Rd, [PC, #imm] points
     into a region; that region IS the pool (data, not code).
"""

import struct
from pathlib import Path
from capstone import Cs, CS_ARCH_ARM, CS_MODE_ARM, CS_MODE_THUMB

BLOB = Path("/home/pantelis/Desktop/Projects/Work/MLRift/captures/sos_subblob.bin")
data = BLOB.read_bytes()

md = Cs(CS_ARCH_ARM, CS_MODE_ARM)
md.detail = True

print("=== Walk 0x230 → 0x600 (post-relocation init) — flag SCTLR/TLB/mode-switch ===")
chunk = data[0x230:0x600]
for ins in md.disasm(chunk, 0x230):
    flag = ""
    m = ins.mnemonic
    op = ins.op_str
    # CP15 SCTLR write: mcr p15, #0, Rd, c1, c0, #0
    if m.startswith("mcr") and "c1, c0" in op and "p15" in op:
        flag = "  <<< SCTLR WRITE (MMU/cache enable)"
    elif m.startswith("mrc") and "c1, c0" in op and "p15" in op:
        flag = "  <-- SCTLR READ"
    # Bit-0 manipulation suggesting Thumb-bit set
    elif m == "orr" and ", #1" in op:
        flag = "  <-- ORR ..., #1 (possible Thumb-bit set)"
    elif m == "add" and ", #1" in op and "pc" not in op:
        flag = "  <-- ADD ..., #1 (possible Thumb-bit set)"
    elif m == "bx" or m == "blx":
        flag = "  <-- BRANCH"
    elif m == "ldr" and "[pc" in op:
        try:
            imm = int(op.split("#")[1].rstrip("]"), 0)
            la = ins.address + 8 + imm
            if 0 <= la < len(data) - 4:
                v = struct.unpack_from("<I", data, la)[0]
                flag = f"  -> *0x{la:x} = 0x{v:08x}"
        except Exception:
            pass
    print(f"  0x{ins.address:08x}: {ins.bytes.hex():<10} {m:<8} {op:<40}{flag}")
