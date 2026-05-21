#!/usr/bin/env python3
"""
Initial disassembly walk of captures/sos_subblob.bin.

Per agent: ARM Cortex-A5, ARMv7-A, 32-bit LE, load=0x0, entry at file
offset 0x190 (after vector table @ 0x100).

This script walks:
  1. Vector table at 0x100 — verify branches go to expected slots
  2. Reset handler at 0x190 — disassemble first 64 instructions
  3. Search for MMIO accesses (loads/stores with PSP-known register
     base patterns, e.g., addresses near 0x03010000 SMN range)
"""

import struct
from pathlib import Path
from capstone import Cs, CS_ARCH_ARM, CS_MODE_ARM

BLOB = Path("/home/pantelis/Desktop/Projects/Work/MLRift/captures/sos_subblob.bin")
data = BLOB.read_bytes()
print(f"loaded {BLOB} ({len(data)} bytes)")
print()

md = Cs(CS_ARCH_ARM, CS_MODE_ARM)
md.detail = True

def disasm_at(file_off, n_insns, label=""):
    print(f"=== {label} (file_off=0x{file_off:x}, vaddr=0x{file_off:x}) ===")
    chunk = data[file_off:file_off + n_insns * 4]
    for ins in md.disasm(chunk, file_off):
        print(f"  0x{ins.address:08x}: {ins.bytes.hex():<10} {ins.mnemonic:<8} {ins.op_str}")
        if ins.address - file_off >= (n_insns - 1) * 4:
            break
    print()

# 1. Vector table at 0x100
disasm_at(0x100, 8, "VECTOR TABLE @ 0x100")

# 2. Reset handler at 0x190
disasm_at(0x190, 32, "RESET HANDLER @ 0x190 (first 32 insns)")

# 3. Look for unique PC-relative loads (LDR Rd, [PC, #imm]) — these
# encode constants like MMIO addresses. ARM encoding:
# 1110_01x1_x0x1_1111_dddd_aaaa_aaaa_aaaa (with U/Imm12)
# i.e., 0xE51Fxxxx or 0xE59Fxxxx
print("=== PC-relative loads (LDR Rd, [PC, #imm]) — first 40 in [0x190, 0x2000] ===")
count = 0
for off in range(0x190, min(len(data) - 4, 0x2000), 4):
    w = struct.unpack_from("<I", data, off)[0]
    # LDR Rd, [PC, #+imm12]:  0xE59Fxxxx  (cond=14, !P U !B !W L, Rn=15, Rd, Imm12)
    if (w & 0xFFFFF000) in (0xE59F0000, 0xE59F1000, 0xE59F2000, 0xE59F3000,
                             0xE59F4000, 0xE59F5000, 0xE59F6000, 0xE59F7000,
                             0xE59F8000, 0xE59F9000, 0xE59FA000, 0xE59FB000,
                             0xE59FC000, 0xE59FD000, 0xE59FE000):
        imm12 = w & 0xFFF
        rd = (w >> 12) & 0xF
        # PC is 8 ahead in ARM mode
        const_addr = off + 8 + imm12
        if 0 <= const_addr < len(data) - 4:
            const_val = struct.unpack_from("<I", data, const_addr)[0]
            print(f"  0x{off:08x}: ldr r{rd}, [pc, #0x{imm12:x}]  -> *0x{const_addr:x} = 0x{const_val:08x}")
            count += 1
            if count >= 40:
                break

print()
print(f"Found {count} PC-relative loads (each could be an MMIO address constant).")
