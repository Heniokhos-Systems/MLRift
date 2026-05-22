#!/usr/bin/env python3
"""
η.3q-16 — disassemble SOS exception vector table + handlers.

We just discovered SOS's virt base = 0x0d110000 and body_start in
the sub-blob is at file offset 0x100. That means:
  virt 0x0d110000 = file 0x100  (start of body = vector table)
  virt 0x0d110020 = file 0x120  (post-vector-table = vector handlers)
  virt 0x0d1102f4 = file 0x3f4  (our fault PC = Data Abort handler entry)

Standard ARMv7-A vector table layout (8 entries × 4 bytes):
  +0x00 Reset
  +0x04 Undefined Instruction
  +0x08 Supervisor Call (SVC)
  +0x0c Prefetch Abort
  +0x10 Data Abort
  +0x14 Reserved
  +0x18 IRQ
  +0x1c FIQ

Each entry is either:
  - `b <handler>` (relative branch)
  - `ldr pc, [pc, #imm]` (absolute via literal pool)

Tasks:
  1. Disasm SOS file offsets 0x100-0x500 in ARM mode
  2. Identify vector table entries and resolve handler addresses
  3. Disasm each handler fully
  4. Resolve literal pool entries — these point to kernel structs,
     stack bases, page tables, exception logging
  5. Identify the C2PMSG-write code in the Data Abort handler (the
     code that wrote our observed triplet to C2PMSG_60/61/62)
"""
import struct
from pathlib import Path
from capstone import Cs, CS_ARCH_ARM, CS_MODE_ARM, CS_MODE_THUMB

SOS = Path("/home/pantelis/Desktop/Projects/Work/MLRift/captures/sos_subblob.bin").read_bytes()
md_arm = Cs(CS_ARCH_ARM, CS_MODE_ARM); md_arm.skipdata = True

LOAD_ADDR = 0x0d110000
BODY_START = 0x100
VIRT_TO_FILE = lambda v: (v - LOAD_ADDR) + BODY_START
FILE_TO_VIRT = lambda f: (f - BODY_START) + LOAD_ADDR

# Disasm the entire 0x100-0x500 region in ARM mode
print("=" * 70)
print("SOS @ virt 0x0d110000 (file 0x100) — first 0x400 bytes in ARM mode")
print("=" * 70)
chunk = SOS[BODY_START:BODY_START + 0x400]
addr = LOAD_ADDR
# Decode 4-byte instructions at every offset (skipdata enabled)
insns = list(md_arm.disasm(chunk, addr))

# Print the vector table region (first 8 instructions)
print("\n--- VECTOR TABLE (8 entries) ---")
for i, ins in enumerate(insns[:8]):
    vec_name = ["Reset", "Undefined", "SVC", "Prefetch Abort",
                "Data Abort", "Reserved", "IRQ", "FIQ"][i]
    file_off = VIRT_TO_FILE(ins.address)
    note = ""
    # Resolve `ldr pc, [pc, #imm]` literal pool target
    if ins.mnemonic == "ldr" and "pc," in ins.op_str.split(",")[0]:
        try:
            imm = int(ins.op_str.split("[pc,")[1].split("]")[0].strip().lstrip("#"), 0)
            lit_virt = (ins.address + 8) + imm  # ARM PC offset = +8
            lit_file = VIRT_TO_FILE(lit_virt)
            if lit_file + 4 <= len(SOS):
                handler = int.from_bytes(SOS[lit_file:lit_file+4], "little")
                note = f"  /* lit @ 0x{lit_virt:x} (file 0x{lit_file:x}) = handler 0x{handler:08x} */"
        except: pass
    if ins.mnemonic == "b":
        try:
            target = int(ins.op_str.lstrip("#"), 0)
            note = f"  /* target = 0x{target:08x} */"
        except: pass
    print(f"  +0x{ins.address - LOAD_ADDR:02x}  [{vec_name:<15}] {ins.bytes.hex():<10} {ins.mnemonic:<6} {ins.op_str}{note}")

# Print full handler region disasm (focus around fault PC 0x0d1102f4 ± 0x80)
print("\n--- DATA ABORT HANDLER and surrounding code ---")
FAULT_PC = 0x0d1102f4
FOCUS_VIRT_START = FAULT_PC - 0x40
FOCUS_VIRT_END   = FAULT_PC + 0x200
for ins in insns:
    if not (FOCUS_VIRT_START <= ins.address < FOCUS_VIRT_END):
        continue
    tag = ""
    if ins.address == FAULT_PC:
        tag = "  <<< FAULT PC (handler entry)"
    if ins.mnemonic == "ldr" and "[pc," in ins.op_str:
        try:
            imm = int(ins.op_str.split("[pc,")[1].split("]")[0].strip().lstrip("#"), 0)
            lit_virt = (ins.address + 8) + imm
            lit_file = VIRT_TO_FILE(lit_virt)
            if 0 <= lit_file + 4 <= len(SOS):
                lv = int.from_bytes(SOS[lit_file:lit_file+4], "little")
                note = ""
                if 0x03200000 <= lv < 0x03200400: note = " MP0_REG"
                elif 0x03010000 <= lv < 0x03020000: note = " MP1_REG"
                elif 0x0d100000 <= lv < 0x0d200000: note = " SOS_virt"
                elif 0xc000 <= lv < 0xd000: note = " PSP_SRAM_PT_region"
                elif lv < 0x40000: note = " PSP_SRAM"
                tag += f"  /* lit @ 0x{lit_virt:x} = 0x{lv:08x}{note} */"
        except: pass
    print(f"  0x{ins.address:08x}: {ins.bytes.hex():<10} {ins.mnemonic:<10} {ins.op_str}{tag}")

# Dump LITERAL POOL entries in this region — find 32-bit words that
# look like meaningful constants (MMIO addresses, fn pointers)
print("\n--- LITERAL POOL constants in vector handler region (0x100-0x500) ---")
seen = {}
for off in range(BODY_START, BODY_START + 0x400, 4):
    v = int.from_bytes(SOS[off:off+4], "little")
    if v == 0: continue
    note = ""
    if 0x03200000 <= v < 0x03200400: note = "  ← MP0_C2PMSG_RANGE"
    elif v == 0x032000d8: note = "  ← DEBUG_STATUS"
    elif 0x03010000 <= v < 0x03020000: note = "  ← MP1_SMU"
    elif 0x0d100000 <= v < 0x0d200000: note = "  ← SOS_internal_addr"
    elif 0xc000 <= v < 0xd000: note = "  ← PSP_SRAM_PT_region"
    elif v < 0x40000 and v > 0x1000: note = "  ← PSP_SRAM"
    elif v == 0xbad1add7: note = "  ← BAD breadcrumb (from fn 0x4e40)"
    if note:
        virt = FILE_TO_VIRT(off)
        print(f"  file 0x{off:x} (virt 0x{virt:08x}): 0x{v:08x}{note}")
