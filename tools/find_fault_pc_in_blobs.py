#!/usr/bin/env python3
"""
η.3q-15 — locate code at fault PC 0x0d1102f4 by testing plausible
load addresses for each plaintext sub-blob.

The fault PC's low offset 0x2f4 (756 bytes) is suspiciously small — it
looks like EARLY code in some module. Common PSP module load bases
include 0x0d100000, 0x0d110000, 0x0d000000, 0x10000000, 0x20000000.

For each blob, try these candidate load_addrs:
  - 0x0d100000, 0x0d110000 (matches fault PC's high half)
  - 0x0d000000, 0x10000000
  - Whatever the $PS1 header field at +0x30 says (PSPReverse load_addr)

Then for each (blob, load_addr) pair where PC ∈ [load_addr, load_addr + size]:
  - Compute file offset = PC - load_addr + body_start
  - Disasm 20 insns at that offset (try ARM + Thumb)
"""
import struct
from pathlib import Path
from capstone import Cs, CS_ARCH_ARM, CS_MODE_ARM, CS_MODE_THUMB

OUT_DIR = Path("/home/pantelis/Desktop/Projects/Work/MLRift/captures")
FAULT_PC = 0x0d1102f4
md_thumb = Cs(CS_ARCH_ARM, CS_MODE_THUMB); md_thumb.skipdata = True
md_arm   = Cs(CS_ARCH_ARM, CS_MODE_ARM);   md_arm.skipdata = True

BLOBS = [
    ("sys_drv",  "sys_drv_subblob.bin"),
    ("sos",      "sos_subblob.bin"),
    ("soc_drv",  "soc_drv_subblob.bin"),
    ("intf_drv", "intf_drv_subblob.bin"),
    ("dbg_drv",  "dbg_drv_subblob.bin"),
    ("spl",      "spl_subblob.bin"),
]

CANDIDATE_LOAD_ADDRS = [
    0x0d100000, 0x0d110000, 0x0d000000,
    0x10000000, 0x20000000, 0x00000000,
    0x0c000000, 0x0e000000, 0x0d010000,
]

# Body start offset in PSP sub-blob is typically 0x1000 (16-byte sig
# + 0x100-byte $PS1 header + 0xef0 padding) — but varies. Try a few.
BODY_START_CANDIDATES = [0x100, 0x400, 0x1000]


def try_disasm(data, file_off, label):
    """Disasm at file_off in both ARM and Thumb mode, return text."""
    if file_off < 0 or file_off + 32 > len(data):
        return None
    chunk = data[file_off:file_off + 64]
    out = []
    for label_mode, md in [("Thumb", md_thumb), ("ARM", md_arm)]:
        out.append(f"  [{label_mode}] {chunk[:16].hex()}")
        insns = list(md.disasm(chunk, FAULT_PC))[:8]
        for ins in insns:
            out.append(f"    {ins.mnemonic:<10} {ins.op_str}")
    return "\n".join(out)


for name, fn in BLOBS:
    p = OUT_DIR / fn
    if not p.exists(): continue
    data = p.read_bytes()

    # Decode header load_addr (PSPReverse-style at +0x30 from $PS1)
    ps1 = data.find(b"$PS1")
    header_load_addr = None
    if ps1 >= 0:
        for off in (0x18, 0x30, 0x44):
            if ps1 + off + 4 <= len(data):
                v = int.from_bytes(data[ps1+off:ps1+off+4], "little")
                if 0x0c000000 <= v < 0x30000000:
                    header_load_addr = (off, v)
                    break

    print(f"\n=== {name} ({len(data)} bytes) ===")
    if header_load_addr:
        print(f"  $PS1 header at 0x{ps1:x}, candidate load_addr @+0x{header_load_addr[0]:x} = 0x{header_load_addr[1]:08x}")

    # Test each candidate load address
    for load_addr in CANDIDATE_LOAD_ADDRS:
        if not (load_addr <= FAULT_PC < load_addr + len(data)):
            continue
        # Compute virt offset
        virt_off = FAULT_PC - load_addr
        # For each body start candidate, compute file offset
        for body_start in BODY_START_CANDIDATES:
            file_off = body_start + virt_off
            if file_off + 16 > len(data):
                continue
            print(f"\n  Try load_addr=0x{load_addr:08x}, body_start=0x{body_start:x} → file_off=0x{file_off:x}")
            r = try_disasm(data, file_off, name)
            if r:
                # Quality filter: only print if at least one Thumb insn looks
                # plausible (not all `movs r0, r0` etc.)
                print(r)
