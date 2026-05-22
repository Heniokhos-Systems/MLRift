#!/usr/bin/env python3
"""
η.3j-RE-3-cont — search SOS plaintext regions for any reference to
the fault address 0x83f00f80, and disassemble the post-MMU jump target
at virt 0x1c8.

If 0x83f00f80 appears verbatim in any plaintext region, that's the
direct page-table input we need to investigate. If it appears as
0x83000000 + 0xf00f80 or similar split form, also useful.

Also search for related landmarks:
  - 0x83f00000 (page-aligned form)
  - 0x83000000, 0x84000000 (256 MiB-aligned neighbors)
  - 0xf00f80 (low part)
  - 0x83000000-0x84000000 ranges as immediates
"""

import struct
from pathlib import Path
from capstone import Cs, CS_ARCH_ARM, CS_MODE_ARM, CS_MODE_THUMB

BLOB = Path("/home/pantelis/Desktop/Projects/Work/MLRift/captures/sos_subblob.bin")
data = BLOB.read_bytes()
N = len(data)
print(f"SOS sub-blob: {N} bytes (0x{N:x})\n")

# 1. Search for verbatim 0x83f00f80 in file (any alignment).
targets = [
    (0x83f00f80, "fault_addr_exact"),
    (0x83f00000, "fault_page_aligned"),
    (0x83000000, "256MB_below"),
    (0x84000000, "256MB_above"),
    (0xf00f80,   "low_part_only"),
    (0x83f0,     "high_part_u16"),
]

print("=== verbatim search for fault-related constants ===")
for tgt, label in targets:
    # search as u32 little-endian
    pat = struct.pack("<I", tgt & 0xFFFFFFFF)
    hits = []
    off = 0
    while True:
        off = data.find(pat, off)
        if off == -1: break
        hits.append(off)
        off += 1
    if hits:
        print(f"  {label} (0x{tgt:x}) found {len(hits)}× at offsets:")
        for h in hits[:20]:
            # Show context: 16 bytes around it
            ctx_start = max(0, h - 4)
            ctx_end   = min(N, h + 8)
            chunk = data[ctx_start:ctx_end]
            hex_str = ' '.join(f'{b:02x}' for b in chunk)
            # Mark which region this offset is in
            region = "encrypted" if 0x7000 <= h < 0x15000 else "plaintext"
            print(f"    @0x{h:06x}  [{region}]  ctx: {hex_str}")
    else:
        print(f"  {label} (0x{tgt:x}): not found")

# 2. Disassemble around virt 0x1c8 in ARM mode (post-MMU branch target).
print("\n=== ARM disasm at virt 0x1c8 (post-MMU branch target from reset handler) ===")
md_arm = Cs(CS_ARCH_ARM, CS_MODE_ARM)

# The MMU is enabled at 0x2C0 with BX ip → ip=0x1c8 (virtual).
# Assuming identity virt=phys mapping (which we know is wrong post-MMU
# but might be partially correct in low memory), 0x1c8 = file offset 0x1c8.
# Let's disasm 64 insns from there.
off = 0x1c8
chunk = data[off:off + 256]
for ins in md_arm.disasm(chunk, off):
    flag = ""
    m = ins.mnemonic
    op = ins.op_str
    if m.startswith("mcr") or m.startswith("mrc"):
        flag = "  CP15"
    elif m in ("bx", "blx"):
        flag = "  BRANCH"
    elif m == "ldr" and "[pc" in op:
        try:
            imm = int(op.split("#")[1].rstrip("]"), 0)
            la = ins.address + 8 + imm
            if 0 <= la < N - 4:
                v = struct.unpack_from("<I", data, la)[0]
                flag = f"  -> *0x{la:x} = 0x{v:08x}"
        except: pass
    print(f"  0x{ins.address:08x}: {ins.bytes.hex():<10} {m:<8} {op:<35}{flag}")

# 3. Look at the very last bytes of plaintext (0x6CE0-0x7000) — the
# transition into the encrypted region. If there's a JUMP to encrypted
# code, we can see where it goes.
print("\n=== Plaintext→encrypted boundary disasm (Thumb, last 0x40 bytes of R1) ===")
md_thumb = Cs(CS_ARCH_ARM, CS_MODE_THUMB)
off = 0x6CC0
chunk = data[off:off + 0x40]
for ins in md_thumb.disasm(chunk, off):
    m = ins.mnemonic
    op = ins.op_str
    flag = ""
    if m in ("bx", "blx", "bl", "b"):
        flag = "  BRANCH"
    elif m in ("ldr", "ldr.w") and "[pc" in op:
        try:
            imm = int(op.split("#")[1].rstrip("]"), 0)
            la = (ins.address + 4 + imm) & ~3
            if 0 <= la < N - 4:
                v = struct.unpack_from("<I", data, la)[0]
                flag = f"  -> *0x{la:x} = 0x{v:08x}"
        except: pass
    print(f"  0x{ins.address:08x}: {ins.bytes.hex():<10} {m:<8} {op:<35}{flag}")

# 4. R2 start (post-decrypt code, 0x15140) — what is the very first thing
# there? Just to characterize the entry point.
print("\n=== R2 start disasm (Thumb, 0x15140 first 0x40 bytes) ===")
off = 0x15140
chunk = data[off:off + 0x40]
for ins in md_thumb.disasm(chunk, off):
    m = ins.mnemonic
    op = ins.op_str
    flag = ""
    if m in ("bx", "blx", "bl", "b"):
        flag = "  BRANCH"
    elif m == "svc":
        flag = "  SVC (kernel call)"
    elif m in ("ldr", "ldr.w") and "[pc" in op:
        try:
            imm = int(op.split("#")[1].rstrip("]"), 0)
            la = (ins.address + 4 + imm) & ~3
            if 0 <= la < N - 4:
                v = struct.unpack_from("<I", data, la)[0]
                flag = f"  -> *0x{la:x} = 0x{v:08x}"
        except: pass
    print(f"  0x{ins.address:08x}: {ins.bytes.hex():<10} {m:<8} {op:<35}{flag}")

# 5. Search for any 32-bit constant in 0x80000000-0x90000000 range
# (the high-aperture range where 0x83f00f80 sits).
print("\n=== All 0x8x_xx_xx_xx (high-mem-mapped) constants in plaintext ===")
plaintext_regions = [(0, 0x7000), (0x15000, min(0x19200, N))]
hits = {}
for lo, hi in plaintext_regions:
    for o in range(lo, hi - 3):
        v = struct.unpack_from("<I", data, o)[0]
        if 0x80000000 <= v < 0x90000000:
            hits.setdefault(v, []).append(o)

if hits:
    by_count = sorted(hits.items(), key=lambda x: -len(x[1]))
    for v, offs in by_count[:30]:
        region_marks = ["plain" if o < 0x7000 else "tail" for o in offs[:3]]
        print(f"  0x{v:08x}  ×{len(offs)}  e.g. @0x{offs[0]:x} ({region_marks[0]})")
else:
    print("  (no 0x8x range constants in plaintext)")
