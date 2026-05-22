#!/usr/bin/env python3
"""
λ.2 — Comprehensive plaintext search across ALL 16 sub-blobs.

Goals:
  1. For sub-blob #2 (the big one), decode the previously-missed region
     at offset 0x15000+ (we cut off at 0x19180 before).
  2. For ALL 16 sub-blobs, search for:
     - fault constants: 0x83f00f80, 0x83f00000, 0x0d1102f4, 0x0d110000
     - MP0 SMN C2PMSG constants: 0x032001f0/f4/f8 (C2PMSG_60/61/62)
     - SOS state-related: 0x80320d17, 0x016ed035, 0x32, 0x0d17
     - PT-region addresses: 0xc000-0xd000
  3. Identify which sub-blob (if any) contains the C2PMSG_60/61/62
     write code we couldn't find in the previously-known plaintext.
"""
import struct
import math
from pathlib import Path
from capstone import Cs, CS_ARCH_ARM, CS_MODE_THUMB, CS_MODE_ARM

CAP_DIR = Path('/home/pantelis/Desktop/Projects/Work/MLRift/captures')
md_thumb = Cs(CS_ARCH_ARM, CS_MODE_THUMB); md_thumb.skipdata = True
md_arm   = Cs(CS_ARCH_ARM, CS_MODE_ARM);   md_arm.skipdata = True

# Find all the subblob files we just extracted
subblob_files = sorted([f for f in CAP_DIR.glob("subblob_*.bin")])
print(f"Found {len(subblob_files)} sub-blob files\n")

# Fault constants to search
TARGETS = [
    (0x83f00f80, "fault_addr_EXACT"),
    (0x83f00000, "fault_addr_page"),
    (0x83000000, "fault_aperture_base"),
    (0x0d1102f4, "fault_PC_EXACT"),
    (0x0d110000, "SOS_virt_base"),
    (0x80320d17, "DEBUG_STATUS observed"),
    (0x016ed035, "warm SoL value"),
    # MP0 C2PMSG SMN constants (PSP-internal addr)
    (0x032001f0, "MP0_SMN_C2PMSG_60_addr"),
    (0x032001f4, "MP0_SMN_C2PMSG_61_addr"),
    (0x032001f8, "MP0_SMN_C2PMSG_62_addr"),
    (0x03200000, "MP0_SMN_base"),
    (0x032000d8, "PSP_DEBUG_STATUS_REG"),
    # PT region
    (0x0000c000, "PT base"),
    (0x0000c048, "PT TTBR0 value"),
]

# Aligned 4-byte search
def aligned_search(data, target):
    pat = struct.pack("<I", target & 0xFFFFFFFF)
    return [o for o in range(0, len(data) - 3, 4) if data[o:o+4] == pat]

print("=" * 90)
print("PHASE 1: Search ALL sub-blobs for fault constants + C2PMSG addresses")
print("=" * 90)
any_hit = False
for sf in subblob_files:
    data = sf.read_bytes()
    hits_in_blob = []
    for tgt, lbl in TARGETS:
        h = aligned_search(data, tgt)
        if h:
            hits_in_blob.append((tgt, lbl, h))
            any_hit = True
    if hits_in_blob:
        print(f"\n{sf.name}: {len(data)} B")
        for tgt, lbl, hits in hits_in_blob:
            print(f"  {lbl} (0x{tgt:08x}): {len(hits)}× @ {[hex(h) for h in hits[:5]]}")
if not any_hit:
    print("  NO HITS across any sub-blob for any target.")

# Range searches
print("\n" + "=" * 90)
print("PHASE 2: Range search for any 0x83Fxxxxx constant (4-byte aligned)")
print("=" * 90)
for sf in subblob_files:
    data = sf.read_bytes()
    hits = {}
    for o in range(0, len(data) - 3, 4):
        v = int.from_bytes(data[o:o+4], 'little')
        if 0x83f00000 <= v < 0x84000000:
            hits.setdefault(v, []).append(o)
    if hits:
        print(f"\n{sf.name}: {len(hits)} unique 0x83Fxxxxx values")
        for v, offs in sorted(hits.items())[:10]:
            tag = "  ★ EXACT FAULT ADDR" if v == 0x83f00f80 else ""
            print(f"  0x{v:08x}  ×{len(offs)}  @{[hex(o) for o in offs[:3]]}{tag}")

# Search for any 0x03200xxx constant (MP0 SMN range)
print("\n" + "=" * 90)
print("PHASE 3: Range search for 0x032000xx-0x032003xx (MP0 SMN range)")
print("=" * 90)
for sf in subblob_files:
    data = sf.read_bytes()
    hits = {}
    for o in range(0, len(data) - 3, 4):
        v = int.from_bytes(data[o:o+4], 'little')
        if 0x03200000 <= v < 0x03200400:
            hits.setdefault(v, []).append(o)
    if hits:
        print(f"\n{sf.name}: {len(hits)} unique 0x03200xxx values")
        for v, offs in sorted(hits.items()):
            byte_off = v - 0x03200000
            dword = byte_off // 4
            c2_num = dword - 0x40 if 0x40 <= dword <= 0xff else None
            tag = f"  = C2PMSG_{c2_num}" if c2_num is not None else ""
            note = ""
            if c2_num == 60: note = "  ★★★ EXCEPTION TYPE WRITE ADDR ★★★"
            elif c2_num == 61: note = "  ★★★ EXCEPTION ADDR WRITE ADDR ★★★"
            elif c2_num == 62: note = "  ★★★ EXCEPTION PC WRITE ADDR ★★★"
            elif v == 0x032000d8: note = "  (DEBUG_STATUS — known used)"
            print(f"  0x{v:08x}  ×{len(offs)}  @{[hex(o) for o in offs[:3]]}{tag}{note}")

# Disasm sub-blob #2's MISSED region (offset 0x15000 onwards)
print("\n" + "=" * 90)
print("PHASE 4: Disasm sub-blob #2's MISSED plaintext region (0x15000+)")
print("=" * 90)
sb2_file = [f for f in subblob_files if 'subblob_02_' in f.name]
if sb2_file:
    sb2 = sb2_file[0].read_bytes()
    print(f"  Sub-blob #2: {len(sb2)} B; missed region 0x15000+")
    missed_start = 0x15000
    # End of plaintext region — find where entropy drops to zero again
    chunk_size = 0x1000
    end = missed_start
    for off in range(missed_start, len(sb2), chunk_size):
        chunk = sb2[off:off+chunk_size]
        if len(chunk) < 256: break
        cnt = [0]*256
        for x in chunk: cnt[x] += 1
        n = len(chunk)
        e = sum(-(c/n)*math.log2(c/n) for c in cnt if c > 0)
        if e < 3: break  # padding
        end = off + chunk_size
    print(f"  Plaintext extent: 0x{missed_start:x} - 0x{end:x} ({end - missed_start} bytes)")

    # First disasm as Thumb (assume same as R1)
    print(f"\n  First 30 Thumb insns starting at 0x{missed_start:x}:")
    insns = list(md_thumb.disasm(sb2[missed_start:end], missed_start))[:30]
    for ins in insns:
        tag = ""
        if "[pc," in ins.op_str:
            try:
                imm = int(ins.op_str.split("[pc,")[1].split("]")[0].lstrip("#"), 0)
                lv_a = ((ins.address + 4) & ~3) + imm
                if lv_a + 4 <= len(sb2):
                    lv = int.from_bytes(sb2[lv_a:lv_a+4], 'little')
                    note = ""
                    if 0x83000000 <= lv < 0x84000000: note = "  ← PSP virt"
                    elif 0x03200000 <= lv < 0x03200400: note = "  ← MP0 SMN"
                    elif 0x03010000 <= lv < 0x03020000: note = "  ← MP1 SMN"
                    elif lv < 0x40000 and lv > 0x100: note = "  ← PSP SRAM"
                    tag = f"  /* 0x{lv:08x}{note} */"
            except: pass
        print(f"  0x{ins.address:08x}: {ins.bytes.hex():<10} {ins.mnemonic:<10} {ins.op_str}{tag}")

    # Find all function starts in the missed region
    fn_starts = []
    for ins in md_thumb.disasm(sb2[missed_start:end], missed_start):
        if ins.mnemonic in ("push", "push.w") and "lr" in ins.op_str:
            fn_starts.append(ins.address)
    print(f"\n  Function starts (push with lr) in 0x{missed_start:x}-0x{end:x}: {len(fn_starts)}")
    if fn_starts:
        print(f"    First 10: {[hex(s) for s in fn_starts[:10]]}")
        print(f"    Last 5:   {[hex(s) for s in fn_starts[-5:]]}")
