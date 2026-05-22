#!/usr/bin/env python3
"""
η.3q-2 — characterize SYS_DRV sub-blob (most likely the PSP kernel).
SYS_DRV is 62 KB, mostly plaintext (entropy avg 164/256 with only 1
of 61 high-entropy chunks). This is the prime candidate for the kernel
that owns the SVC dispatcher + ARM MMU page-table construction.

Tasks:
  1. Look for $PS1 header at +0x10 and read fw size/version
  2. Map plaintext vs encrypted regions (entropy per 1 KB)
  3. Find ARM exception vector table (8 branch instructions at offset 0)
  4. Find SVC handler entry point (vector_table[8] should branch to it)
  5. Look for any reference to address 0x83f00f80 or 0x83f00000 or 0x83000000
  6. Look for code that writes to PSP SRAM 0xc000-0xd000 (page table region)
  7. Hex-dump 0x100-0x300 to see code-vs-data distribution
"""
import struct
from pathlib import Path
from capstone import Cs, CS_ARCH_ARM, CS_MODE_ARM, CS_MODE_THUMB

BLOB = Path("/home/pantelis/Desktop/Projects/Work/MLRift/captures/sys_drv_subblob.bin")
data = BLOB.read_bytes()
N = len(data)
print(f"SYS_DRV: {N} bytes (0x{N:x})\n")

# 1. PSP header check
ps1_off = data.find(b"$PS1")
print(f"=== PSP header ===")
print(f"$PS1 magic at offset 0x{ps1_off:x}")
if ps1_off == 0x10:
    # standard layout: 16-byte sig before header
    body_size = struct.unpack_from("<I", data, ps1_off + 4)[0]
    print(f"  body_size from header: 0x{body_size:x} ({body_size} bytes)")

# 2. Entropy per 1 KB chunk
print(f"\n=== Entropy per 1 KB (low = plaintext, high = encrypted) ===")
print(f"  Offset  | Unique bytes / 1 KB")
for i in range(0, N, 1024):
    chunk = data[i:i+1024]
    if len(chunk) < 16: continue
    u = len(set(chunk))
    bar = "#" * (u // 8)
    flag = "  <ENCRYPTED?" if u > 200 else ""
    print(f"  0x{i:05x} | {u:3d}/256 {bar}{flag}")

# 3. ARM exception vectors — typical layout (8 branches × 4 bytes = 0x20 bytes)
# Often at start of blob, but may follow the 0x100-byte header
print(f"\n=== Looking for ARM vector tables ===")
md_arm = Cs(CS_ARCH_ARM, CS_MODE_ARM)
for try_off in (0x0, 0x100, 0x200, 0x300, 0x400):
    # First 4 bytes should be a B instruction (branch) in ARM mode
    if try_off + 32 > N: continue
    chunk = data[try_off:try_off + 32]
    decoded = list(md_arm.disasm(chunk, try_off))
    branches = sum(1 for ins in decoded if ins.mnemonic.startswith("b"))
    if branches >= 5:
        print(f"\n  Likely vector table @ 0x{try_off:x}:")
        for ins in decoded[:8]:
            print(f"    0x{ins.address:08x}: {ins.bytes.hex():<10} {ins.mnemonic:<6} {ins.op_str}")
        break

# 4. Search for 0x83f00f80 / 0x83000000 / 0x83f00000 / 0x84000000 as 4-byte LE
print(f"\n=== Verbatim search for fault-address constants ===")
for tgt, label in [
    (0x83f00f80, "fault_exact"),
    (0x83f00000, "fault_page"),
    (0x83000000, "0x83 base"),
    (0x84000000, "0x84 base"),
    (0x0d1102f4, "fault_PC"),
    (0x0d110000, "fault_PC_page"),
    (0x0d000000, "0x0d base"),
]:
    pat = struct.pack("<I", tgt & 0xFFFFFFFF)
    locs = []
    o = 0
    while True:
        o = data.find(pat, o)
        if o == -1: break
        locs.append(o)
        o += 1
    if locs:
        print(f"  {label} (0x{tgt:08x}) found {len(locs)}× at: {[hex(l) for l in locs[:10]]}")
    else:
        print(f"  {label} (0x{tgt:08x}): not found")

# 5. Look for any 32-bit constants in the 0x83xxxxxx and 0x0d11xxxx ranges
print(f"\n=== Constants in 0x83000000-0x84000000 range (fault aperture) ===")
hits_83 = {}
for o in range(0, N - 3, 1):  # byte-aligned search to catch unaligned consts
    v = struct.unpack_from("<I", data, o)[0]
    if 0x83000000 <= v < 0x84000000:
        hits_83.setdefault(v, []).append(o)

if hits_83:
    for v, offs in sorted(hits_83.items()):
        marker = ""
        if v == 0x83f00f80: marker = "  <<< EXACT FAULT ADDR"
        elif 0x83f00000 <= v < 0x83f01000: marker = "  <<< within 4KB of fault"
        print(f"  0x{v:08x}  ×{len(offs)}  @{[hex(o) for o in offs[:5]]}{marker}")
else:
    print(f"  No constants found in 0x83000000-0x84000000 range")

print(f"\n=== Constants in 0x0d000000-0x0e000000 range (fault PC region) ===")
hits_0d = {}
for o in range(0, N - 3, 1):
    v = struct.unpack_from("<I", data, o)[0]
    if 0x0d000000 <= v < 0x0e000000:
        hits_0d.setdefault(v, []).append(o)

if hits_0d:
    for v, offs in sorted(hits_0d.items())[:30]:
        marker = ""
        if v == 0x0d1102f4: marker = "  <<< EXACT FAULT PC"
        elif 0x0d110000 <= v < 0x0d111000: marker = "  <<< within 4KB of fault PC"
        print(f"  0x{v:08x}  ×{len(offs)}  @{[hex(o) for o in offs[:3]]}{marker}")
else:
    print(f"  No constants found in 0x0d000000-0x0e000000 range")

# 6. Look for stores to PSP SRAM 0xc000-0xd000 — page table region
print(f"\n=== Constants pointing into 0xc000..0xd000 (page-table region) ===")
hits_pt = {}
for o in range(0, N - 3, 1):
    v = struct.unpack_from("<I", data, o)[0]
    if 0xc000 <= v < 0xd000:
        hits_pt.setdefault(v, []).append(o)

if hits_pt:
    for v, offs in sorted(hits_pt.items())[:20]:
        print(f"  0x{v:08x}  ×{len(offs)}  @{[hex(o) for o in offs[:3]]}")
else:
    print(f"  No page-table-region constants found")
