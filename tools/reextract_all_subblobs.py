#!/usr/bin/env python3
"""
λ.1 — Re-extract ALL 16 sub-blobs from psp_13_0_10_sos.bin with correct
$PS1-magic-based boundaries. Our prior extraction (extract_all_psp_subblobs.py)
used fw_type heuristics and produced 9 sub-blobs, one of which ("sos_subblob")
was actually 3 concatenated sub-blobs. This tool fixes that.

Outputs to captures/subblob_NN_<offset>.bin with NN = index in $PS1 scan order.
Also produces an entropy/structure report.
"""
import math
from pathlib import Path

OUTER = Path('/tmp/mlrift_fw/psp_13_0_10_sos.bin').read_bytes()
print(f"Outer file: {len(OUTER)} bytes\n")

# Find all $PS1 magics
ps1 = []
pos = 0
while True:
    f = OUTER.find(b'$PS1', pos)
    if f == -1: break
    ps1.append(f)
    pos = f + 1

print(f"Found {len(ps1)} $PS1 magics. Sub-blob starts (= $PS1 - 0x10):\n")

def entropy(data):
    if not data: return 0
    cnt = [0]*256
    for x in data: cnt[x] += 1
    n = len(data)
    return sum(-(c/n)*math.log2(c/n) for c in cnt if c > 0)

def chunk_entropies(data, chunk_size=0x1000):
    out = []
    for off in range(0, len(data), chunk_size):
        c = data[off:off+chunk_size]
        if len(c) < chunk_size//4: continue
        e = entropy(c)
        out.append((off, e))
    return out

# Build sub-blob boundary table
captures_dir = Path('/home/pantelis/Desktop/Projects/Work/MLRift/captures')
captures_dir.mkdir(exist_ok=True)

subblobs = []
for i, p in enumerate(ps1):
    start = p - 0x10
    end = (ps1[i+1] - 0x10) if i+1 < len(ps1) else len(OUTER)
    size = end - start
    data = OUTER[start:end]
    subblobs.append((i, start, end, size, data))

print(f"{'#':<3} {'start':>10} {'end':>10} {'size (B)':>10} {'size (K)':>9} {'overall ent':>12} {'tag':<30}")
print("-" * 100)
for i, start, end, size, data in subblobs:
    e = entropy(data)
    # Header field at +0x18 might tell us fw_type
    fw_type_at_18 = int.from_bytes(data[0x18:0x1c], 'little') if size > 0x1c else 0
    fw_type_at_40 = int.from_bytes(data[0x40:0x44], 'little') if size > 0x44 else 0
    # Common labels by size (rough heuristic from our prior extraction)
    tag = ""
    if size < 3000: tag = "small (maybe SPL-table fragment)"
    elif 3000 < size < 8000: tag = "small driver (KDB candidate)"
    elif 30000 < size < 40000: tag = "SOC_DRV candidate"
    elif 25000 < size < 30000: tag = "INTF_DRV candidate"
    elif 14000 < size < 18000: tag = "DBG_DRV candidate"
    elif 50000 < size < 70000: tag = "SYS_DRV candidate"
    elif size > 80000: tag = "BIG — likely main SOS kernel"
    print(f"{i:<3} 0x{start:08x} 0x{end:08x} {size:>10} {size/1024:>6.1f}K  {e:>10.3f}   {tag}")

# Write each to its own file
print(f"\nWriting to {captures_dir}/...")
for i, start, end, size, data in subblobs:
    fn = captures_dir / f"subblob_{i:02d}_at_0x{start:06x}_{size}B.bin"
    fn.write_bytes(data)

# Special focus: detailed entropy map of sub-blob #2 (the "big SOS")
big = subblobs[2]
print(f"\n{'='*80}")
print(f"DETAILED entropy map of sub-blob #2 (0x{big[1]:x}, {big[3]} B) — was our 'SOS'")
print(f"{'='*80}")
for off, e in chunk_entropies(big[4]):
    bar = "█" * int(e)
    note = ""
    if 0x7000 <= off < 0x15000: note = "  (was 'encrypted body' — actually zeros)"
    elif off >= 0x17000 and e > 5: note = "  ★ PLAINTEXT we previously MISSED ★"
    print(f"  +0x{off:05x}  ent={e:.2f}  {bar:<8}{note}")
