#!/usr/bin/env python3
"""
Probe captures/sos_subblob.bin to find real code regions.

We disassemble each 2-byte aligned offset for ~16 insns and score:
  - sensible mix of mnemonics
  - presence of typical prologue patterns (push, sub sp, etc.)
  - non-zero unique opcode count

Goal: enumerate the actual function entry points in the blob so we
target real Thumb-2 code, not data tables.
"""

import struct
from collections import Counter
from pathlib import Path
from capstone import Cs, CS_ARCH_ARM, CS_MODE_ARM, CS_MODE_THUMB

BLOB = Path("/home/pantelis/Desktop/Projects/Work/MLRift/captures/sos_subblob.bin")
data = BLOB.read_bytes()
SIZE = len(data)

md_arm   = Cs(CS_ARCH_ARM, CS_MODE_ARM)
md_thumb = Cs(CS_ARCH_ARM, CS_MODE_THUMB)


# --------------------------------------------------------------------
# 1. Histogram of zero/non-zero halfwords across the file — locate
#    text vs data segments quickly.
# --------------------------------------------------------------------
print(f"loaded {BLOB} ({SIZE} bytes)\n")
print("=== File map: 4 KiB-page non-zero halfword density ===")
PAGE = 0x1000
for p in range(0, SIZE, PAGE):
    chunk = data[p:p + PAGE]
    nonzero_hw = sum(1 for i in range(0, len(chunk) - 1, 2)
                     if chunk[i] | chunk[i + 1])
    total_hw = len(chunk) // 2
    density = nonzero_hw / total_hw if total_hw else 0
    bar = "#" * int(density * 40)
    print(f"  0x{p:05x}..0x{p+PAGE-1:05x}  {nonzero_hw:4d}/{total_hw}  {bar}")
print()


# --------------------------------------------------------------------
# 2. Find ARM-mode `LDR Rd, =literal | 1` constants — Thumb entry
#    points referenced from ARM code or pointer tables.
# --------------------------------------------------------------------
print("=== Thumb function entry candidates (literal pool addrs with bit-0 set) ===")
candidates = set()
for off in range(0, SIZE - 4, 4):
    w = struct.unpack_from("<I", data, off)[0]
    if w & 1 and (w & ~1) < SIZE and (w & ~1) >= 0x100:
        candidates.add(w & ~1)
print(f"  found {len(candidates)} distinct Thumb entry candidates")

# Verify candidates by disasm: try Thumb decode of 16 insns; require
# at least 12 successful and a reasonable opcode mix.
def score_thumb(off: int) -> tuple[int, Counter]:
    if off >= SIZE - 32:
        return (0, Counter())
    chunk = data[off:off + 64]
    mns = Counter()
    n = 0
    last = off
    for ins in md_thumb.disasm(chunk, off):
        mns[ins.mnemonic] += 1
        n += 1
        last = ins.address + ins.size
        if n >= 16:
            break
    coverage = last - off
    return (coverage, mns)

good_entries = []
for c in sorted(candidates):
    cov, mns = score_thumb(c)
    if cov >= 24 and len(mns) >= 6:
        # likely real code (good coverage, varied opcodes)
        good_entries.append((c, cov, mns))

print(f"  {len(good_entries)} pass score>=24 + opcode_variety>=6")
for c, cov, mns in good_entries[:30]:
    top3 = ", ".join(f"{m}={n}" for m, n in mns.most_common(3))
    print(f"    0x{c:06x}  cov={cov}  top3=[{top3}]")
print()


# --------------------------------------------------------------------
# 3. Brute Thumb-scan: at every 2-byte alignment, count consecutive
#    successful decodes — find longest code runs.
# --------------------------------------------------------------------
print("=== Longest Thumb-2 decode runs (start, run_bytes) ===")
runs: list[tuple[int, int]] = []
off = 0x300  # past the ARM vector table + reset handler
while off < SIZE - 32:
    chunk = data[off:off + 128]
    cov = 0
    for ins in md_thumb.disasm(chunk, off):
        cov = (ins.address + ins.size) - off
    if cov >= 32:
        # find true end by continuing
        cur = off + cov
        while cur < SIZE - 16:
            ch = data[cur:cur + 64]
            grew = 0
            for ins in md_thumb.disasm(ch, cur):
                grew = (ins.address + ins.size) - cur
            if grew < 8:
                break
            cur += grew
        runs.append((off, cur - off))
        off = cur
    else:
        off += 2
runs.sort(key=lambda r: -r[1])
print(f"  total runs >=32 bytes: {len(runs)}")
for start, n in runs[:25]:
    print(f"    0x{start:06x}..0x{start+n:06x}  ({n} bytes)")
print()


# --------------------------------------------------------------------
# 4. Brute ARM-scan equivalent
# --------------------------------------------------------------------
print("=== Longest ARM-mode decode runs ===")
arm_runs = []
off = 0x100
while off < SIZE - 32:
    if off & 3:
        off += 1
        continue
    chunk = data[off:off + 256]
    cov = 0
    for ins in md_arm.disasm(chunk, off):
        cov = (ins.address + ins.size) - off
    if cov >= 32:
        cur = off + cov
        while cur < SIZE - 16:
            ch = data[cur:cur + 256]
            grew = 0
            for ins in md_arm.disasm(ch, cur):
                grew = (ins.address + ins.size) - cur
            if grew < 8:
                break
            cur += grew
        arm_runs.append((off, cur - off))
        off = cur
    else:
        off += 4
arm_runs.sort(key=lambda r: -r[1])
print(f"  total runs >=32 bytes: {len(arm_runs)}")
for start, n in arm_runs[:25]:
    print(f"    0x{start:06x}..0x{start+n:06x}  ({n} bytes)")
