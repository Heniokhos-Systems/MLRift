#!/usr/bin/env python3
"""
θ.3-E — find the ARMv7 PSP page table inside the SOS sub-blob.

We learned in θ.3-D that:
  - SOS Reset handler copies 125-256 KiB of itself from virt 0x0d110100
    (SOS body file offset 0x100) to PSP SRAM 0.
  - TTBR0 ends up at the COPIED PT location.
  - The PT therefore lives inside the SOS firmware blob (in the
    plaintext region between file 0x100 and 0x6D00 OR inside the
    encrypted body 0x7000-0x14fff).

ARMv7 first-level PT structure:
  - 16 KiB, 4096 entries, 4 bytes each
  - Each entry maps a 1 MiB section (or fault/coarse/PXN variants)
  - Section descriptor (most common): bits[1:0] = 0b10 (=0x02)
    bits[31:20] = physical section base
    bits[19:12] = AP/TEX/C/B/Domain/etc. flags
  - Page table descriptor: bits[1:0] = 0b01
  - Fault: bits[1:0] = 0b00

Search strategy:
  - Scan SOS body in 16-KiB-aligned 16-KiB chunks
  - For each chunk, count entries that look like ARMv7 section
    descriptors (bottom byte has bits 1:0 = 0b10 in the first byte)
  - A real PT would have lots of "fault" entries (zeros) and some
    "section" entries — typical pattern: blocks of zeros with section
    entries at specific indices.
  - Particularly interesting: PT[0x83f] = entry that maps virt 0x83f0_0000
    section. Compute byte offset (0x83f * 4 = 0x20FC inside PT).

Print:
  - For each candidate PT location, the entry at PT[0x83f]
  - Layout of all non-zero entries
"""
import struct
from pathlib import Path

SOS = Path("/home/pantelis/Desktop/Projects/Work/MLRift/captures/sos_subblob.bin").read_bytes()

print(f"SOS sub-blob: {len(SOS)} bytes\n")

# For each 4-byte aligned starting offset, look at 16 KiB worth of data
# (4096 32-bit entries) and check if it looks like an ARMv7 PT.

# Heuristic for PT-ness:
#   - At least 50 entries with bits[1:0] = 0b10 (section descriptor)
#   - At most 4000 entries are zero (fault)
#   - Multiple distinct section physical bases

def evaluate_pt_candidate(off):
    if off + 16384 > len(SOS): return None
    entries = []
    for i in range(4096):
        v = int.from_bytes(SOS[off + i*4: off + i*4 + 4], "little")
        entries.append(v)
    section_count = sum(1 for v in entries if (v & 3) == 2 and v != 0)
    pagetable_count = sum(1 for v in entries if (v & 3) == 1)
    fault_count = sum(1 for v in entries if v == 0)
    nonzero_nonsection = sum(1 for v in entries if v != 0 and (v & 3) not in (1, 2))
    unique_section_bases = set()
    for v in entries:
        if (v & 3) == 2:
            unique_section_bases.add(v >> 20)
    return {
        "section_count": section_count,
        "pagetable_count": pagetable_count,
        "fault_count": fault_count,
        "nonzero_nonsection": nonzero_nonsection,
        "unique_sec_bases": len(unique_section_bases),
        "entry_83f": entries[0x83f] if 0x83f < len(entries) else None,
        "entry_d11": entries[0xd11] if 0xd11 < len(entries) else None,
        "entry_000": entries[0x000] if 0x000 < len(entries) else None,
    }


# Scan 16-KiB-aligned offsets within the SOS body
candidates = []
for off in range(0, len(SOS) - 16384, 0x1000):  # 4-KiB stride
    r = evaluate_pt_candidate(off)
    if r is None: continue
    # Score: high section_count + reasonable fault_count + few junk
    score = r["section_count"] - r["nonzero_nonsection"] * 2
    if r["section_count"] >= 10 and r["nonzero_nonsection"] < r["section_count"]:
        candidates.append((off, r, score))

candidates.sort(key=lambda x: -x[2])
print(f"Top PT candidate locations (sorted by section_count - junk * 2):\n")
for off, r, score in candidates[:10]:
    print(f"  offset 0x{off:x}  score={score}  "
          f"sections={r['section_count']}  faults={r['fault_count']}  "
          f"junk={r['nonzero_nonsection']}  unique_bases={r['unique_sec_bases']}")
    if r['entry_83f'] is not None:
        marker = "  <<< 0x83f section!" if r['entry_83f'] != 0 else ""
        print(f"    PT[0x83f] = 0x{r['entry_83f']:08x}{marker}")
    if r['entry_d11'] is not None:
        print(f"    PT[0xd11] = 0x{r['entry_d11']:08x}  (SOS virt base section)")
    if r['entry_000'] is not None:
        print(f"    PT[0x000] = 0x{r['entry_000']:08x}  (virt 0 section)")

# For the top candidate, dump ALL non-zero entries
if candidates:
    top_off, top_r, _ = candidates[0]
    print(f"\n\n{'='*70}")
    print(f"Top candidate @ 0x{top_off:x} — all non-zero entries:")
    print(f"{'='*70}")
    for i in range(4096):
        v = int.from_bytes(SOS[top_off + i*4: top_off + i*4 + 4], "little")
        if v != 0:
            virt_section = i << 20
            phys_section = (v >> 20) << 20
            etype = "?"
            if (v & 3) == 2: etype = "section"
            elif (v & 3) == 1: etype = "coarse_pt_ref"
            elif (v & 3) == 0: etype = "fault"
            flag = ""
            if i == 0x83f:
                flag = "  *** virt 0x83f00000 ***"
            elif i == 0xd11:
                flag = "  *** SOS virt base ***"
            print(f"  PT[0x{i:03x}] = 0x{v:08x}  virt 0x{virt_section:08x} → "
                  f"phys 0x{phys_section:08x}  type={etype}{flag}")
