#!/usr/bin/env python3
"""
η.3q-5 — search ALL extracted sub-blobs for verbatim fault-related
constants, with 4-byte alignment ONLY (eliminates code-as-data noise).

Targets:
  - 0x83f00f80 — exact fault addr
  - 0x83f00000 — page-aligned form
  - 0x0d1102f4 — exact fault PC
  - 0x0d110000 — page-aligned form
  - 0x0d000000 — fault PC base
  - 0x83000000 / 0x84000000 — aperture bounds

If we find an aligned 4-byte literal for ANY of these in ANY plaintext
sub-blob, it's a hard signal — the kernel keeps that constant in a
literal pool, meaning code references it.

Also dumps all 4-byte-aligned references to 0x83x000xx / 0x83x000xx
patterns (potential virt addresses in the fault aperture).
"""
import struct
from pathlib import Path

OUT_DIR = Path("/home/pantelis/Desktop/Projects/Work/MLRift/captures")

TARGETS = [
    (0x83f00f80, "fault_addr_exact"),
    (0x83f00000, "fault_addr_page"),
    (0x83000000, "fault_aperture_base_low"),
    (0x84000000, "fault_aperture_base_high"),
    (0x0d1102f4, "fault_PC_exact"),
    (0x0d110000, "fault_PC_page"),
    (0x0d100000, "fault_PC_1MB"),
    (0x0d000000, "fault_PC_module_base"),
    (0x0e000000, "fault_PC_module_next"),
]

BLOBS = [
    ("sys_drv",  "sys_drv_subblob.bin"),
    ("sos",      "sos_subblob.bin"),
    ("kdb",      "kdb_subblob.bin"),
    ("soc_drv",  "soc_drv_subblob.bin"),
    ("intf_drv", "intf_drv_subblob.bin"),
    ("dbg_drv",  "dbg_drv_subblob.bin"),
    ("spl",      "spl_subblob.bin"),
]

def aligned_search(data, target):
    pat = struct.pack("<I", target & 0xFFFFFFFF)
    hits = []
    for o in range(0, len(data) - 3, 4):  # 4-byte aligned ONLY
        if data[o:o+4] == pat:
            hits.append(o)
    return hits


def aligned_range_search(data, lo, hi):
    """Find all 4-byte aligned u32 values in [lo, hi)."""
    out = {}
    for o in range(0, len(data) - 3, 4):
        v = struct.unpack_from("<I", data, o)[0]
        if lo <= v < hi:
            out.setdefault(v, []).append(o)
    return out


for name, fn in BLOBS:
    p = OUT_DIR / fn
    if not p.exists():
        continue
    data = p.read_bytes()
    print(f"=== {name} ({len(data)} bytes) ===")

    # Exact target search (aligned)
    any_hit = False
    for tgt, label in TARGETS:
        hits = aligned_search(data, tgt)
        if hits:
            any_hit = True
            print(f"  {label} (0x{tgt:08x}): {len(hits)}× aligned hits @ {[hex(h) for h in hits[:10]]}")
    if not any_hit:
        print(f"  (none of the fault constants found at 4-byte alignment)")

    # Range search: any 0x83xxxxxx aligned values
    in_aperture = aligned_range_search(data, 0x83000000, 0x84000000)
    if in_aperture:
        # Filter out junk: only show values where many bits are non-trivial
        meaningful = {v: offs for v, offs in in_aperture.items()
                      if v != 0x83000000 or len(offs) == 1}
        if meaningful:
            print(f"  Constants in 0x83xxxxxx range (aligned): {len(meaningful)} unique")
            for v, offs in sorted(meaningful.items())[:20]:
                marker = ""
                if v == 0x83f00f80: marker = "  <<< EXACT FAULT ADDR"
                elif 0x83f00000 <= v < 0x83f01000: marker = "  <<< 4KB of fault"
                print(f"    0x{v:08x}  ×{len(offs)}  @{[hex(o) for o in offs[:4]]}{marker}")

    # Range search: 0x0d_xx_xx_xx aligned values
    in_pc_range = aligned_range_search(data, 0x0d000000, 0x0e000000)
    if in_pc_range:
        print(f"  Constants in 0x0d_xxxxxx range (aligned): {len(in_pc_range)} unique")
        for v, offs in sorted(in_pc_range.items())[:20]:
            marker = ""
            if v == 0x0d1102f4: marker = "  <<< EXACT FAULT PC"
            elif 0x0d110000 <= v < 0x0d111000: marker = "  <<< 4KB of fault PC"
            print(f"    0x{v:08x}  ×{len(offs)}  @{[hex(o) for o in offs[:4]]}{marker}")

    print()
