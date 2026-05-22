#!/usr/bin/env python3
"""
λ.3 — Find code that stores to [base, #0x1f0/0x1f4/0x1f8/0xd8] across ALL sub-blobs.

If the C2PMSG_60/61/62 writer uses `mov base, #0x3200000; str rN, [base, #0x1fX]`
pattern (same as DEBUG_STATUS helper @ 0x1d50), then SMN constants won't appear
as 4-byte-aligned literals — we have to find the store instructions.

Target offsets:
  +0xd8  → MP0 SMN 0x032000d8 (DEBUG_STATUS — known used by helper @ 0x1d50)
  +0x1f0 → MP0 SMN 0x032001f0 (C2PMSG_60 — exception type)
  +0x1f4 → MP0 SMN 0x032001f4 (C2PMSG_61 — exception addr)
  +0x1f8 → MP0 SMN 0x032001f8 (C2PMSG_62 — exception PC)
  +0x244 → MP0 SMN 0x03200244 (C2PMSG_81 — SoL)
  +0x270 → MP0 SMN 0x03200270 (C2PMSG_92 — "BOOT_STATUS")

Strategy: disasm each sub-blob (both Thumb-2 and ARM modes), look for store
ops with these specific offsets. Print containing function + context.

ALSO: look for the `mov.w rN, #0x3200000` (constant load of PSP base) sites
across all sub-blobs — every function that touches MP0 must load this first.
"""
import re
from pathlib import Path
from capstone import Cs, CS_ARCH_ARM, CS_MODE_THUMB, CS_MODE_ARM

CAP_DIR = Path('/home/pantelis/Desktop/Projects/Work/MLRift/captures')
md_thumb = Cs(CS_ARCH_ARM, CS_MODE_THUMB); md_thumb.skipdata = True
md_arm   = Cs(CS_ARCH_ARM, CS_MODE_ARM);   md_arm.skipdata = True

TARGET_OFFSETS = {
    0xd8:  "C2PMSG_54 (PSP_DEBUG_STATUS)",  # actually 0x32000d8
    0x180: "C2PMSG_32",
    0x18c: "C2PMSG_35 (BL_CMD)",
    0x190: "C2PMSG_36 (BL_BUF)",
    0x1f0: "C2PMSG_60 (EXCEPTION TYPE!)",
    0x1f4: "C2PMSG_61 (EXCEPTION ADDR!)",
    0x1f8: "C2PMSG_62 (EXCEPTION PC!)",
    0x208: "C2PMSG_66",
    0x244: "C2PMSG_81 (SoL!)",
    0x270: "C2PMSG_92 (claimed BOOT_STATUS)",
    0x2cc: "C2PMSG_115",
    0x2f8: "C2PMSG_126",
}

subblob_files = sorted([f for f in CAP_DIR.glob("subblob_*.bin")])
print(f"Scanning {len(subblob_files)} sub-blobs for stores to MP0 C2PMSG offsets\n")

# Tag PSP base loads — find `mov.w rN, #0x3200000` patterns
# Encoding: e.g. for r5: 4ff04875 (mov.w r5, #0x3200000)
# This is movw + movt typically, or in this case a single mov.w with imm shift
# Pattern: "mov.w rN, #0x3200000" — let's search by string after disasm

results = {}  # blob → list of (addr, mnem, op_str, offset, label)

# Also track functions that load 0x3200000
psp_base_loaders = {}  # blob → list of fn_start addrs (approximated)

for sf in subblob_files:
    data = sf.read_bytes()
    name = sf.name
    blob_results = []
    blob_psp_loaders = []

    # Try Thumb-2 disasm of whole file (skipdata enabled)
    insns = list(md_thumb.disasm(data, 0))

    # Find PSP base loads
    last_fn_start = 0
    for ins in insns:
        if ins.mnemonic in ("push", "push.w") and "lr" in ins.op_str:
            last_fn_start = ins.address
        if ins.mnemonic in ("mov.w", "mov", "movw", "movs") and "#0x3200000" in ins.op_str:
            blob_psp_loaders.append((ins.address, last_fn_start, ins.op_str))

    # Find stores to interesting offsets
    for ins in insns:
        if not ins.mnemonic.startswith(("str", "stm")): continue
        # Look for "[rN, #<imm>]" pattern
        m = re.search(r'\[r(\d+|sb|sl|fp|ip|sp)(?:,\s*#0x([0-9a-fA-F]+))?', ins.op_str)
        if not m: continue
        if m.group(2):
            try:
                off_val = int(m.group(2), 16)
            except: continue
            if off_val in TARGET_OFFSETS:
                blob_results.append((ins.address, ins.mnemonic, ins.op_str, off_val, TARGET_OFFSETS[off_val]))

    if blob_results or blob_psp_loaders:
        results[name] = blob_results
        psp_base_loaders[name] = blob_psp_loaders

# Print findings
print("=" * 90)
print("PHASE 1: Stores with offset matching MP0 C2PMSG registers")
print("=" * 90)
for blob_name, hits in results.items():
    if not hits: continue
    print(f"\n{blob_name}:")
    # Group by offset
    by_off = {}
    for h in hits:
        by_off.setdefault(h[3], []).append(h)
    for off, group in sorted(by_off.items()):
        lbl = TARGET_OFFSETS[off]
        marker = ""
        if off in (0x1f0, 0x1f4, 0x1f8):
            marker = "  ★★★ EXCEPTION TRIPLET WRITE — GOLD ★★★"
        elif off == 0x244:
            marker = "  ★ SoL writer"
        elif off == 0xd8:
            marker = "  (DEBUG_STATUS — known helper @0x1d50)"
        print(f"  offset +0x{off:x} ({lbl}): {len(group)} hits{marker}")
        for addr, mnem, op, _, _ in group[:6]:
            print(f"    0x{addr:08x}: {mnem} {op}")

print("\n" + "=" * 90)
print("PHASE 2: Functions that load PSP MP0 base #0x3200000 across all sub-blobs")
print("=" * 90)
total = 0
for blob_name, loaders in psp_base_loaders.items():
    if not loaders: continue
    print(f"\n{blob_name}: {len(loaders)} sites load #0x3200000")
    for addr, fn_start, op in loaders[:10]:
        print(f"  fn @ 0x{fn_start:x}, load @ 0x{addr:x}: {op}")
    total += len(loaders)
print(f"\nTotal: {total} sites load PSP MP0 base across all sub-blobs")
