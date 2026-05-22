#!/usr/bin/env python3
"""
η.3q-4 — try to disassemble SYS_DRV body in ARM and Thumb modes.

The body starts at 0x1000 (after 4 KiB of header+padding) and runs to
~0xe000 (before the encrypted chunk at 0xe000-0xe400 + trailing pad).
Entropy 160-200/256 in that range could mean:
  (a) compressed code (LZSS/LZMA) — won't disasm
  (b) lightly-encrypted code (XOR/ARM-stream cipher) — won't disasm
  (c) raw ARM/Thumb code with embedded data — partial disasm

Strategy: try both ARM and Thumb decode at offsets 0x1000, 0x1100,
0x2000, 0x4000, 0x8000, 0xc000. Count successful insns to determine
which mode matches. Look for "real" code signatures (push/pop epilogues,
common opcodes, sensible literal-pool reads).

ALSO try the other plaintext-ish blobs: SOC_DRV, INTF_DRV, DBG_DRV.
"""
import struct
from pathlib import Path
from capstone import Cs, CS_ARCH_ARM, CS_MODE_ARM, CS_MODE_THUMB
from collections import Counter

OUT_DIR = Path("/home/pantelis/Desktop/Projects/Work/MLRift/captures")
md_arm   = Cs(CS_ARCH_ARM, CS_MODE_ARM)
md_thumb = Cs(CS_ARCH_ARM, CS_MODE_THUMB)

def assess_mode(data, start, length, md):
    """Try decoding `length` bytes starting at `start`. Return
    (insns_decoded, bytes_covered, mnemonic_counter)."""
    chunk = data[start:start + length]
    insns = []
    for ins in md.disasm(chunk, start):
        insns.append(ins)
    coverage = sum(ins.size for ins in insns)
    mns = Counter(ins.mnemonic for ins in insns)
    return (len(insns), coverage, mns)


def analyze(name, data):
    print(f"\n{'=' * 70}")
    print(f"ANALYZE: {name} ({len(data)} bytes)")
    print(f"{'=' * 70}")
    test_offsets = [0x1000, 0x1100, 0x2000, 0x4000, 0x8000, 0xc000]
    for off in test_offsets:
        if off + 256 > len(data): continue
        # Try Thumb first
        t_count, t_cov, t_mns = assess_mode(data, off, 256, md_thumb)
        a_count, a_cov, a_mns = assess_mode(data, off, 256, md_arm)
        # Quality metric: prefer the mode that covers MORE bytes AND has
        # a healthy mix of common opcodes (not just one mnemonic).
        t_quality = t_cov + 4 * len(t_mns)  # bonus for mnemonic diversity
        a_quality = a_cov + 4 * len(a_mns)
        verdict = "THUMB" if t_quality > a_quality else "ARM"
        print(f"\n  @0x{off:04x}: Thumb covers {t_cov}/256 bytes, {len(t_mns):2d} distinct mnems  | "
              f"ARM covers {a_cov}/256, {len(a_mns):2d} distinct mnems  → {verdict}")
        if verdict == "THUMB":
            print(f"    Top Thumb mnemonics: {dict(t_mns.most_common(5))}")
        else:
            print(f"    Top ARM mnemonics:   {dict(a_mns.most_common(5))}")


# Process each plaintext-ish sub-blob
for name in ["sys_drv", "soc_drv", "intf_drv", "dbg_drv", "spl"]:
    p = OUT_DIR / f"{name}_subblob.bin"
    if not p.exists():
        print(f"SKIP {name}: not found")
        continue
    analyze(name, p.read_bytes())

# Print a focused disasm of SYS_DRV at the most promising offset
print("\n" + "=" * 70)
print("FOCUSED DISASM: SYS_DRV body in best-matching mode")
print("=" * 70)
sys_drv = (OUT_DIR / "sys_drv_subblob.bin").read_bytes()
# Try Thumb at 0x1000, print first 40 insns
print("\n--- Thumb @ 0x1000 (first 50 insns) ---")
for ins in list(md_thumb.disasm(sys_drv[0x1000:0x1200], 0x1000))[:50]:
    print(f"  0x{ins.address:08x}: {ins.bytes.hex():<10} {ins.mnemonic:<8} {ins.op_str}")

print("\n--- ARM @ 0x1000 (first 50 insns) ---")
for ins in list(md_arm.disasm(sys_drv[0x1000:0x1200], 0x1000))[:50]:
    print(f"  0x{ins.address:08x}: {ins.bytes.hex():<10} {ins.mnemonic:<8} {ins.op_str}")
