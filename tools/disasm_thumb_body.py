#!/usr/bin/env python3
"""
η.3j-RE-3 — structured Thumb-2 body analysis of captures/sos_subblob.bin.

Probe results (tools/probe_regions.py) revealed:
  - 0x0000..0x6FFF: plaintext ARM + Thumb code + small data
  - 0x7000..0x14FFF: 56 KB ENCRYPTED PAYLOAD (decodes "successfully"
    in both ARM and Thumb modes => high entropy => signed firmware
    that PSP BL decrypts into SRAM at runtime)
  - 0x15000..0x19FFF: plaintext Thumb code (post-decrypt stub / handler)
  - 0x1A000..end:    likely signature / cert chain

So real plaintext Thumb regions are 0x500..0x6D00 and 0x15140..0x19180.
We skip the encrypted blob entirely (no recovery without sig keys).

This script uses a restart-on-failure decoder so capstone doesn't bail
at the first literal-pool byte. Structured output:
  1. Literal-pool constant histogram, classified by likely address-space
  2. Polling-loop candidates (back-branch ≤16 bytes, preceded by LDR+CMP)
  3. BL call-frequency hot-spots
  4. MMIO write sequences (str.w to a literal-pool-loaded base)
"""

import struct
from collections import Counter, defaultdict
from pathlib import Path
from capstone import Cs, CS_ARCH_ARM, CS_MODE_THUMB

BLOB = Path("/home/pantelis/Desktop/Projects/Work/MLRift/captures/sos_subblob.bin")
data = BLOB.read_bytes()
BLOB_SIZE = len(data)

REGIONS = [
    ("R1", 0x500,   0x6D00),
    ("R2", 0x15140, 0x19180),
]

md = Cs(CS_ARCH_ARM, CS_MODE_THUMB)
md.detail = True


def robust_disasm_thumb(lo: int, hi: int):
    """Decode Thumb-2 instructions in [lo, hi); on capstone halt advance
    by 2 bytes and retry. Returns a list of instructions."""
    out = []
    cur = lo
    while cur < hi:
        # capstone.disasm is a generator; we restart it after every fail.
        progress = 0
        for ins in md.disasm(data[cur:hi], cur):
            out.append(ins)
            progress = (ins.address + ins.size) - cur
        if progress == 0:
            # couldn't decode the halfword at `cur`; skip 2 bytes
            cur += 2
        else:
            cur += progress
    return out


def classify_const(v: int) -> str:
    if v == 0:
        return "zero"
    if 0xFFFF0000 <= v <= 0xFFFFFFFF:
        return "sentinel_FFFF"
    if (v & 0xFFFF0000) in (0xDEAD0000, 0xCAFE0000, 0xBEEF0000, 0xC0DE0000, 0xFEED0000):
        return "sentinel_magic"
    if v < BLOB_SIZE:
        return "in_blob"
    if v < 0x20000:
        return "psp_sram_low"
    if v < 0x80000:
        return "psp_sram_high"
    if 0x03000000 <= v < 0x03020000:
        return "smn_range"
    if 0x06000000 <= v < 0x07000000:
        return "smn_range_6"
    if 0x10000000 <= v < 0x80000000:
        return "gpu_bar0_like"
    if 0x80000000 <= v < 0xFFFF0000:
        return "high_mmio"
    return "other"


def disasm_region(label: str, lo: int, hi: int):
    print(f"\n{'=' * 70}")
    print(f"REGION {label}: 0x{lo:x}..0x{hi:x}  ({hi - lo} bytes)")
    print(f"{'=' * 70}\n")

    insns = robust_disasm_thumb(lo, hi)
    print(f"  decoded {len(insns)} Thumb-2 instructions "
          f"({len(insns) * 2}–{len(insns) * 4} bytes of code)\n")

    by_addr = {ins.address: ins for ins in insns}
    addrs   = sorted(by_addr.keys())

    lit_consts = []                              # (ins_addr, lit_addr, mnemonic, value)
    bl_targets: Counter = Counter()
    branch_back = []                             # (ins_addr, target_addr)
    cmp_sites = []
    ldr_mem_sites = []
    movw_seen = {}                               # addr -> (reg, low16)
    pair_movw_movt = []                          # (addr, reg, full32)

    for i, ins in enumerate(insns):
        m = ins.mnemonic
        op = ins.op_str

        if m in ("ldr", "ldr.w") and "[pc" in op:
            try:
                imm = int(op.split("#")[1].rstrip("]"), 0)
                la = (ins.address + 4 + imm) & ~3
                if 0 <= la < len(data) - 4:
                    v = struct.unpack_from("<I", data, la)[0]
                    lit_consts.append((ins.address, la, m, v))
            except (ValueError, IndexError):
                pass

        elif m == "bl":
            try:
                tgt = int(op.lstrip("#"), 0)
                bl_targets[tgt] += 1
            except (ValueError, IndexError):
                pass

        elif m in ("b", "b.w", "bne", "beq", "bne.w", "beq.w",
                   "bgt", "blt", "bge", "ble", "bhi", "bls", "bcs", "bcc",
                   "cbz", "cbnz"):
            try:
                # cbz/cbnz: "rN, #addr"; b*: "#addr"
                tgt = int(op.split("#")[-1], 0)
                if tgt < ins.address:
                    branch_back.append((ins.address, tgt))
            except (ValueError, IndexError):
                pass

        elif m in ("cmp", "cmn", "tst", "teq"):
            cmp_sites.append(ins.address)

        elif m in ("ldr", "ldr.w", "ldrh", "ldrb") and "[" in op and "[pc" not in op:
            ldr_mem_sites.append(ins.address)

        # movw/movt pair detection (literal address synthesis)
        elif m in ("movw", "mov.w") and ", #" in op:
            try:
                parts = op.split(", ")
                reg = parts[0]
                imm = int(parts[-1].lstrip("#"), 0)
                movw_seen[reg] = (ins.address, imm)
            except (ValueError, IndexError):
                pass
        elif m == "movt" and ", #" in op:
            try:
                parts = op.split(", ")
                reg = parts[0]
                imm = int(parts[-1].lstrip("#"), 0)
                if reg in movw_seen and ins.address - movw_seen[reg][0] < 16:
                    full = (imm << 16) | movw_seen[reg][1]
                    pair_movw_movt.append((ins.address, reg, full))
            except (ValueError, IndexError):
                pass

    # ------------------------------------------------------------------
    # 1. Literal pool histogram
    # ------------------------------------------------------------------
    val_to_sites: dict[int, list[int]] = defaultdict(list)
    for addr, _la, _mn, v in lit_consts:
        val_to_sites[v].append(addr)

    # Also fold in movw/movt synthesized constants
    movw_val_to_sites: dict[int, list[int]] = defaultdict(list)
    for addr, _reg, v in pair_movw_movt:
        movw_val_to_sites[v].append(addr)

    all_const_to_sites = defaultdict(list)
    for v, sites in val_to_sites.items():
        all_const_to_sites[v].extend(sites)
    for v, sites in movw_val_to_sites.items():
        all_const_to_sites[v].extend(sites)

    by_class: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for v, sites in all_const_to_sites.items():
        by_class[classify_const(v)].append((v, len(sites)))

    print(f"  ── Constants ({len(lit_consts)} LDR-pool + "
          f"{len(pair_movw_movt)} movw/movt, "
          f"{len(all_const_to_sites)} unique) ──")
    interesting = [
        "smn_range", "smn_range_6", "gpu_bar0_like", "high_mmio",
        "psp_sram_high", "psp_sram_low",
        "sentinel_FFFF", "sentinel_magic",
    ]
    for cls in interesting:
        rows = by_class.get(cls, [])
        if not rows:
            continue
        rows.sort(key=lambda r: -r[1])
        print(f"\n    [{cls}]  ({len(rows)} unique)")
        for v, n in rows[:30]:
            example = all_const_to_sites[v][0]
            print(f"      0x{v:08x}  ×{n:<3}  e.g. @0x{example:x}")

    inb   = by_class.get("in_blob", [])
    other = by_class.get("other", [])
    zero  = by_class.get("zero", [])
    print(f"\n    [in_blob]    {len(inb)} unique self-refs")
    print(f"    [other]      {len(other)} unique")
    print(f"    [zero]       {len(zero)} unique")

    # ------------------------------------------------------------------
    # 2. Polling-loop candidates
    # ------------------------------------------------------------------
    print(f"\n  ── Polling-loop candidates (back-branch ≤32 insns,"
          f" preceded by LDR+CMP) ──")
    addr_idx = {a: i for i, a in enumerate(addrs)}
    cmp_set = set(cmp_sites)
    ldr_set = set(ldr_mem_sites)

    poll_candidates = []
    for ins_addr, tgt in branch_back:
        delta = ins_addr - tgt
        if delta > 64:
            continue
        i = addr_idx.get(ins_addr, -1)
        if i < 0:
            continue
        window_addrs = addrs[max(0, i - 8):i]
        has_ldr = any(a in ldr_set for a in window_addrs)
        has_cmp = any(a in cmp_set for a in window_addrs)
        if has_ldr and has_cmp:
            poll_candidates.append((tgt, ins_addr, delta))

    print(f"    found {len(poll_candidates)} candidates")
    for tgt, ba, d in poll_candidates[:50]:
        i = addr_idx.get(tgt, -1)
        if i < 0:
            continue
        body = []
        for j in range(i, len(addrs)):
            a = addrs[j]
            if a > ba:
                break
            ins = by_addr[a]
            body.append(f"        0x{a:08x}: {ins.mnemonic:<8} {ins.op_str}")
        print(f"\n    loop @ 0x{tgt:x}..0x{ba:x}  ({d} bytes)")
        for line in body:
            print(line)

    # ------------------------------------------------------------------
    # 3. BL target hot-spots
    # ------------------------------------------------------------------
    print(f"\n  ── Top 15 BL call targets (probable utility fns) ──")
    for tgt, n in bl_targets.most_common(15):
        in_region = lo <= tgt < hi
        in_blob   = 0 <= tgt < BLOB_SIZE
        marker = "" if in_region else ("  [outside region]" if in_blob else "  [outside blob!]")
        print(f"    0x{tgt:08x}  ×{n}{marker}")

    return {
        "lit_consts": lit_consts,
        "bl_targets": bl_targets,
        "poll_candidates": poll_candidates,
        "by_class": by_class,
        "all_const_to_sites": all_const_to_sites,
        "movw_movt": pair_movw_movt,
    }


def main():
    print(f"loaded {BLOB} ({BLOB_SIZE} bytes, 0x{BLOB_SIZE:x})\n")
    print("Skipping encrypted payload region 0x7000..0x15000 (high-entropy,")
    print("decodes 'successfully' in both ARM and Thumb modes — signature of")
    print("AES-encrypted firmware that PSP BL decrypts into SRAM at runtime).")

    results = {}
    for label, lo, hi in REGIONS:
        results[label] = disasm_region(label, lo, hi)

    # Cross-region MMIO-like summary
    print(f"\n\n{'=' * 70}")
    print("CROSS-REGION MMIO-LIKE CONSTANT SUMMARY")
    print(f"{'=' * 70}")
    mmio_classes = ("smn_range", "smn_range_6", "gpu_bar0_like", "high_mmio")
    all_mmio: Counter = Counter()
    for label, res in results.items():
        for cls in mmio_classes:
            for v, n in res["by_class"].get(cls, []):
                all_mmio[v] += n
    print(f"\n  unique MMIO-like constants across both regions: {len(all_mmio)}")
    for v, n in all_mmio.most_common(50):
        print(f"    0x{v:08x}  ×{n}  ({classify_const(v)})")


if __name__ == "__main__":
    main()
