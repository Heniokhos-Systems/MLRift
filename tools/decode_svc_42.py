#!/usr/bin/env python3
"""
η.3p-2 — decode every call site of SVC #0x42 (hypothesized = map_physical).
For each site, look BACKWARDS to find what was loaded into r0/r1/r2/r3,
especially r0 (the address argument).

Goal: find any SVC #0x42 call whose r0 is in 0x80000000..0x90000000
(high system aperture, where the fault address 0x83f00f80 lives).
If found, that's the user-mode mapping request — we'd need to ensure
the kernel honors it in our environment.

Also tabulates the top-10 SVCs with their argument patterns.
"""

import struct
from collections import Counter, defaultdict
from pathlib import Path
from capstone import Cs, CS_ARCH_ARM, CS_MODE_THUMB

BLOB = Path("/home/pantelis/Desktop/Projects/Work/MLRift/captures/sos_subblob.bin")
data = BLOB.read_bytes()

md = Cs(CS_ARCH_ARM, CS_MODE_THUMB)


def robust_thumb(lo, hi):
    out = []
    cur = lo
    while cur < hi:
        progress = 0
        for ins in md.disasm(data[cur:hi], cur):
            out.append(ins)
            progress = (ins.address + ins.size) - cur
        if progress == 0:
            cur += 2
        else:
            cur += progress
    return out


def parse_reg(s):
    """Get leading register name from operand string."""
    s = s.strip().rstrip(',')
    if "[" in s:
        s = s.split("[", 1)[0].strip()
    return s


def resolve_literal_load(ins, image_bytes):
    """For 'ldr Rd, [pc, #imm]', return (rd, value) or None."""
    if ins.mnemonic not in ("ldr", "ldr.w"):
        return None
    if "[pc" not in ins.op_str:
        return None
    try:
        rd = parse_reg(ins.op_str.split(",")[0])
        imm = int(ins.op_str.split("#")[1].rstrip("]"), 0)
        la = (ins.address + 4 + imm) & ~3
        if 0 <= la < len(image_bytes) - 4:
            v = struct.unpack_from("<I", image_bytes, la)[0]
            return (rd, v)
    except (ValueError, IndexError):
        pass
    return None


def trace_reg_value(insns, svc_idx, target_reg, max_back=20):
    """Walk backwards from svc_idx looking for the most recent
    instruction that sets target_reg. Returns (kind, value) where
    kind is 'imm'/'lit'/'reg_copy'/'unknown'."""
    for j in range(svc_idx - 1, max(0, svc_idx - max_back) - 1, -1):
        ins = insns[j]
        op = ins.op_str.replace(",", " ").split()
        if not op:
            continue
        dst = op[0]
        if dst != target_reg:
            continue
        m = ins.mnemonic
        # movs Rd, #imm
        if m == "movs" and len(op) >= 2 and op[1].startswith("#"):
            try:
                return ("imm", int(op[1].lstrip("#"), 0), ins.address)
            except: pass
        # mov.w Rd, #imm
        if m == "mov.w" and len(op) >= 2 and op[1].startswith("#"):
            try:
                return ("imm", int(op[1].lstrip("#"), 0) & 0xFFFFFFFF, ins.address)
            except: pass
        # mov Rd, Rs
        if m in ("mov", "mov.w") and len(op) >= 2 and op[1].startswith("r") or op[1] in ("sb","sl","fp","ip","sp","lr","r0","r1","r2","r3","r4","r5","r6","r7","r8","r9","r10","r11","r12"):
            return ("reg_copy", op[1], ins.address)
        # movw Rd, #imm (low half)
        if m == "movw" and len(op) >= 2 and op[1].startswith("#"):
            try:
                lo = int(op[1].lstrip("#"), 0) & 0xFFFF
                # Look for paired movt
                for k in range(j + 1, min(svc_idx, j + 5)):
                    n = insns[k]
                    nop = n.op_str.replace(",", " ").split()
                    if n.mnemonic == "movt" and nop and nop[0] == target_reg and len(nop) >= 2 and nop[1].startswith("#"):
                        hi = int(nop[1].lstrip("#"), 0) & 0xFFFF
                        return ("imm", lo | (hi << 16), ins.address)
                return ("imm", lo, ins.address)
            except: pass
        # ldr Rd, [pc, #imm] — literal pool
        if m in ("ldr", "ldr.w") and "[pc" in ins.op_str:
            r = resolve_literal_load(ins, data)
            if r and r[0] == target_reg:
                return ("lit", r[1], ins.address)
        # any other write to target_reg
        return ("unknown_set", m + " " + ins.op_str, ins.address)
    return ("not_set", None, None)


# Decode R2
insns = robust_thumb(0x15140, 0x19180)
print(f"Decoded {len(insns)} Thumb-2 insns in R2\n")

# Find all SVC sites
svc_sites = []
for i, ins in enumerate(insns):
    if ins.mnemonic != "svc":
        continue
    try:
        num = int(ins.op_str.lstrip("#"), 0)
    except: continue
    svc_sites.append((i, ins.address, num))

print(f"Total SVC sites: {len(svc_sites)}\n")

# Filter to SVC #0x42 (= 66)
svc_42 = [(i, pc, num) for i, pc, num in svc_sites if num == 0x42]
print(f"=== ALL {len(svc_42)} SVC #0x42 (map_physical?) call sites ===\n")

high_r0_count = 0
unique_r0_imms = Counter()
for idx, (i, pc, _) in enumerate(svc_42, 1):
    print(f"  [{idx:2d}] @0x{pc:x}")
    for rn in ("r0", "r1", "r2", "r3"):
        kind, val, src_pc = trace_reg_value(insns, i, rn, max_back=20)
        if kind == "imm":
            marker = ""
            if isinstance(val, int):
                if 0x80000000 <= val < 0x90000000:
                    marker = "  <<< HIGH APERTURE (0x83f00f80 region!)"
                    high_r0_count += 1 if rn == "r0" else 0
                elif 0x03000000 <= val < 0x04000000:
                    marker = "  (SMN range)"
                elif 0x10000000 <= val < 0x80000000:
                    marker = "  (BAR0-like)"
                if rn == "r0":
                    unique_r0_imms[val] += 1
            print(f"        {rn} = 0x{val:08x}  (lit/imm @0x{src_pc:x}){marker}")
        elif kind == "lit":
            print(f"        {rn} = 0x{val:08x}  (literal pool @0x{src_pc:x})")
            if rn == "r0":
                unique_r0_imms[val] += 1
        elif kind == "reg_copy":
            print(f"        {rn} ← {val}  (mov @0x{src_pc:x})")
        elif kind == "unknown_set":
            print(f"        {rn}: set by '{val}' @0x{src_pc:x}")
        else:
            print(f"        {rn}: not set in last 20 insns")
    print()

print(f"\n=== Summary of r0 values passed to SVC #0x42 ===")
for v, n in sorted(unique_r0_imms.items(), key=lambda kv: -kv[1]):
    marker = "  <<< matches fault aperture" if 0x80000000 <= v < 0x90000000 else ""
    print(f"  0x{v:08x}  ×{n}{marker}")

if high_r0_count == 0:
    print(f"\n  No SVC #0x42 call site passes r0 in 0x8xxxxxxx range.")
    print(f"  The user-mode R2 code does NOT request mapping of the 0x83f00f80")
    print(f"  region — that mapping (if it exists) must be set up by the")
    print(f"  kernel itself during its own boot phase (inside the encrypted body).")


# Also do the same for the next 2 most common SVCs to characterize them.
for target_svc in [0x11, 0x16, 0x37, 0x25]:
    sites = [(i, pc, num) for i, pc, num in svc_sites if num == target_svc]
    print(f"\n=== SVC #0x{target_svc:02x} — {len(sites)} sites, sample arg patterns ===")
    # Only show first 6 sites to keep output reasonable
    r0_imms = Counter()
    r0_kinds = Counter()
    for idx, (i, pc, _) in enumerate(sites[:6], 1):
        kind0, val0, _ = trace_reg_value(insns, i, "r0", max_back=20)
        kind1, val1, _ = trace_reg_value(insns, i, "r1", max_back=20)
        kind2, val2, _ = trace_reg_value(insns, i, "r2", max_back=20)
        r0_kinds[kind0] += 1
        if kind0 == "imm" or kind0 == "lit":
            r0_imms[val0] += 1
        fmt = lambda k, v: f"{v if not isinstance(v,int) else hex(v)}" if v is not None else "?"
        print(f"  [{idx}] @0x{pc:x}  r0={fmt(kind0, val0)} ({kind0})  r1={fmt(kind1, val1)} ({kind1})  r2={fmt(kind2, val2)} ({kind2})")
    if r0_imms:
        print(f"  r0 value frequencies (first 6 sites): {dict(r0_imms)}")
