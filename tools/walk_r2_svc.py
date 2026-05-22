#!/usr/bin/env python3
"""
η.3p-1 — enumerate all SVC instructions in R2 (post-decrypt C runtime)
and find any code that writes to PSP SRAM page-table region (0xc000-0xc200).

SVC #N is a syscall into the encrypted PSP kernel. Each distinct N is
a kernel API the user-mode code calls into. If we can identify a kernel
API related to MMU page-table management, that's the smoking gun.

Also looks in R1 for:
  - MCR p15, c2, ... (TTBR0/1 writes)
  - MCR p15, c8, ... (TLB invalidation)
  - Stores to addresses in [0xc000, 0xc200) (initial page-table content
    setup, pre-MMU-enable)
"""

import struct
from collections import Counter, defaultdict
from pathlib import Path
from capstone import Cs, CS_ARCH_ARM, CS_MODE_ARM, CS_MODE_THUMB

BLOB = Path("/home/pantelis/Desktop/Projects/Work/MLRift/captures/sos_subblob.bin")
data = BLOB.read_bytes()

md_arm   = Cs(CS_ARCH_ARM, CS_MODE_ARM)
md_thumb = Cs(CS_ARCH_ARM, CS_MODE_THUMB)


def robust_thumb(lo, hi):
    out = []
    cur = lo
    while cur < hi:
        progress = 0
        for ins in md_thumb.disasm(data[cur:hi], cur):
            out.append(ins)
            progress = (ins.address + ins.size) - cur
        if progress == 0:
            cur += 2
        else:
            cur += progress
    return out


def robust_arm(lo, hi):
    out = []
    cur = lo
    while cur < hi:
        progress = 0
        for ins in md_arm.disasm(data[cur:hi], cur):
            out.append(ins)
            progress = (ins.address + ins.size) - cur
        if progress == 0:
            cur += 4
        else:
            cur += progress
    return out


# =====================================================================
# 1. R2 SVC enumeration (Thumb)
# =====================================================================
print("=" * 70)
print("R2 SVC ENUMERATION (post-decrypt C runtime, 0x15140..0x19180)")
print("=" * 70)
insns = robust_thumb(0x15140, 0x19180)
print(f"  Decoded {len(insns)} Thumb-2 instructions in R2\n")

svc_calls = []   # (file_pc, svc_num, surrounding_context)
addr_by_idx = {i: ins.address for i, ins in enumerate(insns)}
idx_by_addr = {ins.address: i for i, ins in enumerate(insns)}

for i, ins in enumerate(insns):
    if ins.mnemonic != "svc":
        continue
    try:
        num = int(ins.op_str.lstrip("#"), 0)
    except (ValueError, IndexError):
        continue
    # Capture the 4 preceding insns (typical: arg setup via mov r0/r1/r2/r3)
    ctx = []
    for j in range(max(0, i - 5), i):
        c = insns[j]
        ctx.append((c.address, c.mnemonic, c.op_str))
    svc_calls.append((ins.address, num, ctx))

svc_counter = Counter(num for _, num, _ in svc_calls)
print(f"  Found {len(svc_calls)} SVC sites across {len(svc_counter)} distinct numbers\n")

print("  --- SVC numbers by frequency ---")
for num, count in sorted(svc_counter.items(), key=lambda kv: -kv[1]):
    print(f"    SVC #0x{num:02x} (= {num:3d})   ×{count}")

# Print full context for first occurrence of each SVC number
print("\n  --- First occurrence of each SVC (with arg-setup context) ---")
seen = set()
for pc, num, ctx in svc_calls:
    if num in seen:
        continue
    seen.add(num)
    print(f"\n  ---- SVC #0x{num:02x} (= {num}) — first call site @ 0x{pc:x} ----")
    for addr, mn, op in ctx:
        print(f"    0x{addr:08x}: {mn:<8} {op}")
    print(f"    0x{pc:08x}: svc      #{num}   <-- SVC entry")

# =====================================================================
# 2. R1 MCR p15 writes (CP15 system register accesses)
# =====================================================================
print("\n" + "=" * 70)
print("R1 CP15 ACCESSES (looking for MMU c2 writes / TLB c8 ops)")
print("=" * 70)
r1_arm = robust_arm(0x100, 0x500)   # The plaintext ARM-mode reset region
print(f"  Decoded {len(r1_arm)} ARM instructions in 0x100..0x500\n")

cp15_targets = {
    "c2":  "TTBR/TTBCR (translation tables)",
    "c3":  "DACR (domain access control)",
    "c7":  "Cache ops",
    "c8":  "TLB ops",
    "c10": "Memory attributes",
    "c12": "VBAR (vector base)",
    "c1":  "SCTLR (MMU/cache enable)",
    "c0":  "Misc (CTR/MIDR/etc.)",
}

print(f"  --- ALL MCR/MRC p15 in 0x100..0x500 ---")
for ins in r1_arm:
    if not (ins.mnemonic.startswith("mcr") or ins.mnemonic.startswith("mrc")):
        continue
    op = ins.op_str
    # parse "p15, #0, r1, c2, c0, #0"
    label = ""
    for cr, desc in cp15_targets.items():
        if f", {cr}," in op or op.endswith(f", {cr}"):
            label = f"  [{desc}]"
            break
    print(f"    0x{ins.address:08x}: {ins.mnemonic:<8} {op:<40}{label}")

# =====================================================================
# 3. R1 stores to page-table region [0xc000, 0xc200)
# =====================================================================
print("\n" + "=" * 70)
print("R1 STORES TARGETING PSP SRAM PAGE-TABLE REGION 0xc000..0xc200")
print("=" * 70)

# These would be of form:
#   ldr r_base, =0xc000 (or similar)
#   str r_val, [r_base, #offset]
# In R1 ARM mode, look at PC-relative LDR constants in the page-table range
# and immediately following str/strd/stm to that register.

print("  Scanning ARM region 0x100..0x600 for page-table setup patterns...")
for ins in r1_arm:
    if ins.mnemonic != "ldr":
        continue
    op = ins.op_str
    if "[pc" not in op:
        continue
    try:
        imm = int(op.split("#")[1].rstrip("]"), 0)
        la = ins.address + 8 + imm
        if 0 <= la < len(data) - 4:
            v = struct.unpack_from("<I", data, la)[0]
            if 0xc000 <= v < 0xd000:
                rd = op.split(",")[0].strip()
                # Look ahead 12 insns for stores via this register
                i_here = next((j for j, x in enumerate(r1_arm) if x.address == ins.address), -1)
                stores_to_rd = []
                if i_here >= 0:
                    for k in range(i_here + 1, min(i_here + 12, len(r1_arm))):
                        n = r1_arm[k]
                        if n.mnemonic in ("str", "strb", "strd", "stm", "stmia", "stmib"):
                            if f"[{rd}" in n.op_str or n.op_str.endswith(rd) or f", {rd}" in n.op_str:
                                stores_to_rd.append((n.address, n.mnemonic, n.op_str))
                marker = "  <<< PAGE TABLE REGION" if 0xc000 <= v < 0xc200 else ""
                print(f"  0x{ins.address:08x}: ldr {rd}, =0x{v:08x}{marker}")
                for sa, smn, sop in stores_to_rd:
                    print(f"      0x{sa:08x}: {smn:<6} {sop}")
    except (ValueError, IndexError):
        pass

# =====================================================================
# 4. R2 LDR Rd, [PC, #imm] constants — look for known PSP-virt addresses
#    (e.g. anything in 0x80000000-0x90000000 high-aperture range)
# =====================================================================
print("\n" + "=" * 70)
print("R2 HIGH-ADDRESS CONSTANTS (look for 0x83f00f80 or relatives)")
print("=" * 70)
high_consts = Counter()
for ins in insns:
    if ins.mnemonic not in ("ldr", "ldr.w"):
        continue
    if "[pc" not in ins.op_str:
        continue
    try:
        imm = int(ins.op_str.split("#")[1].rstrip("]"), 0)
        la = (ins.address + 4 + imm) & ~3
        if 0 <= la < len(data) - 4:
            v = struct.unpack_from("<I", data, la)[0]
            if 0x80000000 <= v < 0x90000000:
                high_consts[v] += 1
    except (ValueError, IndexError):
        pass

if high_consts:
    print(f"  Found {len(high_consts)} unique high-aperture constants in R2:")
    for v, n in high_consts.most_common(30):
        marker = "  <<< MATCHES FAULT 0x83f00f80" if v == 0x83f00f80 else ""
        marker_n = "  <<< NEIGHBOR (within 4 KB of fault)" if abs(v - 0x83f00f80) < 0x1000 else ""
        print(f"    0x{v:08x}  ×{n}{marker}{marker_n}")
else:
    print("  No high-aperture constants found in R2 literal pools.")
