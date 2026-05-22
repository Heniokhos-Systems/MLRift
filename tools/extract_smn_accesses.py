#!/usr/bin/env python3
"""
η.3j-RE-3a — extract SMN MMIO access patterns from R1 (0x500..0x6D00).

Strategy:
  - For every LDR Rd, [PC, #imm] that loads a constant in 0x03010000..0x03020000
    range, look ahead 16 insns for [Rd, #off] or [Rd] accesses.
  - Track the immediately-following pattern:
      * str Rs, [Rd, #off]    -> WRITE to SMN reg
      * ldr Rs, [Rd, #off]    -> READ  from SMN reg
      * add Rt, Rd, Rs        -> Rt = SMN base + Rs (Rs is dyn offset)
  - For each SMN base, list all (file_pc, offset, op_kind) accesses.

Also disassemble specific helper functions of interest:
  - 0x6188 — recurs in polling loops (likely SMN-read helper)
  - 0x4d8  — called from 0x23ba/0x24c8 (memcmp-like?)
  - 0x2cd4, 0x104c, 0x4f9c — called from 0x155e poll
"""

import struct
from collections import defaultdict, Counter
from pathlib import Path
from capstone import Cs, CS_ARCH_ARM, CS_MODE_THUMB

BLOB = Path("/home/pantelis/Desktop/Projects/Work/MLRift/captures/sos_subblob.bin")
data = BLOB.read_bytes()

md = Cs(CS_ARCH_ARM, CS_MODE_THUMB)
md.detail = True


def robust_disasm(lo: int, hi: int):
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


def reg_of(operand: str) -> str:
    """Extract leading register name from an op_str like 'r0, [r1]'."""
    s = operand.strip()
    if "," in s:
        s = s.split(",", 1)[0]
    if "[" in s:
        s = s.split("[", 1)[0]
    return s.strip()


def first_reg(op_str: str) -> str:
    return reg_of(op_str.split(",")[0])


def base_reg_from_bracket(op_str: str) -> str:
    """For 'r0, [r1, #0x10]' return 'r1'."""
    if "[" not in op_str:
        return ""
    inside = op_str.split("[", 1)[1].split("]")[0]
    return inside.split(",")[0].strip()


def parse_offset(op_str: str) -> int | None:
    """For '[r1, #0x10]' return 0x10, for '[r1]' return 0."""
    if "[" not in op_str:
        return None
    inside = op_str.split("[", 1)[1].split("]")[0]
    if "," not in inside:
        return 0
    rest = inside.split(",", 1)[1].strip()
    if rest.startswith("#"):
        try:
            return int(rest.lstrip("#"), 0)
        except ValueError:
            return None
    return None


print("=== R1 SMN access extraction (base in 0x03010000..0x03020000) ===\n")
insns = robust_disasm(0x500, 0x6D00)
addr_idx = {ins.address: i for i, ins in enumerate(insns)}

smn_accesses: dict[int, list[tuple[int, str, int]]] = defaultdict(list)
# smn_base -> list of (ins_pc, kind, offset)

# Track register liveness from each SMN literal load.
# For each LDR Rd, [PC, #imm] with const in SMN range:
#   - Note Rd is "tainted" with that base
#   - Walk forward up to 16 insns; if we see str/ldr through Rd, record
for i, ins in enumerate(insns):
    if ins.mnemonic not in ("ldr", "ldr.w"):
        continue
    if "[pc" not in ins.op_str:
        continue
    try:
        imm = int(ins.op_str.split("#")[1].rstrip("]"), 0)
        la = (ins.address + 4 + imm) & ~3
        if la >= len(data) - 4:
            continue
        v = struct.unpack_from("<I", data, la)[0]
    except (ValueError, IndexError):
        continue

    if not (0x03010000 <= v < 0x03020000):
        continue

    rd = first_reg(ins.op_str)  # e.g. 'r5'
    base = v

    # walk forward 16 insns or until rd is overwritten
    rd_live = True
    for j in range(i + 1, min(i + 17, len(insns))):
        fwd = insns[j]
        fm = fwd.mnemonic
        fop = fwd.op_str
        if not rd_live:
            break

        # Memory access through Rd?
        if fm in ("str", "str.w", "strh", "strb") and base_reg_from_bracket(fop) == rd:
            off = parse_offset(fop)
            smn_accesses[base].append((fwd.address, "WRITE", off or 0))
        elif fm in ("ldr", "ldr.w", "ldrh", "ldrb") and base_reg_from_bracket(fop) == rd:
            off = parse_offset(fop)
            smn_accesses[base].append((fwd.address, "READ", off or 0))
        elif fm in ("add", "add.w") and first_reg(fop) == rd:
            # Rd = Rd + Rsomething : tainted base now has dynamic offset
            # we'll record this as "DYN" and not track further
            smn_accesses[base].append((fwd.address, "ADD_DYN", 0))
            rd_live = False

        # Overwrite check: if rd appears as dest in mov / ldr (non-mem-through-rd)
        if "," in fop:
            dest = first_reg(fop)
            if dest == rd and fm not in ("str", "str.w", "strh", "strb"):
                # check if rd is being overwritten (not just used as base)
                if fm.startswith(("mov", "ldr", "add", "sub", "lsl", "lsr",
                                  "and", "orr", "eor", "asr")):
                    if fm.startswith(("ldr",)) and base_reg_from_bracket(fop) == rd:
                        # ldr Rd, [Rd] -- this is OK, Rd value changes after but
                        # the access is through old Rd which we already logged
                        pass
                    else:
                        rd_live = False

# Print results grouped by SMN base.
total_acc = sum(len(v) for v in smn_accesses.values())
print(f"  found {total_acc} SMN accesses across {len(smn_accesses)} bases\n")

for base in sorted(smn_accesses.keys()):
    accs = smn_accesses[base]
    print(f"  SMN base 0x{base:08x}  ({len(accs)} accesses)")
    # Aggregate by (kind, offset)
    by_pair: Counter = Counter()
    for pc, kind, off in accs:
        by_pair[(kind, off)] += 1
    for (kind, off), n in sorted(by_pair.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        full = base + off
        print(f"      {kind:<8} +0x{off:<4x} = 0x{full:08x}  ×{n}")
    print()


# --------------------------------------------------------------------
# 2. Disassemble suspect helper functions
# --------------------------------------------------------------------
print("\n" + "=" * 70)
print("HELPER FUNCTION DISASSEMBLY")
print("=" * 70 + "\n")

HELPERS = [
    (0x6188, 0x80, "called from polling loops 0x1e10, 0x2736"),
    (0x4d8,  0x60, "called from 0x23ba, 0x24c8"),
    (0x2cd4, 0x80, "called from 0x155e poll"),
    (0x104c, 0x40, "called from 0x155e poll"),
    (0x4f9c, 0x40, "called from 0x155e poll"),
]

for off, length, ctx in HELPERS:
    print(f"=== fn @ 0x{off:x}  ({ctx}) ===")
    chunk = data[off:off + length]
    for ins in md.disasm(chunk, off):
        # flag literal-pool loads
        flag = ""
        if ins.mnemonic in ("ldr", "ldr.w") and "[pc" in ins.op_str:
            try:
                imm = int(ins.op_str.split("#")[1].rstrip("]"), 0)
                la = (ins.address + 4 + imm) & ~3
                if 0 <= la < len(data) - 4:
                    v = struct.unpack_from("<I", data, la)[0]
                    flag = f"  -> *0x{la:x} = 0x{v:08x}"
            except (ValueError, IndexError):
                pass
        print(f"  0x{ins.address:08x}: {ins.bytes.hex():<10} {ins.mnemonic:<8} {ins.op_str:<35}{flag}")
        # Stop at first unconditional return
        if ins.mnemonic in ("bx", "pop.w") and ("lr" in ins.op_str or "pc" in ins.op_str):
            if ins.mnemonic == "bx" or "pc" in ins.op_str:
                break
    print()


# --------------------------------------------------------------------
# 3. Inspect the 0x1e10 polling loop's broader call context
# --------------------------------------------------------------------
print("\n=== POLLING-LOOP CONTEXT: trace caller of 0x1e10 region ===\n")
# Find what calls into 0x1e0X (the function containing the polling loop)
target_min = 0x1e00
target_max = 0x1e80
callers = []
for ins in insns:
    if ins.mnemonic == "bl":
        try:
            tgt = int(ins.op_str.lstrip("#"), 0)
            if target_min <= tgt <= target_max:
                callers.append((ins.address, tgt))
        except (ValueError, IndexError):
            pass
print(f"  callers of [0x{target_min:x}..0x{target_max:x}]: {len(callers)}")
for pc, tgt in callers:
    print(f"    @0x{pc:x}  bl  0x{tgt:x}")

# Same for 0x6188 (the suspect helper)
print(f"\n  callers of 0x6188:")
for ins in insns:
    if ins.mnemonic == "bl":
        try:
            tgt = int(ins.op_str.lstrip("#"), 0)
            if tgt == 0x6188:
                print(f"    @0x{ins.address:x}  bl 0x6188")
        except (ValueError, IndexError):
            pass
