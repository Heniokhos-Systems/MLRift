#!/usr/bin/env python3
"""
η.3j-RE-3b — decode the three call sites of WFI helper 0x6188.

0x6188 = dsb sy; wfi; bx lr  (wait for interrupt)
Callers: 0x1e10, 0x1e8c, 0x2736.

For each caller:
  1. Find the enclosing function (scan backwards for push {...} prologue)
  2. Print the function body
  3. Track where each register used in the poll predicate is loaded
  4. Identify any literal-pool constants
"""

import struct
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


# Decode a large window so we can find function boundaries.
ALL_INSNS = robust_disasm(0x500, 0x6D00)
by_addr = {ins.address: ins for ins in ALL_INSNS}
addrs = sorted(by_addr.keys())
addr_idx = {a: i for i, a in enumerate(addrs)}


def find_fn_start(call_addr: int) -> int:
    """Scan backwards for a push.w {...lr...} or push {... lr ...} prologue."""
    i = addr_idx.get(call_addr, -1)
    if i < 0:
        return call_addr
    for j in range(i, max(i - 200, 0), -1):
        a = addrs[j]
        ins = by_addr[a]
        if ins.mnemonic in ("push", "push.w") and "lr" in ins.op_str:
            return a
    # fallback: 60 insns back, aligned
    j = max(i - 60, 0)
    return addrs[j]


def find_fn_end(start: int) -> int:
    """Scan forwards from start for first pop.w {..., pc} or bx lr."""
    i = addr_idx.get(start, -1)
    if i < 0:
        return start + 0x100
    for j in range(i + 1, min(i + 200, len(addrs))):
        a = addrs[j]
        ins = by_addr[a]
        if ins.mnemonic in ("pop", "pop.w") and "pc" in ins.op_str:
            return a + ins.size
        if ins.mnemonic == "bx" and "lr" in ins.op_str:
            return a + ins.size
    return addrs[min(i + 100, len(addrs) - 1)]


def annotate(ins) -> str:
    """Annotate a single instruction with literal-pool dereference if applicable."""
    m, op = ins.mnemonic, ins.op_str
    if m in ("ldr", "ldr.w") and "[pc" in op:
        try:
            imm = int(op.split("#")[1].rstrip("]"), 0)
            la = (ins.address + 4 + imm) & ~3
            if 0 <= la < len(data) - 4:
                v = struct.unpack_from("<I", data, la)[0]
                kind = ""
                if 0x03010000 <= v < 0x03020000:
                    kind = " SMN"
                elif v < 0x20000:
                    kind = " SRAM"
                elif 0x10000000 <= v < 0x80000000:
                    kind = " BAR0?"
                return f"  -> *0x{la:x} = 0x{v:08x}{kind}"
        except (ValueError, IndexError):
            pass
    return ""


def dump_fn(call_addr: int, label: str):
    start = find_fn_start(call_addr)
    end = find_fn_end(start)
    print(f"\n{'-' * 70}")
    print(f"FUNCTION enclosing 0x{call_addr:x} ({label})")
    print(f"  estimated bounds: 0x{start:x}..0x{end:x}  ({end - start} bytes)")
    print(f"{'-' * 70}")
    for ins in ALL_INSNS:
        if ins.address < start or ins.address >= end:
            continue
        marker = "  >>>" if ins.address == call_addr else "     "
        ann = annotate(ins)
        print(f"  {marker} 0x{ins.address:08x}: {ins.mnemonic:<8} {ins.op_str:<35}{ann}")


# 3 WFI call sites
dump_fn(0x1e10, "WFI #1 in poll-for-state==0xc")
dump_fn(0x1e8c, "WFI #2")
dump_fn(0x2736, "WFI #3 in poll-for-state==0x23")

# Also dump the function called BY the polling functions: 0x6188 callees
# and 0x5d22 (the only known caller of 0x1e3c).
print("\n\n=== Function at 0x5d22 (parent of 0x1e3c poll fn) ===")
dump_fn(0x5d22, "parent caller")
