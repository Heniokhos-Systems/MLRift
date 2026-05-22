#!/usr/bin/env python3
"""
η.3j-RE-3d — decode the remaining mailbox helper functions.

  0x1dac  uses SMN 0x03010400 — likely COMMAND CODE writer
  0x65cc  called from end of 0x41f8 with arg 0x10 — likely "finalize/wait"
  0x4d34  called from 0x41f8 and tail-called from 0x1f02 — likely state-set
  0x4db4  called from 0x1ebc with arg 1 — likely "init send"

Also decode the FULL parent of the WFI #1 wrapper (0x5c..0x5e range)
and find what command code is in r0 when WFI #1 wrapper is called.
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


ALL_INSNS = robust_disasm(0x500, 0x6D00)


def annotate(ins):
    if ins.mnemonic in ("ldr", "ldr.w") and "[pc" in ins.op_str:
        try:
            imm = int(ins.op_str.split("#")[1].rstrip("]"), 0)
            la = (ins.address + 4 + imm) & ~3
            if 0 <= la < len(data) - 4:
                v = struct.unpack_from("<I", data, la)[0]
                kind = ""
                if 0x03010000 <= v < 0x03020000:
                    kind = " SMN"
                elif 0x03200000 <= v < 0x03300000:
                    kind = " SMN_DBG"
                elif v < 0x20000:
                    kind = " SRAM"
                elif 0x10000000 <= v < 0x80000000:
                    kind = " BAR0?"
                return f"  -> *0x{la:x} = 0x{v:08x}{kind}"
        except (ValueError, IndexError):
            pass
    return ""


def dump_fn_at(start: int, max_bytes: int, label: str):
    print(f"\n{'-' * 70}")
    print(f"FUNCTION 0x{start:x}  ({label})")
    print(f"{'-' * 70}")
    end = start + max_bytes
    for ins in ALL_INSNS:
        if ins.address < start:
            continue
        if ins.address >= end:
            break
        ann = annotate(ins)
        print(f"  0x{ins.address:08x}: {ins.mnemonic:<8} {ins.op_str:<35}{ann}")
        if ins.mnemonic in ("pop", "pop.w") and "pc" in ins.op_str:
            break
        if ins.mnemonic == "bx" and "lr" in ins.op_str:
            break


# Decode the trigger/state helpers
dump_fn_at(0x1dac, 0x40, "uses SMN 0x03010400 (cmd reg?)")
dump_fn_at(0x65cc, 0x40, "called from end of 0x41f8 with r0=0x10")
dump_fn_at(0x4d34, 0x40, "called with r0=0 from 0x41f8, r0=3 from 0x1ebc")
dump_fn_at(0x4db4, 0x40, "called with r0=1 from 0x1ebc")


# Walk full parent of WFI #1 to find what r0,r1,r2 are at the bl 0x1de4 site
print(f"\n\n{'=' * 70}")
print("PARENT OF WFI #1 — broader context")
print(f"{'=' * 70}")

# Find function containing 0x5dc2 (the BL to 0x1de4)
target = 0x5dc2
addr_idx = {ins.address: i for i, ins in enumerate(ALL_INSNS)}
i = addr_idx.get(target)
# Scan backwards for prologue
fn_start = target
for j in range(i, max(i - 400, 0), -1):
    if ALL_INSNS[j].mnemonic in ("push", "push.w") and "lr" in ALL_INSNS[j].op_str:
        fn_start = ALL_INSNS[j].address
        break
print(f"\nFunction containing 0x{target:x} starts at 0x{fn_start:x}")

# Print 30 insns before the BL to see what sets up registers
print(f"\n  --- 30 insns leading up to 0x{target:x}:")
for j in range(max(0, i - 30), i + 2):
    ins = ALL_INSNS[j]
    marker = "  >>>" if ins.address == target else "     "
    ann = annotate(ins)
    print(f"  {marker} 0x{ins.address:08x}: {ins.mnemonic:<8} {ins.op_str:<35}{ann}")


# Also dump the function's entry to see args
print(f"\n  --- Function entry 0x{fn_start:x}..+30 insns:")
i0 = addr_idx[fn_start]
for j in range(i0, min(i0 + 30, len(ALL_INSNS))):
    ins = ALL_INSNS[j]
    ann = annotate(ins)
    print(f"      0x{ins.address:08x}: {ins.mnemonic:<8} {ins.op_str:<35}{ann}")
