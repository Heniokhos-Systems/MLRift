#!/usr/bin/env python3
"""
η.3j-RE-3c — decode the SMN mailbox send/receive helpers and find all
call sites of the WFI poll function 0x1de4.

Discovered protocol:
  Mailbox CONTROL  reg = SMN 0x0301071c  (write 0x80000000 to trigger)
  Mailbox PAYLOAD  reg = SMN 0x03010200  (16 bytes)
  Send helper           = 0x41f8(payload_buf, 0x03010200, 4, ...)
  Recv helper           = 0x1ebc(buf, 0x03010200, 4, ...)

Functions to decode:
  0x41f8 — mailbox SEND (write 4 dwords from buf to SMN 0x03010200)
  0x1ebc — mailbox RECV (read 4 dwords from SMN 0x03010200 into buf)
  0x1d50 — log/dbg helper (called with r0=2, r1=op_code)
  0x1de4 — WFI #1 wrapper (entry point for cmd 0xC)
  0x26e8 — WFI #3 wrapper (entry point for cmd 0x23)

For each, dump body + scan call sites within the full 0x500..0x6D00 range.
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
    in_fn = False
    seen_push = False
    last_return = None
    for ins in ALL_INSNS:
        if ins.address < start:
            continue
        if ins.address >= end:
            break
        # Stop after first complete return chain (don't print past end of fn)
        ann = annotate(ins)
        print(f"  0x{ins.address:08x}: {ins.mnemonic:<8} {ins.op_str:<35}{ann}")
        if ins.mnemonic in ("pop", "pop.w") and "pc" in ins.op_str:
            last_return = ins.address
            # don't stop yet — function may have multiple returns
        elif ins.mnemonic == "bx" and "lr" in ins.op_str:
            last_return = ins.address
        # crude bound: stop after seeing 60 insns
        if ins.address - start > 0xC0:
            break


def callers_of(target: int) -> list[int]:
    out = []
    for ins in ALL_INSNS:
        if ins.mnemonic == "bl":
            try:
                tgt = int(ins.op_str.lstrip("#"), 0)
                if tgt == target:
                    out.append(ins.address)
            except (ValueError, IndexError):
                pass
        elif ins.mnemonic == "b" or ins.mnemonic == "b.w":
            try:
                tgt = int(ins.op_str.lstrip("#"), 0)
                if tgt == target:
                    out.append(ins.address)
            except (ValueError, IndexError):
                pass
    return out


# Decode helper bodies
dump_fn_at(0x41f8, 0x80, "SEND: payload -> SMN mailbox")
dump_fn_at(0x1ebc, 0x80, "RECV: SMN mailbox -> buf")
dump_fn_at(0x1d50, 0x60, "LOG/DBG: state report?")

# Print callers of each WFI wrapper
print(f"\n\n{'=' * 70}")
print("CALLERS")
print(f"{'=' * 70}")

for tgt, label in [
    (0x1de4, "WFI #1 wrapper (cmd 0xC)"),
    (0x26e8, "WFI #3 wrapper (cmd 0x23)"),
    (0x41f8, "mailbox SEND helper"),
    (0x1ebc, "mailbox RECV helper"),
    (0x6188, "WFI primitive"),
    (0x1d50, "LOG/DBG helper"),
]:
    cs = callers_of(tgt)
    print(f"\n  0x{tgt:x}  ({label}):  {len(cs)} callers")
    for c in cs[:30]:
        print(f"    @0x{c:x}")


# For the WFI #1 wrapper, also dump its caller context
print(f"\n\n{'=' * 70}")
print("0x1de4 WFI #1 WRAPPER CALL-SITES — caller context")
print(f"{'=' * 70}")
for caller in callers_of(0x1de4):
    print(f"\n  caller @ 0x{caller:x}:")
    # print 6 insns before and 6 after
    addr_idx = {ins.address: i for i, ins in enumerate(ALL_INSNS)}
    i = addr_idx.get(caller)
    if i is None:
        continue
    for j in range(max(0, i - 6), min(len(ALL_INSNS), i + 7)):
        ins = ALL_INSNS[j]
        marker = "  >>>" if ins.address == caller else "     "
        ann = annotate(ins)
        print(f"  {marker} 0x{ins.address:08x}: {ins.mnemonic:<8} {ins.op_str:<35}{ann}")
