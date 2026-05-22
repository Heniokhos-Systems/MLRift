#!/usr/bin/env python3
"""
η.3q-11 — find the SOS code path that writes DEBUG_STATUS state 0x32.

Our external readout: 0x80320d17 (bits 16-23 = 0x32).
That value reaches the DEBUG_STATUS helper @ 0x1d50 via r1, in mode 2
(r0=2). The 4 direct callers with immediate r1 don't pass 0x32 — so
0x32 must come through a wrapper-style call site (0x5552 or 0x5d08)
where r1 is passed via register.

Strategy:
  1. Locate the containing function for each wrapper site (walk back
     to nearest `push {...lr}`).
  2. Disasm that containing function fully.
  3. Find all callers (`bl <fn_start>` across both plaintext regions).
  4. At each caller, look backwards for r1 setup — print all of them.
  5. Tag any caller whose r1 setup matches `r1, #0x32`.
"""
from pathlib import Path
from capstone import Cs, CS_ARCH_ARM, CS_MODE_THUMB

SOS = Path("/home/pantelis/Desktop/Projects/Work/MLRift/captures/sos_subblob.bin").read_bytes()
md = Cs(CS_ARCH_ARM, CS_MODE_THUMB)
md.skipdata = True

PLAINTEXT_REGIONS = [(0x500, 0x6D00), (0x15140, 0x19180)]
all_insns = []
for s, e in PLAINTEXT_REGIONS:
    all_insns.extend(list(md.disasm(SOS[s:e], s)))
addr_to_idx = {ins.address: i for i, ins in enumerate(all_insns)}


def find_containing_fn(target_addr):
    """Walk back from target until we hit a push with lr."""
    if target_addr not in addr_to_idx: return None
    idx = addr_to_idx[target_addr]
    for i in range(idx, max(0, idx - 500), -1):
        ins = all_insns[i]
        if ins.mnemonic in ("push", "push.w") and "lr" in ins.op_str:
            return all_insns[i].address
    return None


def disasm_function(fn_start, max_insns=80):
    """Print function from fn_start to first pop with pc / bx lr."""
    if fn_start not in addr_to_idx: return
    idx = addr_to_idx[fn_start]
    print(f"\n--- Function @ 0x{fn_start:x} ---")
    for ins in all_insns[idx:idx + max_insns]:
        marker = ""
        if ins.mnemonic == "bl" and "#0x1d50" in ins.op_str:
            marker = "  <<< calls DEBUG_STATUS writer"
        print(f"  0x{ins.address:08x}: {ins.bytes.hex():<10} {ins.mnemonic:<10} {ins.op_str}{marker}")
        if (ins.mnemonic == "pop" and "pc" in ins.op_str) or \
           (ins.mnemonic.startswith("bx") and "lr" in ins.op_str):
            print("  --- end ---")
            return


def find_callers(fn_start):
    """Find every BL/BLX to fn_start (and fn_start+1 for Thumb)."""
    callers = []
    for i, ins in enumerate(all_insns):
        if ins.mnemonic not in ("bl", "blx"): continue
        opstr = ins.op_str.strip()
        if not opstr.startswith("#"): continue
        try:
            t = int(opstr.lstrip("#"), 0)
            if t == fn_start or t == fn_start + 1:
                callers.append(i)
        except Exception:
            pass
    return callers


def extract_args_at_call(call_idx, lookback=20):
    """Walk backwards, capture last set of r0/r1/r2/r3 immediates."""
    regs = {0: None, 1: None, 2: None, 3: None, 4: None,
            5: None, 6: None, 7: None}
    for i in range(call_idx - 1, max(0, call_idx - lookback), -1):
        ins = all_insns[i]
        op = ins.op_str.replace(" ", "")
        if ins.mnemonic in ("pop", "bx", "b.w", "b") and i != call_idx - 1:
            break
        if ins.mnemonic in ("movs", "mov.w", "movw", "mov") and \
           op.startswith("r") and "#" in op:
            try:
                reg_str, rest = op.split(",", 1)
                reg = int(reg_str[1:])
                imm = int(rest.split("#", 1)[1].split(",")[0], 0)
                if reg in regs and regs[reg] is None:
                    regs[reg] = imm
            except Exception: pass
    return regs


# ====================================================================
print("=" * 70)
print("Finding containing functions for 0x5552 and 0x5d08 wrappers")
print("=" * 70)
for wrapper_addr in (0x5552, 0x5d08):
    fn = find_containing_fn(wrapper_addr)
    if fn is None:
        print(f"\nWrapper @ 0x{wrapper_addr:x}: containing fn NOT FOUND")
        continue
    print(f"\nWrapper @ 0x{wrapper_addr:x}: containing fn starts @ 0x{fn:x}")
    disasm_function(fn, 200)

    callers = find_callers(fn)
    print(f"\n  Callers of fn@0x{fn:x}: {len(callers)} sites")
    state_32_callers = []
    for ci in callers:
        call_addr = all_insns[ci].address
        regs = extract_args_at_call(ci, lookback=25)
        regs_str = " ".join(
            f"r{r}=0x{v:x}" if v is not None else "" for r, v in regs.items()
            if v is not None
        )
        marker = ""
        if regs.get(1) == 0x32:
            marker = "  *** r1=0x32 → STATE-0x32 SOURCE!"
            state_32_callers.append(call_addr)
        print(f"    0x{call_addr:08x}: bl fn  | {regs_str}{marker}")

    if state_32_callers:
        print(f"\n  ** Found {len(state_32_callers)} caller(s) passing r1=0x32:")
        for ca in state_32_callers:
            print(f"     0x{ca:08x}")
