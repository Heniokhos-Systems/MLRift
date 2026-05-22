#!/usr/bin/env python3
"""
η.3q-13 — decode SOS main mailbox loop and identify post-state-0x32 path.

Building on η.3q-11/12:
  - fn 0x5260 = command dispatcher for cmds 0x90-0xde
  - fn 0x51c4 = OUTER dispatcher wrapper: reads mailbox struct *0x7018,
    permission check, routes to fn 0x57a4 (low cmds) or fn 0x5260 (high)

Open questions:
  1. Who calls fn 0x51c4? (= the SOS main loop)
  2. What does fn 0x57a4 do? (parallel dispatcher for cmds 0..0x8f)
  3. What's at fn 0x178c? (permission check — accesses mailbox struct)
  4. Where does fn 0x4e40 lead? (called on error path with r0=struct)
  5. What code follows 0x5234 (post-dispatch tail)?
  6. Does SOS itself ever WRITE the cmd 0xc2 + state 0x32 into the
     mailbox struct? (= self-dispatch hypothesis)
"""
import struct
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


def disasm_fn(start, label="", max_insns=80, stop_at=None):
    print(f"\n{'=' * 70}")
    print(f"FUNCTION @ 0x{start:x}  {label}")
    print(f"{'=' * 70}")
    if start not in addr_to_idx:
        # try off-by-2
        for d in (-2, 2, -1, 1):
            if start + d in addr_to_idx:
                start = start + d; break
        else:
            print("  not in disasm map"); return
    idx = addr_to_idx[start]
    for ins in all_insns[idx:idx + max_insns]:
        tag = ""
        if "[pc," in ins.op_str:
            try:
                imm = int(ins.op_str.split("[pc,")[1].split("]")[0].strip().lstrip("#"), 0)
                pc_align = (ins.address + 4) & ~3
                lit_addr = pc_align + imm
                if lit_addr + 4 <= len(SOS):
                    lv = int.from_bytes(SOS[lit_addr:lit_addr+4], "little")
                    note = ""
                    if 0x01000000 <= lv < 0x08000000:
                        note = f" SMN base"
                    elif lv < 0x40000:
                        note = " PSP SRAM"
                    elif 0x80000000 <= lv < 0xa0000000:
                        note = " virt-mem (suspicious)"
                    tag = f"  /* lit = 0x{lv:08x}{note} */"
            except: pass
        if ins.mnemonic in ("bl", "blx"):
            tag += "  → CALL"
        if ins.mnemonic == "svc":
            tag += f"  → SVC"
        print(f"  0x{ins.address:08x}: {ins.bytes.hex():<10} {ins.mnemonic:<10} {ins.op_str}{tag}")
        if ins.mnemonic == "pop" and "pc" in ins.op_str:
            print("  --- end ---"); return
        if ins.mnemonic.startswith("bx") and "lr" in ins.op_str:
            print("  --- end ---"); return
        if stop_at and ins.address >= stop_at:
            print(f"  --- stopped at 0x{stop_at:x} ---"); return


def find_callers(fn_start):
    callers = []
    for ins in all_insns:
        if ins.mnemonic not in ("bl", "blx"): continue
        op = ins.op_str.strip()
        if not op.startswith("#"): continue
        try:
            t = int(op.lstrip("#"), 0)
            if t == fn_start or t == fn_start + 1:
                callers.append(ins.address)
        except: pass
    return callers


def find_containing_fn(addr):
    if addr not in addr_to_idx: return None
    idx = addr_to_idx[addr]
    for i in range(idx, max(0, idx - 1500), -1):
        ins = all_insns[i]
        if ins.mnemonic in ("push", "push.w") and "lr" in ins.op_str:
            return ins.address
    return None


# ============================================================
# 1. Find containing fn of 0x51c4 (outer dispatcher wrapper)
#    Note: 0x51c4 ITSELF is push.w {...lr} — so it IS the start
# ============================================================
print("=" * 70)
print("Q1: Who calls fn 0x51c4 (outer dispatcher wrapper)?")
print("=" * 70)
callers_51c4 = find_callers(0x51c4)
print(f"  {len(callers_51c4)} direct callers: {[hex(c) for c in callers_51c4]}")
for c in callers_51c4:
    fn = find_containing_fn(c)
    print(f"    @ 0x{c:x} (in fn @ 0x{fn:x})" if fn else f"    @ 0x{c:x}")

# ============================================================
# 2. Disasm fn 0x57a4 (parallel dispatcher, cmds 0..0x8f)
# ============================================================
disasm_fn(0x57a4, "(parallel dispatcher, cmds 0..0x8f)", max_insns=60)

# ============================================================
# 3. Disasm fn 0x178c (permission check on inbound command)
# ============================================================
disasm_fn(0x178c, "(permission check)", max_insns=40)

# ============================================================
# 4. Disasm fn 0x4e40 (error path with r0=mailbox struct)
# ============================================================
disasm_fn(0x4e40, "(error handler — runs when dispatch fails)", max_insns=40)

# ============================================================
# 5. Disasm fn 0x51c4 forward 0x5234 onward (post-dispatch tail)
# ============================================================
print("\n" + "=" * 70)
print("Q5: fn 0x51c4 — post-dispatch tail (0x5234 onward)")
print("=" * 70)
if 0x5234 in addr_to_idx:
    idx = addr_to_idx[0x5234]
    for ins in all_insns[idx:idx + 60]:
        tag = ""
        if "[pc," in ins.op_str:
            try:
                imm = int(ins.op_str.split("[pc,")[1].split("]")[0].strip().lstrip("#"), 0)
                pc_align = (ins.address + 4) & ~3
                lit_addr = pc_align + imm
                lv = int.from_bytes(SOS[lit_addr:lit_addr+4], "little") if lit_addr+4 <= len(SOS) else 0
                tag = f"  /* lit = 0x{lv:08x} */"
            except: pass
        if ins.mnemonic in ("bl", "blx"):
            tag += "  → CALL"
        print(f"  0x{ins.address:08x}: {ins.bytes.hex():<10} {ins.mnemonic:<10} {ins.op_str}{tag}")
        if ins.mnemonic == "pop" and "pc" in ins.op_str:
            print("  --- end of fn 0x51c4 ---"); break
        if ins.mnemonic.startswith("bx") and "lr" in ins.op_str:
            print("  --- end of fn 0x51c4 ---"); break

# ============================================================
# 6. Self-dispatch hypothesis: search for any code that writes
#    cmd value 0xc2 + state 0x32 into a struct in PSP SRAM
# ============================================================
print("\n" + "=" * 70)
print("Q6: Does any SOS code write IMM 0xc2 or 0x32 to a mailbox struct?")
print("=" * 70)
# Look for "movs rN, #0xc2" followed within 20 insns by store to [rN, #0]
# OR look for stores of #0x32 to any [rN, #4] (would be writing to dword[1])
candidates = []
for i, ins in enumerate(all_insns):
    op = ins.op_str.replace(" ", "")
    if ins.mnemonic in ("movs", "mov.w", "movw") and "#0xc2" in op:
        # check if followed by store within 10 insns
        for j in range(i+1, min(len(all_insns), i+10)):
            next_ins = all_insns[j]
            if next_ins.mnemonic.startswith("str"):
                candidates.append((ins.address, "0xc2 setup", next_ins.address, next_ins.op_str))
                break
    if ins.mnemonic in ("movs", "mov.w", "movw") and ("#0x32" in op and "r" in op[:2]):
        for j in range(i+1, min(len(all_insns), i+10)):
            next_ins = all_insns[j]
            if next_ins.mnemonic.startswith("str") and ("#4]" in next_ins.op_str or ", #4" in next_ins.op_str):
                candidates.append((ins.address, "0x32 setup", next_ins.address, next_ins.op_str))
                break

if candidates:
    for c in candidates:
        print(f"  setup @ 0x{c[0]:x} ({c[1]}) → store @ 0x{c[2]:x}: {c[3]}")
else:
    print("  no obvious self-dispatch construction found")
