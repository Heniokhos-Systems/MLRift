#!/usr/bin/env python3
"""
η.3q-12 — decode the SOS command dispatcher (fn @ 0x5260).

  1. Read the tbb table @ 0x5298 (length 0x4e bytes).
  2. For each command index i ∈ [0..0x4e], compute handler addr =
     0x5298 + table[i] * 2. Command number sent over mailbox = 0x90 + i.
  3. Identify the index whose handler reaches the DEBUG_STATUS wrapper
     at 0x554a (must trace a few hops since 0x554a isn't a direct target).
  4. Disasm caller @ 0x5224 in context (how mailbox feeds this dispatcher).
"""
from pathlib import Path
from capstone import Cs, CS_ARCH_ARM, CS_MODE_THUMB

SOS = Path("/home/pantelis/Desktop/Projects/Work/MLRift/captures/sos_subblob.bin").read_bytes()
md = Cs(CS_ARCH_ARM, CS_MODE_THUMB)
md.skipdata = True

# Decode tbb table — 78 entries starting at 0x5298
TBB_TABLE_BASE = 0x5298
TBB_TABLE_LEN = 0x4f  # 0..0x4e inclusive
print("=" * 70)
print(f"SOS command dispatcher table (fn 0x5260, tbb @ 0x5294)")
print(f"Table base: 0x{TBB_TABLE_BASE:x}, entries: {TBB_TABLE_LEN}")
print(f"Command number sent over mailbox = 0x90 + index")
print("=" * 70)

handler_to_indices = {}
for i in range(TBB_TABLE_LEN):
    byte = SOS[TBB_TABLE_BASE + i]
    handler = TBB_TABLE_BASE + byte * 2
    handler_to_indices.setdefault(handler, []).append(i)
    cmd = 0x90 + i

# Decode all reachable handlers (one disasm pass)
PLAINTEXT_REGIONS = [(0x500, 0x6D00), (0x15140, 0x19180)]
all_insns = []
for s, e in PLAINTEXT_REGIONS:
    all_insns.extend(list(md.disasm(SOS[s:e], s)))
addr_to_idx = {ins.address: i for i, ins in enumerate(all_insns)}

# For each unique handler, trace up to N insns until we hit a backbranch
# to either 0x5378 (common success tail) or 0x554a (DEBUG_STATUS wrapper)
# or another well-known site.
TAIL_LABELS = {
    0x5378: "common_ret",
    0x554a: "DEBUG_STATUS_wrapper",
    0x5794: "tail_5794",
    0x55e4: "tail_55e4",
    0x5318: "ret_5318 (b 0x5794)",
    0x537c: "tail_537c",
}

def trace_handler(addr, max_hops=80):
    """Walk insns from addr until we hit a known tail or end."""
    if addr not in addr_to_idx: return ("not_disasm", None)
    idx = addr_to_idx[addr]
    for j in range(idx, min(len(all_insns), idx + max_hops)):
        ins = all_insns[j]
        if ins.address in TAIL_LABELS:
            return (TAIL_LABELS[ins.address], ins.address)
        # Direct branch — follow
        if ins.mnemonic in ("b", "b.w"):
            op = ins.op_str.strip().lstrip("#")
            try:
                tgt = int(op, 0)
                if tgt in TAIL_LABELS:
                    return (TAIL_LABELS[tgt], tgt)
            except: pass
        # Stop at function return
        if ins.mnemonic == "pop" and "pc" in ins.op_str:
            return ("local_ret", ins.address)
        if ins.mnemonic.startswith("bx") and "lr" in ins.op_str:
            return ("local_ret", ins.address)
    return ("max_hops", None)


print("\nIndex | Cmd# (mailbox) | Handler @  | First-tail")
print("-" * 70)
for i in range(TBB_TABLE_LEN):
    byte = SOS[TBB_TABLE_BASE + i]
    handler = TBB_TABLE_BASE + byte * 2
    tail, tail_addr = trace_handler(handler)
    marker = ""
    if tail == "DEBUG_STATUS_wrapper":
        marker = "  *** WRITES STATE!"
    cmd = 0x90 + i
    print(f"  0x{i:02x} |   0x{cmd:02x}        | 0x{handler:04x}    | {tail}{marker}")

# Now disasm the caller @ 0x5224
print("\n" + "=" * 70)
print("Caller of fn 0x5260 @ 0x5224 — context (50 insns back)")
print("=" * 70)
if 0x5224 in addr_to_idx:
    idx = addr_to_idx[0x5224]
    start = max(0, idx - 50)
    for ins in all_insns[start:idx + 8]:
        marker = ""
        if ins.address == 0x5224:
            marker = "  <<< CALLS DISPATCHER (fn 0x5260)"
        # Tag any literal-pool load
        if "[pc," in ins.op_str:
            try:
                imm = int(ins.op_str.split("[pc,")[1].split("]")[0].strip().lstrip("#"), 0)
                pc_align = (ins.address + 4) & ~3
                lit_addr = pc_align + imm
                if lit_addr + 4 <= len(SOS):
                    lit_val = int.from_bytes(SOS[lit_addr:lit_addr+4], "little")
                    marker += f"  /* lit = 0x{lit_val:08x} */"
            except: pass
        print(f"  0x{ins.address:08x}: {ins.bytes.hex():<10} {ins.mnemonic:<10} {ins.op_str}{marker}")
