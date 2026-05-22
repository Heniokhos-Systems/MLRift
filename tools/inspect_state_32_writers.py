#!/usr/bin/env python3
"""
η.3q-14 — inspect the 3 SOS sites that write 0x32 to [rN, #4]:
  - 0x2fe2 → store at 0x2ff4
  - 0x3d76 → store at 0x3d8e
  - 0x3e1e → store at 0x3e34

For each, disasm 40 insns of context to determine:
  a. What struct rN points to (mailbox struct *0x7018, or local stack, or other?)
  b. What other dwords are written to the same struct (dword[0]=mode, etc.)
  c. Is there a subsequent SVC or BL that DISPATCHES the constructed message?

Plus: search ALL plaintext for stores of cmd value 0xc2 (decimal 194)
followed by dispatcher invocation pattern.
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


def disasm_window(center_addr, before=40, after=10, label=""):
    print(f"\n{'=' * 70}")
    print(f"CONTEXT around 0x{center_addr:x}  {label}")
    print(f"{'=' * 70}")
    if center_addr not in addr_to_idx:
        for d in (-2, 2, -1, 1):
            if center_addr + d in addr_to_idx:
                center_addr = center_addr + d; break
        else:
            print("  not in disasm"); return
    idx = addr_to_idx[center_addr]
    start = max(0, idx - before)
    for ins in all_insns[start:idx + after]:
        tag = ""
        if "[pc," in ins.op_str:
            try:
                imm = int(ins.op_str.split("[pc,")[1].split("]")[0].strip().lstrip("#"), 0)
                pc_align = (ins.address + 4) & ~3
                lit_addr = pc_align + imm
                if lit_addr + 4 <= len(SOS):
                    lv = int.from_bytes(SOS[lit_addr:lit_addr+4], "little")
                    note = ""
                    if lv == 0x00007018:
                        note = "  ← MAILBOX STRUCT PTR ADDR"
                    elif lv == 0x00007040:
                        note = "  ← OTHER SRAM (in 0x51c4 too)"
                    elif lv == 0x0000702c:
                        note = "  ← SRAM struct"
                    elif lv == 0x03010000:
                        note = "  ← MP1/SMU mailbox base"
                    elif lv == 0x03200000 or lv == 0x032000d8:
                        note = "  ← MP0/PSP base"
                    elif lv < 0x40000:
                        note = "  ← PSP SRAM"
                    tag = f"  /* lit = 0x{lv:08x}{note} */"
            except: pass
        if ins.mnemonic in ("bl", "blx"):
            tag += "  → CALL"
        if "#0x32" in ins.op_str.replace(" ", "") or "#0x32," in ins.op_str:
            tag += "  *** 0x32 IMMEDIATE"
        if "#0xc2" in ins.op_str:
            tag += "  *** 0xc2 IMMEDIATE"
        if ins.address == center_addr:
            tag += "  <<< STORE OF 0x32"
        print(f"  0x{ins.address:08x}: {ins.bytes.hex():<10} {ins.mnemonic:<10} {ins.op_str}{tag}")


# Inspect each candidate
disasm_window(0x2ff4, before=30, after=5, label="(0x2fe2 = movs 0x32; 0x2ff4 = str)")
disasm_window(0x3d8e, before=30, after=5, label="(0x3d76 = movs 0x32; 0x3d8e = str)")
disasm_window(0x3e34, before=30, after=5, label="(0x3e1e = movs 0x32; 0x3e34 = str)")

# Now search for ALL stores of 0xc2 (movs Rn, #0xc2 → str Rn ...) ANYWHERE
print("\n\n" + "=" * 70)
print("All sites where 0xc2 is moved to a register (likely cmd id or just data)")
print("=" * 70)
for i, ins in enumerate(all_insns):
    op = ins.op_str.replace(" ", "")
    if ins.mnemonic in ("movs", "mov.w", "movw") and "#0xc2" in op:
        # check next 6 insns for a STR or BL
        followups = []
        for j in range(i+1, min(len(all_insns), i+8)):
            nxt = all_insns[j]
            if nxt.mnemonic.startswith("str") or nxt.mnemonic in ("bl", "blx"):
                followups.append(f"0x{nxt.address:x}:{nxt.mnemonic} {nxt.op_str}")
                if len(followups) >= 3: break
        print(f"  0x{ins.address:08x}: {ins.mnemonic} {ins.op_str}  → followups: {followups}")
