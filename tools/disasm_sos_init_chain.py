#!/usr/bin/env python3
"""
η.3q-10 — Disasm the 4 SOS init functions called pre-state-0x22:
  0x6bd4, 0x4b44, 0x4c18, 0x4c64
plus the function containing the 0x5552 wrapper (state-0x32 source).

For each function:
  - Print prologue + first 60 insns
  - Tag any 32-bit literal that looks like an SMN address
    (0x0xxxxxxx — top byte 0x01-0x07 are AMD SoC15 SMN apertures)
  - Tag any literal in SRAM range (0x00000-0x40000)
  - Highlight bl/blx/svc calls
"""
from pathlib import Path
from capstone import Cs, CS_ARCH_ARM, CS_MODE_THUMB

SOS = Path("/home/pantelis/Desktop/Projects/Work/MLRift/captures/sos_subblob.bin").read_bytes()
md = Cs(CS_ARCH_ARM, CS_MODE_THUMB)
md.skipdata = True

# Decode both plaintext regions
PLAINTEXT_REGIONS = [(0x500, 0x6D00), (0x15140, 0x19180)]
all_insns = []
for start, end in PLAINTEXT_REGIONS:
    all_insns.extend(list(md.disasm(SOS[start:end], start)))
addr_to_idx = {ins.address: i for i, ins in enumerate(all_insns)}

# Resolve PC-relative literal: addr is (PC+4) & ~3 + offset
def resolve_pc_lit(ins):
    """For ldr Rx, [pc, #imm] return the (literal_addr, literal_value)."""
    op = ins.op_str
    if "[pc" not in op: return None
    # parse "rN, [pc, #IMM]"
    try:
        imm_str = op.split("[pc,")[1].split("]")[0].strip().lstrip("#")
        imm = int(imm_str, 0)
    except Exception:
        return None
    # In Thumb-2, PC = current address + 4 (and aligned to 4)
    pc_align = (ins.address + 4) & ~3
    lit_addr = pc_align + imm
    if lit_addr + 4 > len(SOS): return None
    lit_val = int.from_bytes(SOS[lit_addr:lit_addr+4], "little")
    return lit_addr, lit_val


TARGET_FUNCS = [
    (0x6bd4, "init_step_1 (called from 0x2f9a)"),
    (0x4b44, "init_step_2 (called from 0x2f9e)"),
    (0x4c18, "init_step_3 (called from 0x2fa2)"),
    (0x4c64, "init_step_4 (called from 0x2fa6)"),
    (0xf56,  "pre_chain_check (called from 0x2f94)"),
]

for addr, label in TARGET_FUNCS:
    print(f"\n{'=' * 70}")
    print(f"FUNCTION @ 0x{addr:x}: {label}")
    print(f"{'=' * 70}")
    if addr not in addr_to_idx:
        for delta in (0, -2, 2, -1, 1):
            if addr + delta in addr_to_idx:
                addr = addr + delta; break
        else:
            print(f"  not found in disasm; first 64 bytes of SOS at that offset:")
            print(f"  {SOS[addr:addr+64].hex()}")
            continue
    idx = addr_to_idx[addr]
    seen_pop = False
    for ins in all_insns[idx:idx + 80]:
        lit_info = ""
        if "[pc" in ins.op_str:
            r = resolve_pc_lit(ins)
            if r:
                la, lv = r
                tag = ""
                if 0x01000000 <= lv < 0x08000000:
                    tag = f"  ← SMN aperture (top byte 0x{lv >> 24:02x})"
                elif 0x03200000 <= lv < 0x03200400:
                    tag = "  ← MP0 reg!"
                elif 0x03010000 <= lv < 0x03020000:
                    tag = "  ← MP1/SMU reg!"
                elif lv < 0x40000:
                    tag = "  ← PSP SRAM"
                lit_info = f"  /* lit@0x{la:x} = 0x{lv:08x}{tag} */"
        call_info = ""
        if ins.mnemonic in ("bl", "blx"):
            call_info = "  → CALL"
        elif ins.mnemonic == "svc":
            call_info = "  → SVC"
        print(f"  0x{ins.address:08x}: {ins.bytes.hex():<10} {ins.mnemonic:<10} {ins.op_str}{lit_info}{call_info}")
        if ins.mnemonic == "pop" and "pc" in ins.op_str:
            seen_pop = True
            print(f"  --- end of function ---")
            break
        if ins.mnemonic.startswith("bx") and "lr" in ins.op_str:
            seen_pop = True
            print(f"  --- end of function ---")
            break

# ============================================================
# Find the containing function for 0x5552 by walking back to nearest
# push instruction with lr (function prologue).
# ============================================================
print(f"\n\n{'=' * 70}")
print("CONTAINING FUNCTION OF 0x5552 (state-0x32 wrapper) — walk back to prologue")
print(f"{'=' * 70}")
if 0x5552 in addr_to_idx:
    target_idx = addr_to_idx[0x5552]
    # walk back to find push {...lr}
    func_start_idx = None
    for i in range(target_idx, max(0, target_idx - 1000), -1):
        ins = all_insns[i]
        if (ins.mnemonic in ("push", "push.w") and "lr" in ins.op_str):
            func_start_idx = i
            break
    if func_start_idx is not None:
        func_start = all_insns[func_start_idx].address
        print(f"Function containing 0x5552 likely starts at 0x{func_start:x}\n")
        print(f"First 30 insns of this function:")
        for ins in all_insns[func_start_idx:func_start_idx + 30]:
            print(f"  0x{ins.address:08x}: {ins.bytes.hex():<10} {ins.mnemonic:<10} {ins.op_str}")

        # Find callers of func_start: search for bl #<func_start> in all_insns
        callers = []
        for i, ins in enumerate(all_insns):
            if ins.mnemonic in ("bl", "blx") and "#" in ins.op_str:
                try:
                    target = int(ins.op_str.lstrip("#").split(",")[0], 0)
                    if target == func_start or target == func_start + 1:
                        callers.append(ins.address)
                except Exception:
                    pass
        print(f"\nCallers of fn@0x{func_start:x}: {len(callers)} sites: {[hex(c) for c in callers[:30]]}")
