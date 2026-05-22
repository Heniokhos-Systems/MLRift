#!/usr/bin/env python3
"""
θ.3-A — decode SYS_DRV kernel main entry (the function called from
SOS Reset handler via `bx 0x2ee0`).

We established earlier that SYS_DRV file 0x50c4 contains the SVC
dispatcher prologue (perfect match for SOS SVC vector's blx 0x50c5),
which implies SYS_DRV body loads at SRAM 0 (load=0, body=0 mapping).

Under that mapping: SRAM 0x2ee0 = SYS_DRV file 0x2ee0.

But file 0x2ee0 is MID-FUNCTION. The real function start must be
earlier. This tool:
  1. Finds all function starts (push.w with lr) in SYS_DRV
  2. Identifies the function CONTAINING file offset 0x2ee0
  3. Disasms the full function
  4. Highlights:
     - Any literal in 0x83xxxxxx range (PSP virt peripheral)
     - Any literal in 0x0cxxxxxx range (PSP PT region 0xc000)
     - Any SVC call
     - Any access to SOC15-style register addresses
"""
import struct
from pathlib import Path
from capstone import Cs, CS_ARCH_ARM, CS_MODE_THUMB

SYS_DRV = Path("/home/pantelis/Desktop/Projects/Work/MLRift/captures/sys_drv_subblob.bin").read_bytes()
md = Cs(CS_ARCH_ARM, CS_MODE_THUMB); md.skipdata = True

all_insns = list(md.disasm(SYS_DRV, 0))
print(f"SYS_DRV: {len(SYS_DRV)} bytes, {len(all_insns)} insns\n")
addr_to_idx = {ins.address: i for i, ins in enumerate(all_insns)}

# Step 1: Find all function starts
fn_starts = []
for i, ins in enumerate(all_insns):
    if ins.mnemonic in ("push", "push.w") and "lr" in ins.op_str:
        fn_starts.append(ins.address)
print(f"Found {len(fn_starts)} potential function starts in SYS_DRV\n")

# Step 2: Find function containing 0x2ee0
target = 0x2ee0
containing_fn = None
for s in reversed(fn_starts):
    if s <= target:
        containing_fn = s
        break

if containing_fn is None:
    print(f"NO function start found before 0x{target:x}")
else:
    distance = target - containing_fn
    print(f"Function containing 0x{target:x} starts at 0x{containing_fn:x}")
    print(f"  (target is 0x{distance:x} bytes into the function)\n")

# Step 3: Disasm the full function
def disasm_fn(start, max_insns=300):
    print("=" * 70)
    print(f"FUNCTION @ 0x{start:x} (= SRAM 0x{start:x} = SOS-jump-target if start ≤ 0x2ee0)")
    print("=" * 70)
    if start not in addr_to_idx: return
    idx = addr_to_idx[start]
    for ins in all_insns[idx:idx + max_insns]:
        tag = ""
        op_n = ins.op_str.replace(" ", "")

        # Resolve literal pool refs
        if "[pc," in op_n:
            try:
                imm = int(op_n.split("[pc,")[1].split("]")[0].lstrip("#"), 0)
                lit_virt = ((ins.address + 4) & ~3) + imm
                if lit_virt + 4 <= len(SYS_DRV):
                    lv = int.from_bytes(SYS_DRV[lit_virt:lit_virt+4], "little")
                    note = ""
                    if 0x83000000 <= lv < 0x84000000: note = "  ← PSP virt peripheral (0x83x)"
                    elif 0x0c000000 <= lv < 0x0d000000: note = "  ← 0x0cxxxxxx"
                    elif 0x0d000000 <= lv < 0x0e000000: note = "  ← SOS virt range"
                    elif 0x03200000 <= lv < 0x03200400: note = "  ← MP0 reg"
                    elif 0x03010000 <= lv < 0x03020000: note = "  ← MP1/SMU reg"
                    elif 0xc000 <= lv < 0xd000: note = "  ← PT region 0xc000-0xd000"
                    elif 0x9000 <= lv < 0x10000: note = "  ← PSP SRAM stacks 0x9000+"
                    elif 0x00d00000 <= lv < 0x00e00000: note = "  ← kernel state region 0xd0xxxx"
                    elif lv < 0x100000: note = "  ← PSP SRAM"
                    tag = f"  /* lit = 0x{lv:08x}{note} */"
            except: pass

        if ins.address == target:
            tag = f"  <<< SOS reset bx target (0x2ee1, Thumb)" + tag
        if ins.mnemonic in ("bl", "blx") and "#" in ins.op_str:
            tag += "  → CALL"
        if ins.mnemonic == "svc":
            tag += "  → SVC"
        if ins.mnemonic == "mcr" or ins.mnemonic == "mrc":
            tag += "  ← CP15 (MMU/cache)"

        print(f"  0x{ins.address:08x}: {ins.bytes.hex():<10} {ins.mnemonic:<10} {ins.op_str}{tag}")

        # Stop at function end markers
        if ins.mnemonic == "pop" and "pc" in ins.op_str:
            print("  --- end of fn (pop pc) ---")
            return
        if ins.mnemonic.startswith("bx") and "lr" in ins.op_str:
            print("  --- end of fn (bx lr) ---")
            return

if containing_fn is not None:
    disasm_fn(containing_fn, 250)

# Step 4: Walk callees of this function recursively to map kernel main's
# call tree (depth-limited to 2)
print("\n\n" + "=" * 70)
print("Callees of kernel main function (one level deep)")
print("=" * 70)
if containing_fn:
    idx = addr_to_idx[containing_fn]
    callees = set()
    for ins in all_insns[idx:idx + 250]:
        if ins.mnemonic == "pop" and "pc" in ins.op_str: break
        if ins.mnemonic.startswith("bx") and "lr" in ins.op_str: break
        if ins.mnemonic in ("bl", "blx") and ins.op_str.startswith("#"):
            try:
                t = int(ins.op_str.lstrip("#"), 0)
                t = t & ~1  # strip Thumb bit
                callees.add(t)
            except: pass
    print(f"  {len(callees)} unique callees from kernel main")
    for ca in sorted(callees):
        # Find what each callee does (first 5 insns)
        if ca not in addr_to_idx:
            for d in (-2, 2, -1, 1):
                if ca + d in addr_to_idx:
                    ca = ca + d; break
            else: continue
        ci = addr_to_idx[ca]
        snippet = []
        for ins in all_insns[ci:ci+5]:
            snippet.append(f"{ins.mnemonic} {ins.op_str}")
            # Tag any 0x83 / 0xc000 / 0x83f00 literal
            if "[pc," in ins.op_str:
                try:
                    imm = int(ins.op_str.split("[pc,")[1].split("]")[0].strip().lstrip("#"), 0)
                    lv_a = ((ins.address + 4) & ~3) + imm
                    if lv_a + 4 <= len(SYS_DRV):
                        lvv = int.from_bytes(SYS_DRV[lv_a:lv_a+4], "little")
                        snippet[-1] += f" /* lit=0x{lvv:08x} */"
                except: pass
        print(f"\n  fn @ 0x{ca:x}:")
        for s in snippet:
            print(f"     {s}")

# Step 5: Scan SYS_DRV for any direct reference to 0x83f00000 / 0x83f00f80
# (the fault page / fault addr).
print("\n\n" + "=" * 70)
print("Direct refs to fault constants in SYS_DRV (4-byte aligned)")
print("=" * 70)
for tgt, lbl in [(0x83f00f80, "FAULT_ADDR_EXACT"),
                 (0x83f00000, "fault_addr_page"),
                 (0x83000000, "fault_aperture_base"),
                 (0x83e00000, "fault_aperture_alt"),
                 (0x0d1102f4, "FAULT_PC_EXACT"),
                 (0x0d110000, "SOS_virt_base"),
                 (0xc000, "PT_BASE"),
                 (0xc048, "TTBR0_value")]:
    pat = struct.pack("<I", tgt)
    hits = []
    for o in range(0, len(SYS_DRV) - 3, 4):
        if SYS_DRV[o:o+4] == pat:
            hits.append(o)
    if hits:
        print(f"  {lbl} (0x{tgt:08x}): {len(hits)}× @ {[hex(h) for h in hits[:5]]}")

# Also: scan for any 0x83fxxxxx constants at 4-byte alignment
# (might reveal something close to the fault address)
print("\nAll 0x83fxxxxx constants (aligned) in SYS_DRV:")
hits_83f = {}
for o in range(0, len(SYS_DRV) - 3, 4):
    v = int.from_bytes(SYS_DRV[o:o+4], "little")
    if 0x83f00000 <= v < 0x84000000:
        hits_83f.setdefault(v, []).append(o)
for v, offs in sorted(hits_83f.items())[:20]:
    print(f"  0x{v:08x}: {len(offs)}× @ {[hex(o) for o in offs[:3]]}")
