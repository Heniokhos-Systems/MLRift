#!/usr/bin/env python3
"""
θ.3-D — decode the callees of SOS kernel main fn (file 0x2ec4 in SYS_DRV).

Callees identified from earlier disasm:
  0x9de0  - called with (r0=ctx_plus_0x54_or_0x74, r1=1)
  0x2bd8  - called before 0x9de0
  0x1668  - memcpy-like (blx)
  0x15e0  - another memcpy-like (blx)

Specifically interested in 0x9de0 — it might be a PT update / MMU mapping
operation (because the I/O dispatcher's loop pattern reminds us of
generic file/handle read code that needs to map pages).

For each callee:
  - Disasm prologue + body
  - Tag literals that look like 0x83x / 0xc000 / 0x0d11x addresses
  - Tag any MCR/MRC (CP15 = MMU/cache operations)
  - Tag any SVCs

Also: search SYS_DRV for ALL writes/reads to PT memory at 0xc000-0xd000.
That's where SOS's MMU page table lives; any code that touches it is
relevant to fault analysis.
"""
import struct
from pathlib import Path
from capstone import Cs, CS_ARCH_ARM, CS_MODE_THUMB

SYS_DRV = Path("/home/pantelis/Desktop/Projects/Work/MLRift/captures/sys_drv_subblob.bin").read_bytes()
md = Cs(CS_ARCH_ARM, CS_MODE_THUMB); md.skipdata = True
all_insns = list(md.disasm(SYS_DRV, 0))
addr_to_idx = {ins.address: i for i, ins in enumerate(all_insns)}


def disasm_fn_to_return(addr, label, max_insns=80):
    print(f"\n{'=' * 70}")
    print(f"FUNCTION @ 0x{addr:x}  {label}")
    print(f"{'=' * 70}")
    if addr not in addr_to_idx:
        for d in (-2, 2, -1, 1):
            if addr + d in addr_to_idx:
                addr = addr + d; break
        else:
            print(f"  not in disasm map"); return
    idx = addr_to_idx[addr]
    for ins in all_insns[idx:idx + max_insns]:
        tag = ""
        op_n = ins.op_str.replace(" ", "")
        if "[pc," in op_n:
            try:
                imm = int(op_n.split("[pc,")[1].split("]")[0].lstrip("#"), 0)
                lit_virt = ((ins.address + 4) & ~3) + imm
                if lit_virt + 4 <= len(SYS_DRV):
                    lv = int.from_bytes(SYS_DRV[lit_virt:lit_virt+4], "little")
                    note = ""
                    if 0x83000000 <= lv < 0x84000000: note = "  ← PSP virt peripheral!"
                    elif 0x0d100000 <= lv < 0x0d200000: note = "  ← SOS virt!"
                    elif 0xc000 <= lv < 0xd000: note = "  ← PT REGION!"
                    elif 0x00d00000 <= lv < 0x00e00000: note = "  ← kernel state"
                    elif lv < 0x40000 and lv > 0: note = "  PSP_SRAM"
                    tag = f"  /* lit = 0x{lv:08x}{note} */"
            except: pass
        if ins.mnemonic in ("bl", "blx"):
            tag += "  → CALL"
        if ins.mnemonic == "svc":
            tag += "  → SVC"
        if ins.mnemonic in ("mcr", "mrc"):
            tag += "  ← CP15 (MMU/cache op!)"
        # Look for stores to address that might be the PT (0xc000+)
        if ins.mnemonic.startswith("str") and "[r" in ins.op_str:
            tag += "  (STORE)"
        print(f"  0x{ins.address:08x}: {ins.bytes.hex():<10} {ins.mnemonic:<10} {ins.op_str}{tag}")
        if ins.mnemonic == "pop" and "pc" in ins.op_str:
            print("  --- end of fn ---"); return
        if ins.mnemonic.startswith("bx") and "lr" in ins.op_str:
            print("  --- end of fn ---"); return


# Decode the 4 main callees
for addr, lbl in [(0x9de0, "called 2x by kernel main"),
                  (0x2bd8, "called between iterations"),
                  (0x1668, "blx memcpy-like #1"),
                  (0x15e0, "blx memcpy-like #2")]:
    disasm_fn_to_return(addr, lbl, 50)

# Now: SEARCH ALL OF SYS_DRV for any code that loads a literal in the PT
# range (0xc000-0xd000). These functions touch the PSP's page table.
print("\n\n" + "=" * 70)
print("ALL SYS_DRV references to PT memory addresses (literals in 0xc000-0xd000)")
print("=" * 70)
pt_refs = {}  # addr → literal
for ins in all_insns:
    if "[pc," in ins.op_str.replace(" ", ""):
        try:
            imm = int(ins.op_str.split("[pc,")[1].split("]")[0].strip().lstrip("#"), 0)
            lit_virt = ((ins.address + 4) & ~3) + imm
            if lit_virt + 4 <= len(SYS_DRV):
                lv = int.from_bytes(SYS_DRV[lit_virt:lit_virt+4], "little")
                if 0xc000 <= lv < 0xd000:
                    pt_refs[ins.address] = lv
        except: pass

print(f"  {len(pt_refs)} sites reference 0xc000-0xd000 literals")
for addr, lv in sorted(pt_refs.items())[:30]:
    # Find the containing function
    fn_start = None
    for s in sorted([a for a in addr_to_idx
                     if all_insns[addr_to_idx[a]].mnemonic in ("push", "push.w")
                     and "lr" in all_insns[addr_to_idx[a]].op_str]):
        if s > addr: break
        fn_start = s
    print(f"  @ 0x{addr:x} (in fn @ 0x{fn_start:x}): lit = 0x{lv:08x}")

# ALSO: search for code that does MCR p15 (cache/MMU operations)
print("\n\n" + "=" * 70)
print("ALL SYS_DRV CP15 operations (MCR/MRC p15) — MMU/cache management")
print("=" * 70)
cp15_sites = []
for ins in all_insns:
    if ins.mnemonic in ("mcr", "mrc") and "p15" in ins.op_str:
        cp15_sites.append((ins.address, ins.mnemonic, ins.op_str))

print(f"  {len(cp15_sites)} CP15 sites in SYS_DRV")
for addr, mn, op in cp15_sites[:30]:
    print(f"  0x{addr:x}: {mn} {op}")
