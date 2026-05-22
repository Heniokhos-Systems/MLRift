#!/usr/bin/env python3
"""
η.3q-21 — disasm all 13 SVC #0x4d call sites in SYS_DRV.

The exception logger fn that SOS calls (via `blx 0xcbd` at SOS virt
0x0d11028c) issues SVC #0x4d to escalate the fault to a kernel-mode
handler. One of the 13 SYS_DRV SVC #0x4d sites should match the
signature:
  receives (r0=type [1/2/3], r1=PC) from SOS handler
  → does some processing
  → svc #0x4d
  → returns

For each site, walk back to nearest push to find the containing fn,
then disasm prologue + body. Tag any literal that looks like an MP0
SMN address (would be the C2PMSG-write target).
"""
from pathlib import Path
from capstone import Cs, CS_ARCH_ARM, CS_MODE_THUMB

SYS_DRV = Path("/home/pantelis/Desktop/Projects/Work/MLRift/captures/sys_drv_subblob.bin").read_bytes()
md = Cs(CS_ARCH_ARM, CS_MODE_THUMB); md.skipdata = True

all_insns = list(md.disasm(SYS_DRV, 0))
addr_to_idx = {ins.address: i for i, ins in enumerate(all_insns)}

# Re-find all SVC #0x4d sites with their actual addresses
svc_sites = []
for i, ins in enumerate(all_insns):
    if ins.mnemonic == "svc":
        try:
            num = int(ins.op_str.lstrip("#"), 0)
            if num == 0x4d:
                svc_sites.append((ins.address, i))
        except: pass

print(f"Found {len(svc_sites)} SVC #0x4d sites in SYS_DRV\n")


def find_fn_start(idx):
    for i in range(idx, max(0, idx - 100), -1):
        ins = all_insns[i]
        if ins.mnemonic in ("push", "push.w") and "lr" in ins.op_str:
            return ins.address, i
    return None, None


def has_mp0_reg_in_fn(start_idx, max_insns=80):
    """Check if function from start_idx loads any MP0 SMN address."""
    found = []
    for ins in all_insns[start_idx:start_idx + max_insns]:
        if "[pc," in ins.op_str:
            try:
                imm = int(ins.op_str.split("[pc,")[1].split("]")[0].strip().lstrip("#"), 0)
                lit_virt = ((ins.address + 4) & ~3) + imm
                if lit_virt + 4 <= len(SYS_DRV):
                    lv = int.from_bytes(SYS_DRV[lit_virt:lit_virt+4], "little")
                    if 0x03200000 <= lv < 0x03200400:
                        found.append((ins.address, lit_virt, lv))
            except: pass
    return found


for svc_addr, svc_idx in svc_sites:
    fn_start, fn_idx = find_fn_start(svc_idx)
    if fn_start is None:
        print(f"\n--- SVC #0x4d @ 0x{svc_addr:x} (containing fn not found) ---")
        continue
    mp0_refs = has_mp0_reg_in_fn(fn_idx, 80)
    mp0_summary = f"  [{len(mp0_refs)} MP0_REG refs]" if mp0_refs else "  [no MP0_REG refs]"
    print(f"\n--- SVC #0x4d @ 0x{svc_addr:x}, fn starts @ 0x{fn_start:x} (offset 0x{svc_addr - fn_start:x}){mp0_summary} ---")
    # Print fn prologue + svc site context
    print(f"  Function prologue + first 25 insns:")
    for ins in all_insns[fn_idx:min(fn_idx + 25, svc_idx + 5)]:
        tag = ""
        op_n = ins.op_str.replace(" ", "")
        if "[pc," in op_n:
            try:
                imm = int(op_n.split("[pc,")[1].split("]")[0].lstrip("#"), 0)
                lit_virt = ((ins.address + 4) & ~3) + imm
                if lit_virt + 4 <= len(SYS_DRV):
                    lv = int.from_bytes(SYS_DRV[lit_virt:lit_virt+4], "little")
                    note = ""
                    if 0x03200000 <= lv < 0x03200400: note = " MP0_REG"
                    elif 0x03010000 <= lv < 0x03020000: note = " MP1_SMU"
                    elif lv < 0x40000: note = " PSP_SRAM"
                    tag = f"  /* lit = 0x{lv:08x}{note} */"
            except: pass
        if ins.mnemonic == "svc":
            tag += "  <<< SVC #0x4d"
        if ins.address == fn_start:
            tag += "  <<< FN START"
        # Mark r0/r1/r2/r3 arg setups
        if ins.mnemonic in ("movs", "mov.w", "mov") and op_n.startswith(("r0,#", "r1,#", "r2,#", "r3,#")):
            tag += "  (arg setup)"
        print(f"    0x{ins.address:08x}: {ins.bytes.hex():<10} {ins.mnemonic:<10} {ins.op_str}{tag}")
