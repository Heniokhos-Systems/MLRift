#!/usr/bin/env python3
"""
λ.5 — Find ALL SVC #0x42 (map_physical) sites in sub-blob #2 R2 region.

For each site:
  - What physical address is being mapped (r0 immediately before svc)?
  - What size (r2 immediately before svc)?
  - What flags / kernel state offset is the result stored to?

Goal: discover if any map_physical call sets up a mapping that covers
virt 0x83f00000-0x83f01000 (the fault page). If so, we'll know what
that virt addr is supposed to point to.
"""
import re
from pathlib import Path
from capstone import Cs, CS_ARCH_ARM, CS_MODE_THUMB

CAP_DIR = Path('/home/pantelis/Desktop/Projects/Work/MLRift/captures')
sf = list(CAP_DIR.glob("subblob_02_*.bin"))[0]
data = sf.read_bytes()

md = Cs(CS_ARCH_ARM, CS_MODE_THUMB); md.skipdata = True
all_insns = list(md.disasm(data, 0))
addr_to_idx = {ins.address: i for i, ins in enumerate(all_insns)}

# Find ALL SVC #0x42 sites (map_physical) and walk back for r0/r1/r2 setup
def find_arg_loads(svc_idx, lookback=15):
    """Walk back from svc_idx looking for the latest immediate or
    PC-relative literal load into r0/r1/r2/r3."""
    regs = {}
    for i in range(svc_idx - 1, max(0, svc_idx - lookback), -1):
        ins = all_insns[i]
        # mov[.w] rN, #imm
        m = re.match(r'^r(\d+),\s*#(.+)$', ins.op_str.replace(' ', ''))
        if m and ins.mnemonic in ("mov", "mov.w", "movs", "movw"):
            reg = int(m.group(1))
            try:
                v = int(m.group(2).split(',')[0], 0)
                if reg not in regs:
                    regs[reg] = ('imm', v)
            except: pass
            continue
        # ldr rN, [pc, #imm]
        if ins.mnemonic in ("ldr", "ldr.w") and "[pc," in ins.op_str:
            try:
                rs = ins.op_str.split(',')[0].strip()
                if not rs.startswith('r'): continue
                reg = int(rs[1:])
                imm = int(ins.op_str.split("[pc,")[1].split("]")[0].lstrip("#").strip(), 0)
                lit_v = ((ins.address + 4) & ~3) + imm
                if lit_v + 4 <= len(data):
                    v = int.from_bytes(data[lit_v:lit_v+4], 'little')
                    if reg not in regs:
                        regs[reg] = ('lit', v)
            except: pass
            continue
        # mov rN, rM (reg copy — propagate)
        m = re.match(r'^r(\d+),\s*r(\d+)$', ins.op_str.replace(' ', ''))
        if m and ins.mnemonic in ("mov", "mov.w"):
            dst = int(m.group(1))
            src = int(m.group(2))
            if dst not in regs and src in regs:
                regs[dst] = regs[src]
    return regs

svc_42_sites = []
for i, ins in enumerate(all_insns):
    if ins.mnemonic == "svc" and ins.op_str.strip() in ("#0x42", "#66"):
        svc_42_sites.append(i)

print(f"Found {len(svc_42_sites)} SVC #0x42 (map_physical) sites in sub-blob #2\n")

# Categorize by physical address mapped
mapping_table = []
for idx in svc_42_sites:
    site = all_insns[idx].address
    regs = find_arg_loads(idx, 20)
    r0 = regs.get(0, ('?', None))
    r1 = regs.get(1, ('?', None))
    r2 = regs.get(2, ('?', None))
    mapping_table.append((site, r0, r1, r2))

# Sort by r0 (phys address)
print(f"{'site':>10}   {'r0 (phys)':>14}  {'r1 (size?)':>14}  {'r2 (flags?)':>14}")
print("-" * 80)
for site, r0, r1, r2 in mapping_table:
    r0_s = f"0x{r0[1]:08x}" if r0[1] is not None else "?"
    r1_s = f"0x{r1[1]:08x}" if r1[1] is not None else "?"
    r2_s = f"0x{r2[1]:08x}" if r2[1] is not None else "?"
    flag = ""
    if r0[1] is not None and 0x83000000 <= r0[1] < 0x84000000:
        flag = "  ★★★ MAPS THE 0x83 RANGE ★★★"
    elif r0[1] == 0x03200000:
        flag = "  (MP0 mapping)"
    elif r0[1] == 0x03010000:
        flag = "  (MP1/SMU mapping)"
    elif r0[1] == 0x03240000:
        flag = "  (NBIO IH mapping)"
    elif r0[1] is not None and 0x80000000 <= r0[1] < 0xa0000000:
        flag = "  (VRAM mapping?)"
    elif r0[1] is not None and r0[1] >= 0x80000000:
        flag = "  (high address)"
    print(f"  0x{site:08x}   {r0_s:>14}  {r1_s:>14}  {r2_s:>14}{flag}")

# Also enumerate unique r0 values
print(f"\nUnique r0 (phys address) values across {len(svc_42_sites)} map_physical calls:")
r0_values = {}
for _, r0, _, _ in mapping_table:
    if r0[1] is not None:
        r0_values.setdefault(r0[1], 0)
        r0_values[r0[1]] += 1
for v, c in sorted(r0_values.items()):
    flag = ""
    if 0x83000000 <= v < 0x84000000:
        flag = "  ★ in fault range"
    elif v == 0x03200000:
        flag = "  (MP0)"
    elif v == 0x03010000:
        flag = "  (MP1/SMU)"
    elif v == 0x03240000:
        flag = "  (NBIO IH)"
    elif 0x80000000 <= v < 0x90000000:
        flag = "  (potential VRAM range)"
    print(f"  0x{v:08x}  ×{c}{flag}")
