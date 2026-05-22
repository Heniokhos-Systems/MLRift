#!/usr/bin/env python3
"""
λ.4 — decode the new DEBUG_STATUS writers we found in sub-blob #2.

Known sites:
  0x1d86 — helper @ 0x1d50 (already decoded)
  0x2fac — state-0x22 error path (already decoded)
  0x166f2 — NEW (R2 region)
  0x16eec — NEW (R2 region)

For each new site, walk back to function prologue, show containing fn,
identify what's being written to DEBUG_STATUS and from what code path.

Also extract the 12 PSP-base-loading functions in sub-blob #2 — these
are the ONLY plaintext code that talks to MP0 SMN at all. Mapping them
gives us the complete MP0-write surface.
"""
from pathlib import Path
from capstone import Cs, CS_ARCH_ARM, CS_MODE_THUMB

CAP_DIR = Path('/home/pantelis/Desktop/Projects/Work/MLRift/captures')
sf = list(CAP_DIR.glob("subblob_02_*.bin"))[0]
data = sf.read_bytes()

md = Cs(CS_ARCH_ARM, CS_MODE_THUMB); md.skipdata = True
all_insns = list(md.disasm(data, 0))
addr_to_idx = {ins.address: i for i, ins in enumerate(all_insns)}

def find_fn_start(addr):
    if addr not in addr_to_idx: return None
    idx = addr_to_idx[addr]
    for i in range(idx, max(0, idx - 200), -1):
        ins = all_insns[i]
        if ins.mnemonic in ("push", "push.w") and "lr" in ins.op_str:
            return ins.address
    return None

def disasm_fn(start, max_insns=80, highlight_addr=None):
    if start not in addr_to_idx: return
    idx = addr_to_idx[start]
    for ins in all_insns[idx:idx + max_insns]:
        tag = ""
        op_n = ins.op_str.replace(" ", "")
        if "[pc," in op_n:
            try:
                imm = int(op_n.split("[pc,")[1].split("]")[0].lstrip("#"), 0)
                lit_v = ((ins.address + 4) & ~3) + imm
                if lit_v + 4 <= len(data):
                    lv = int.from_bytes(data[lit_v:lit_v+4], 'little')
                    note = ""
                    if 0x83000000 <= lv < 0x84000000: note = "  ← 0x83x"
                    elif 0x03200000 <= lv < 0x03200400: note = "  ← MP0_REG"
                    elif 0x03010000 <= lv < 0x03020000: note = "  ← MP1_SMU"
                    elif 0x00d00000 <= lv < 0x00e00000: note = "  ← kernel state"
                    elif lv < 0x40000 and lv > 0x100: note = "  ← PSP SRAM"
                    tag = f"  /* 0x{lv:08x}{note} */"
            except: pass
        if highlight_addr and ins.address == highlight_addr:
            tag += "  ★★★ THIS LINE"
        if ins.mnemonic in ("bl", "blx"): tag += "  CALL"
        if ins.mnemonic == "svc": tag += "  SVC"
        print(f"  0x{ins.address:08x}: {ins.bytes.hex():<10} {ins.mnemonic:<10} {ins.op_str}{tag}")
        if ins.mnemonic == "pop" and "pc" in ins.op_str:
            print("  --- end ---"); return
        if ins.mnemonic.startswith("bx") and "lr" in ins.op_str:
            print("  --- end ---"); return

# Decode the two NEW DEBUG_STATUS write sites
for target in (0x166f2, 0x16eec):
    fn = find_fn_start(target)
    if fn is None:
        print(f"\n## No containing fn found for 0x{target:x}")
        continue
    print(f"\n{'=' * 80}")
    print(f"## DEBUG_STATUS write @ 0x{target:x}, containing fn @ 0x{fn:x}")
    print(f"## (offset {target - fn:#x} into fn)")
    print(f"{'=' * 80}")
    disasm_fn(fn, 60, highlight_addr=target)

# Extract the 12 PSP-base loading functions
print(f"\n\n{'=' * 80}")
print("ALL 12 sites loading #0x3200000 in sub-blob #2 — complete MP0 plaintext surface")
print(f"{'=' * 80}")
psp_loaders = []
last_fn_start = 0
for ins in all_insns:
    if ins.mnemonic in ("push", "push.w") and "lr" in ins.op_str:
        last_fn_start = ins.address
    if "#0x3200000" in ins.op_str and ins.mnemonic in ("mov.w", "mov", "movw", "movs"):
        psp_loaders.append((ins.address, last_fn_start))

for load_addr, fn_start in psp_loaders:
    role = ""
    if fn_start == 0xdbc: role = "(probably reset / init helper)"
    elif fn_start == 0x1d50: role = "★ DEBUG_STATUS writer (known)"
    elif fn_start == 0x24f4: role = ""
    elif fn_start == 0x2f34: role = "★ state-0x22 path (init chain caller)"
    elif fn_start == 0x2fe0: role = "(near state-0x22 helpers)"
    elif fn_start == 0x3ec0: role = ""
    elif fn_start == 0x4b78: role = "(init chain — snapshot fn)"
    elif fn_start == 0x6a04: role = ""
    elif fn_start == 0x6b68: role = ""
    elif fn_start == 0x6bd4: role = "(init_step_1 — known)"
    print(f"  load @ 0x{load_addr:08x}  fn @ 0x{fn_start:08x}  {role}")
