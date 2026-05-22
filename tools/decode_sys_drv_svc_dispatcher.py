#!/usr/bin/env python3
"""
η.3q-19 — decode SYS_DRV SVC dispatcher.

SYS_DRV file 0x50c4 (= PSP SRAM 0x50c4) is the entry called from
SOS's SVC handler via `blx r4` (r4 = 0x50c5 Thumb). The previous
disasm showed a clean function prologue there.

Tasks:
  1. Walk back from 0x50c4 to find the actual function START (in
     case 0x50c4 is mid-function and the entry is earlier).
  2. Disasm full SVC dispatcher body.
  3. Identify the SVC dispatch mechanism (tbb/tbh/jmp-table/switch).
  4. Find SVC #0x4d's handler.
  5. Verify by searching SOS plaintext for the matching SVC #0x4d
     issuer pattern.
"""
from pathlib import Path
from capstone import Cs, CS_ARCH_ARM, CS_MODE_THUMB

SYS_DRV = Path("/home/pantelis/Desktop/Projects/Work/MLRift/captures/sys_drv_subblob.bin").read_bytes()
md = Cs(CS_ARCH_ARM, CS_MODE_THUMB); md.skipdata = True

print(f"SYS_DRV: {len(SYS_DRV)} bytes\n")

# Decode the entire SYS_DRV as Thumb (skipdata enabled)
all_insns = list(md.disasm(SYS_DRV, 0))
print(f"  decoded {len(all_insns)} insns\n")
addr_to_idx = {ins.address: i for i, ins in enumerate(all_insns)}

# 1. Find the function containing 0x50c4 (walk back to nearest push w/ lr)
target = 0x50c4
if target not in addr_to_idx:
    for d in (-2, 2, -1, 1):
        if target + d in addr_to_idx:
            target = target + d; break

idx = addr_to_idx[target]
fn_start = None
for i in range(idx, max(0, idx - 50), -1):
    ins = all_insns[i]
    if ins.mnemonic in ("push", "push.w") and "lr" in ins.op_str:
        fn_start = ins.address
        break

fn_start_addr = fn_start if fn_start is not None else target
print(f"SVC dispatcher entry @ 0x{target:x}, function starts @ 0x{fn_start_addr:x}")
print(f"  (offset within function: 0x{target - fn_start_addr:x})\n")

# 2. Disasm full body starting from fn_start
print("=" * 70)
print(f"SVC dispatcher disasm — {200} insns from 0x{fn_start or target:x}")
print("=" * 70)
start = addr_to_idx.get(fn_start or target, idx)
seen_push = False
brace = 0
for ins in all_insns[start:start + 200]:
    tag = ""
    op_n = ins.op_str.replace(" ", "")
    # Literal-pool resolver
    if "[pc," in op_n:
        try:
            imm = int(op_n.split("[pc,")[1].split("]")[0].lstrip("#"), 0)
            lit_virt = ((ins.address + 4) & ~3) + imm
            if lit_virt + 4 <= len(SYS_DRV):
                lv = int.from_bytes(SYS_DRV[lit_virt:lit_virt+4], "little")
                note = ""
                if 0x03200000 <= lv < 0x03200400: note = "  MP0_REG"
                elif lv == 0x032000d8: note = "  DEBUG_STATUS"
                elif 0x03010000 <= lv < 0x03020000: note = "  MP1_SMU"
                elif lv < 0x40000: note = "  PSP_SRAM"
                tag = f"  /* lit @ 0x{lit_virt:x} = 0x{lv:08x}{note} */"
        except: pass
    if ins.mnemonic in ("tbb", "tbh"):
        tag += "  *** DISPATCH TABLE BRANCH"
    if ins.mnemonic in ("bl", "blx") and ins.op_str.startswith("#"):
        tag += "  → CALL"
    if ins.mnemonic == "svc":
        tag += "  → SVC TRAP"
    if "#0x4d" in op_n:
        tag += "  *** mentions 0x4d"
    # Mark entry point
    if ins.address == target:
        tag += "  <<< entry from SOS SVC vector"
    print(f"  0x{ins.address:08x}: {ins.bytes.hex():<10} {ins.mnemonic:<10} {ins.op_str}{tag}")
    if ins.mnemonic in ("pop", "pop.w") and "pc" in ins.op_str:
        print("  --- end of fn ---")
        break

# 3. Search SOS plaintext for SVC #0x4d issuers (sanity check)
print("\n" + "=" * 70)
print("All SVC #0x4d sites across SYS_DRV + SOS (should appear in exception logger)")
print("=" * 70)
SOS_BLOB = Path("/home/pantelis/Desktop/Projects/Work/MLRift/captures/sos_subblob.bin").read_bytes()

def find_svc_sites(blob, mode_label, regions):
    md_local = Cs(CS_ARCH_ARM, CS_MODE_THUMB); md_local.skipdata = True
    insns_loc = []
    for s, e in regions:
        insns_loc.extend(list(md_local.disasm(blob[s:e], s)))
    hits = []
    for ins in insns_loc:
        if ins.mnemonic == "svc":
            try:
                num = int(ins.op_str.lstrip("#"), 0)
                if num == 0x4d:
                    hits.append(ins.address)
            except: pass
    print(f"  {mode_label}: {len(hits)} SVC #0x4d sites: {[hex(h) for h in hits[:10]]}")

find_svc_sites(SYS_DRV, "SYS_DRV", [(0, len(SYS_DRV))])
find_svc_sites(SOS_BLOB, "SOS R1",  [(0x100, 0x6D00)])
find_svc_sites(SOS_BLOB, "SOS R2",  [(0x15140, 0x19180)])
