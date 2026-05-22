#!/usr/bin/env python3
"""
η.3q-9 — LOCATE the SOS call site that writes DEBUG_STATUS = 0x80320d17.
This is the EXACT location where SOS halts (last state we read externally).

DEBUG_STATUS write helper @ 0x1d50 takes (r0=mode, r1=value, r2=flag).
Mode 1 = write low-16 from r1; mode 2 = write high-16 from r1.

Our external read: 0x80320d17
  → 0x80000000 set (helper always sets bit 31)
  → high 16 = 0x0032   (mode 2 write with r1=0x32)
  → low  16 = 0x0d17   (mode 1 write with r1=0x0d17)

Two call sites had no extractable immediate in the naive walk: 0x5552
and 0x5d08. One of them is the mode-2 r1=0x32 site. Hand-disasm both
deeply (40 insns back) to find:
  - the r1=0x32 / r1=0x0d17 immediates
  - the surrounding control flow (what condition made SOS take this path)
  - what code path leads to this call (= what SOS was DOING when it halted)
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

# Targets: bl 0x1d50 call sites where backwards walk failed.
SUSPECT_SITES = [0x5552, 0x5d08, 0x272e, 0x2770, 0x2fc4, 0xec2]

print("Deep disasm at each bl 0x1d50 caller (40 insns back, 5 forward):\n")
for site in SUSPECT_SITES:
    if site not in addr_to_idx:
        # Find closest known address (in case of off-by-2)
        for delta in (0, -2, 2, -1, 1):
            if site + delta in addr_to_idx:
                site = site + delta
                break
        else:
            print(f"  SKIP {hex(site)} — not in disasm map")
            continue
    idx = addr_to_idx[site]
    start = max(0, idx - 40)
    print(f"=" * 70)
    print(f"=== bl 0x1d50 call site @ 0x{site:04x} ===")
    print(f"=" * 70)
    for ins in all_insns[start:idx + 6]:
        marker = ""
        if ins.address == site:
            marker = "  <<< CALL"
        # highlight r1 / r0 setups
        op = ins.op_str.replace(" ", "")
        if op.startswith("r0,#") or op.startswith("r1,#") or op.startswith("r2,#"):
            marker += "  (arg setup)"
        if "0x32" in op or "0x0d17" in op or "#0x320" in op or "#0xd17" in op:
            marker += "  *** matches readout!"
        print(f"  0x{ins.address:08x}: {ins.bytes.hex():<10} {ins.mnemonic:<10} {ins.op_str}{marker}")
    print()
