#!/usr/bin/env python3
"""
η.3q-8 — extract SMU message numbers SOS sends during boot.

Strategy:
  1. Disasm helpers at SOS offsets 0x1d50 and 0x4870 — these look
     like SMU message-send helpers (called immediately after writing
     0x80000000 to mailbox+0x71c, with r1 = message number).
  2. Find every BL/BLX to those helpers in SOS plaintext.
  3. Backtrack from each call site to capture the immediate that
     was loaded into r1 (and r0, r2) — these are the message number
     + args.
  4. Cross-reference against amdgpu's PUBLIC SMU MSG list to find
     PRIVATE/UNDOCUMENTED messages SOS uses.
"""
import struct
from pathlib import Path
from capstone import Cs, CS_ARCH_ARM, CS_MODE_THUMB

SOS = Path("/home/pantelis/Desktop/Projects/Work/MLRift/captures/sos_subblob.bin").read_bytes()
md = Cs(CS_ARCH_ARM, CS_MODE_THUMB)
md.detail = True
md.skipdata = True  # CRITICAL: keep decoding past invalid bytes

# Helper candidates: 0x1d50 (called from 0x2fc4) and 0x4870 (called from 0xef0)
HELPER_CANDIDATES = [0x1d50, 0x4870, 0x5fdc, 0x4a1c]

# SOS plaintext regions (from η.3j RE):
#   R1: 0x500 - 0x6D00  (plaintext code)
#   R2: 0x15140 - 0x19180 (plaintext code)
# Encrypted: 0x7000 - 0x14FFF
PLAINTEXT_REGIONS = [(0x500, 0x6D00), (0x15140, 0x19180)]

# Decode each plaintext region; offsets in `all_insns` keep absolute addrs.
print("Decoding SOS plaintext regions as Thumb-2 (SKIPDATA enabled)...")
all_insns = []
for start, end in PLAINTEXT_REGIONS:
    chunk = SOS[start:end]
    region_insns = list(md.disasm(chunk, start))
    print(f"  region 0x{start:x}-0x{end:x}: {len(region_insns)} insns")
    all_insns.extend(region_insns)
print(f"  total: {len(all_insns)} insns\n")

# Build address → index map for quick lookup
addr_to_idx = {ins.address: i for i, ins in enumerate(all_insns)}

# ============================================================
# Step 1: Show helper prologue at each candidate
# ============================================================
print("=" * 70)
print("HELPER FUNCTION PROLOGUES")
print("=" * 70)
for h in HELPER_CANDIDATES:
    print(f"\n--- Helper @ 0x{h:x} (first 30 insns) ---")
    if h not in addr_to_idx:
        # try +1 (Thumb code addrs sometimes off-by-1)
        print(f"  not found at exact addr, trying nearby")
        for off in (h, h-2, h+2, h-1, h+1):
            if off in addr_to_idx:
                print(f"  found at 0x{off:x}")
                h = off; break
        else:
            continue
    idx = addr_to_idx[h]
    for ins in all_insns[idx:idx+30]:
        # Stop at function epilogue
        flag = ""
        if "ldr" in ins.mnemonic and "pc," in ins.op_str.replace(" ", ""):
            flag = "  <<< literal load"
        if ins.mnemonic == "pop" and "pc" in ins.op_str:
            print(f"  0x{ins.address:08x}: {ins.bytes.hex():<10} {ins.mnemonic:<10} {ins.op_str}{flag}")
            print(f"  --- end of function ---")
            break
        if ins.mnemonic.startswith("bx") and "lr" in ins.op_str:
            print(f"  0x{ins.address:08x}: {ins.bytes.hex():<10} {ins.mnemonic:<10} {ins.op_str}{flag}")
            print(f"  --- end of function ---")
            break
        print(f"  0x{ins.address:08x}: {ins.bytes.hex():<10} {ins.mnemonic:<10} {ins.op_str}{flag}")

# ============================================================
# Step 2: Find all BL/BLX calls to each helper, then backtrack
#         to capture immediate args (r0, r1, r2) before the call.
# ============================================================
print("\n\n" + "=" * 70)
print("ALL BL CALL SITES TO SMU HELPERS — extracting message numbers")
print("=" * 70)

# We'll look for "bl #<target>" insns where target ≈ HELPER_CANDIDATES.
# Then walk backwards ~16 insns to find movs/mov.w into r0/r1/r2.
def extract_imm_setup(insns, call_idx, max_back=16):
    """Walk backwards from call_idx, collect the latest immediate
    set into r0/r1/r2 (and r3 for completeness)."""
    regs = {0: None, 1: None, 2: None, 3: None}
    for i in range(call_idx - 1, max(0, call_idx - max_back), -1):
        ins = insns[i]
        op = ins.op_str.replace(" ", "")
        # match patterns like "r1,#0x22" or "r0,#2"
        if (ins.mnemonic in ("movs", "mov.w", "mov", "movw") and
            op.startswith("r") and "#" in op):
            try:
                reg_str, imm_str = op.split(",", 1)
                if reg_str.startswith("r"):
                    reg = int(reg_str[1:])
                    if reg in regs and regs[reg] is None:
                        imm_str = imm_str.split("#", 1)[1].split(",")[0]
                        v = int(imm_str, 0)
                        regs[reg] = v
            except Exception:
                pass
        # don't go past a branch / return
        if ins.mnemonic in ("pop", "bx", "b.w", "b") and i != call_idx - 1:
            break
    return regs

for helper_addr in HELPER_CANDIDATES:
    # The actual call target in Thumb might appear as bl #<helper_addr>
    callers = []
    for i, ins in enumerate(all_insns):
        if ins.mnemonic in ("bl", "blx"):
            # Parse target: format is "#0x1234" or similar
            opstr = ins.op_str.strip()
            if opstr.startswith("#"):
                try:
                    target = int(opstr.lstrip("#"), 0)
                    if target == helper_addr or target == helper_addr + 1:
                        callers.append(i)
                except Exception:
                    pass
    print(f"\nHelper @ 0x{helper_addr:x}: {len(callers)} call sites")
    if not callers: continue
    for ci in callers[:40]:
        call_ins = all_insns[ci]
        regs = extract_imm_setup(all_insns, ci)
        regs_str = " ".join(
            f"r{r}=0x{v:x}" if v is not None else f"r{r}=?"
            for r, v in regs.items() if v is not None
        )
        print(f"  0x{call_ins.address:08x}: bl helper  | setup: {regs_str}")
