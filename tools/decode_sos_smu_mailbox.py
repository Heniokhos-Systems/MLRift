#!/usr/bin/env python3
"""
η.3q-7 — disassemble the 14 SOS plaintext sites that reference the
SMU mailbox base SMN 0x03010000. These are the most likely "SOS
asks SMU to do something during boot" call sites.

For each site, find the literal-pool entry, then disasm the function
prologue backwards to discover the call signature. Print enough context
to see WHICH MP1 message numbers SOS sends.

Also: widen C2PMSG search to ALL 0x032xxxxx aligned constants — to
catch the exception-handler write site even if my offset assumption
was wrong.
"""
import struct
from pathlib import Path
from capstone import Cs, CS_ARCH_ARM, CS_MODE_ARM, CS_MODE_THUMB

OUT_DIR = Path("/home/pantelis/Desktop/Projects/Work/MLRift/captures")
SOS = (OUT_DIR / "sos_subblob.bin").read_bytes()

md_arm   = Cs(CS_ARCH_ARM, CS_MODE_ARM)
md_thumb = Cs(CS_ARCH_ARM, CS_MODE_THUMB)
md_thumb.detail = True

# All 14 sites where 0x03010000 appears as a 4-byte-aligned constant
SITES_MAILBOX_BASE = []
for o in range(0, len(SOS) - 3, 4):
    v = struct.unpack_from("<I", SOS, o)[0]
    if v == 0x03010000:
        SITES_MAILBOX_BASE.append(o)

print(f"Found {len(SITES_MAILBOX_BASE)} aligned MAILBOX_BASE constants in SOS: {[hex(s) for s in SITES_MAILBOX_BASE]}")

# For each constant, the literal pool entry is referenced by an ldr.literal
# instruction within ~256 bytes BEFORE it (typical Thumb-2 literal pool reach).
# Try disasm-backwards from each constant offset.
print()
for lit_off in SITES_MAILBOX_BASE[:6]:
    print(f"\n=== Literal MAILBOX_BASE @ 0x{lit_off:x} ===")
    # Try Thumb disasm starting 0x80 bytes before the literal
    start = max(0, lit_off - 0x80)
    chunk = SOS[start:lit_off + 4]
    insns_thumb = list(md_thumb.disasm(chunk, start))
    print(f"  Thumb-2 disasm from 0x{start:x} (last 16 insns before literal):")
    for ins in insns_thumb[-16:]:
        flag = ""
        if "ldr" in ins.mnemonic and "pc" in ins.op_str:
            flag = "  <<< literal-pool load"
        print(f"    0x{ins.address:08x}: {ins.bytes.hex():<10} {ins.mnemonic:<10} {ins.op_str}{flag}")

    # Show what comes AFTER the constant (next ~32 bytes) — possibly more pool data
    after = SOS[lit_off + 4:lit_off + 0x24]
    print(f"  Pool entries after MAILBOX_BASE:")
    for i in range(0, len(after) - 3, 4):
        v = struct.unpack_from("<I", after, i)[0]
        marker = ""
        if v == 0x0301071c: marker = "  (CTRL reg)"
        elif v == 0x03010200: marker = "  (DATA reg)"
        elif 0x03010000 <= v < 0x03020000: marker = "  (SMU regs range)"
        print(f"    +0x{i:02x}  0x{v:08x}{marker}")


# Widen C2PMSG scan: find ALL 0x032xxxxx aligned constants
print("\n\n" + "=" * 70)
print("Widened C2PMSG range scan — ALL 0x032xxxxx constants (aligned)")
print("=" * 70)
for name, fn in [("sys_drv", "sys_drv_subblob.bin"),
                 ("sos",     "sos_subblob.bin"),
                 ("soc_drv", "soc_drv_subblob.bin"),
                 ("intf_drv","intf_drv_subblob.bin"),
                 ("dbg_drv", "dbg_drv_subblob.bin")]:
    p = OUT_DIR / fn
    if not p.exists(): continue
    data = p.read_bytes()
    hits = {}
    for o in range(0, len(data) - 3, 4):
        v = struct.unpack_from("<I", data, o)[0]
        if 0x03200000 <= v < 0x03200400:  # MP0 register block extent
            hits.setdefault(v, []).append(o)
    if hits:
        print(f"\n  {name}: {len(hits)} unique 0x03200xxx constants")
        for v, offs in sorted(hits.items()):
            # Identify by C2PMSG offset
            byte_off = v - 0x03200000
            dword = byte_off // 4
            if dword >= 0x40 and dword < 0xff:
                cnum = dword - 0x40 if dword < 0xa0 else dword - 0x40
                marker = f"  ≈ C2PMSG_{dword - 0x40}" if 0x40 <= dword < 0x140 else ""
            else:
                marker = ""
            print(f"    0x{v:08x}  ×{len(offs):2d}  @{[hex(o) for o in offs[:3]]}{marker}")
    else:
        print(f"\n  {name}: no 0x032xxxxx constants found")
