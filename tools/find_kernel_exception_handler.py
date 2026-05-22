#!/usr/bin/env python3
"""
η.3q-6 — find SYS_DRV / SOC_DRV / INTF_DRV / DBG_DRV code that writes
to PSP-internal MP0 C2PMSG SMN registers. Specifically the kernel
exception handler that wrote C2PMSG_60=4, C2PMSG_61=0x83f00f80,
C2PMSG_62=0x0d1102f4 during our cold-boot fault.

PSP-internal MP0 SMN layout (from η.3j-RE-3):
  base = 0x03200000
  C2PMSG_N register address = 0x03200000 + N*4 + something

Actually the offset within MP0 register block is the DWORD index:
  C2PMSG_35 dword = 0x63   → byte 0x18C → SMN 0x0320018C
  C2PMSG_58 dword = 0x7A   → byte 0x1E8 → SMN 0x032001E8
  C2PMSG_60 dword = 0x7C   → byte 0x1F0 → SMN 0x032001F0
  C2PMSG_61 dword = 0x7D   → byte 0x1F4 → SMN 0x032001F4
  C2PMSG_62 dword = 0x7E   → byte 0x1F8 → SMN 0x032001F8
  C2PMSG_81 dword = 0x91   → byte 0x244 → SMN 0x03200244
  C2PMSG_92 dword = 0x9C   → byte 0x270 → SMN 0x03200270

Looking for any of these constants at 4-byte aligned offsets in
plaintext sub-blobs. The match site is highly likely to be inside
or near the kernel exception handler.
"""
import struct
from pathlib import Path

OUT_DIR = Path("/home/pantelis/Desktop/Projects/Work/MLRift/captures")

C2PMSG_REGS = {
    35: 0x0320018C,
    58: 0x032001E8,
    60: 0x032001F0,
    61: 0x032001F4,
    62: 0x032001F8,
    66: 0x03200208,
    81: 0x03200244,
    89: 0x03200264,
    90: 0x03200268,
    91: 0x0320026C,
    92: 0x03200270,
    115: 0x032002CC,
    118: 0x032002D8,
    126: 0x032002F8,
}

BLOBS = [
    ("sys_drv",  "sys_drv_subblob.bin"),
    ("sos",      "sos_subblob.bin"),
    ("soc_drv",  "soc_drv_subblob.bin"),
    ("intf_drv", "intf_drv_subblob.bin"),
    ("dbg_drv",  "dbg_drv_subblob.bin"),
    ("spl",      "spl_subblob.bin"),
]


def aligned_hits(data, target):
    pat = struct.pack("<I", target & 0xFFFFFFFF)
    out = []
    for o in range(0, len(data) - 3, 4):
        if data[o:o+4] == pat:
            out.append(o)
    return out


print("Searching for PSP-internal C2PMSG SMN constants in plaintext sub-blobs.\n")

for bname, fn in BLOBS:
    p = OUT_DIR / fn
    if not p.exists(): continue
    data = p.read_bytes()
    found_any = False
    block_lines = [f"=== {bname} ===\n"]
    for cnum, addr in C2PMSG_REGS.items():
        hits = aligned_hits(data, addr)
        if hits:
            found_any = True
            marker = ""
            if cnum in (60, 61, 62):
                marker = "  <<< KERNEL EXCEPTION HANDLER WRITES THESE!"
            elif cnum == 81:
                marker = "  <<< SOL register"
            elif cnum == 58:
                marker = "  <<< SOS version register"
            elif cnum == 92:
                marker = "  <<< BOOT_STATUS (steady=0xBA)"
            block_lines.append(f"  C2PMSG_{cnum} = SMN 0x{addr:08x}: {len(hits)}× at {[hex(h) for h in hits[:6]]}{marker}\n")
    # Also search 0x032000d8 (PSP DEBUG STATUS) — RE-3 finding
    for label, smn in [("DEBUG_STATUS_REG", 0x032000D8),
                        ("DEBUG_BASE",      0x03200000),
                        ("MAILBOX_BASE",    0x03010000),
                        ("MAILBOX_CTRL",    0x0301071C),
                        ("MAILBOX_DATA",    0x03010200)]:
        hits = aligned_hits(data, smn)
        if hits:
            found_any = True
            block_lines.append(f"  {label} = SMN 0x{smn:08x}: {len(hits)}× at {[hex(h) for h in hits[:6]]}\n")
    if found_any:
        print("".join(block_lines))
    else:
        print(f"=== {bname} === (no C2PMSG/debug register constants found at 4-byte alignment)\n")
