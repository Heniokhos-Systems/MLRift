#!/usr/bin/env python3
"""
η.3q-22 — find which plaintext driver owns code at PSP SRAM 0xcbc.

SOS's common exception logger calls SRAM 0xcbc via `blx ip` (ip=0xcbd Thumb).
SYS_DRV doesn't have a clean function prologue there under any plausible
mapping. So check SOC_DRV, INTF_DRV, DBG_DRV, SPL.

Strategy: for each driver, try multiple (load_addr, body_start) configs.
Disasm at SRAM 0xcbc and quality-check if there's a function prologue
nearby (push.w with lr in first 8 insns).
"""
import struct
from pathlib import Path
from capstone import Cs, CS_ARCH_ARM, CS_MODE_THUMB

md = Cs(CS_ARCH_ARM, CS_MODE_THUMB); md.skipdata = True

BLOBS = [
    ("sys_drv",  "sys_drv_subblob.bin"),
    ("soc_drv",  "soc_drv_subblob.bin"),
    ("intf_drv", "intf_drv_subblob.bin"),
    ("dbg_drv",  "dbg_drv_subblob.bin"),
    ("spl",      "spl_subblob.bin"),
]

# Look for any code that could be at SRAM 0xcbc under various configs
TARGET_SRAM = 0xcbc

# Configs: (load_addr in SRAM, body_start in file)
CONFIGS = [
    (0x0,    0x0,    "body @ file 0"),
    (0x0,    0x100,  "body @ file 0x100"),
    (0x0,    0x400,  "body @ file 0x400"),
    (0x0,    0x1000, "body @ file 0x1000"),
    (0x0,    0x2000, "body @ file 0x2000"),
]


def has_prologue_near(insns):
    """Check first 8 insns for a function prologue (push with lr)."""
    for ins in insns[:8]:
        if ins.mnemonic in ("push", "push.w") and "lr" in ins.op_str:
            return True
    return False


for name, fn in BLOBS:
    p = Path("/home/pantelis/Desktop/Projects/Work/MLRift/captures") / fn
    if not p.exists(): continue
    data = p.read_bytes()
    print(f"\n=== {name} ({len(data)} bytes) ===")
    for load_addr, body_start, label in CONFIGS:
        # Need: SRAM_addr = load_addr + (file_off - body_start)
        # So file_off = (SRAM_addr - load_addr) + body_start
        file_off = (TARGET_SRAM - load_addr) + body_start
        if file_off + 16 > len(data): continue
        if file_off < 0: continue
        # Decode 64 bytes from file_off — also look BACK 16 bytes
        # in case prologue is just before
        look_back = 0x20
        actual_start = max(0, file_off - look_back)
        chunk = data[actual_start:file_off + 64]
        # Decode at SRAM addr (with appropriate offset)
        decode_addr = TARGET_SRAM - (file_off - actual_start)
        insns = list(md.disasm(chunk, decode_addr))
        # Quality filter
        has_proto = has_prologue_near(insns)
        # Also: count "interesting" mnemonics in first 16 insns
        interesting = sum(1 for ins in insns[:16]
                          if ins.mnemonic in ("push", "pop", "bl", "blx", "svc",
                                              "ldr", "str", "mov", "movs", "cmp", "bne", "beq"))
        marker = ""
        if has_proto:
            marker = "  *** PROLOGUE FOUND"
        elif interesting >= 10:
            marker = "  (looks like code, no immediate prologue)"
        print(f"\n  [{label}] file 0x{file_off:x}: {chunk[look_back:look_back+16].hex()}{marker}")
        if has_proto or interesting >= 10:
            # Print 12 insns from the target
            for ins in insns:
                if ins.address < TARGET_SRAM - 0x10: continue
                if ins.address > TARGET_SRAM + 0x30: break
                tag = "  <<< SRAM 0xcbc" if ins.address == TARGET_SRAM else ""
                print(f"    0x{ins.address:08x}: {ins.bytes.hex():<10} {ins.mnemonic:<10} {ins.op_str}{tag}")
