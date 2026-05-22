#!/usr/bin/env python3
"""
η.3q-3 — decode the $PS1 PSP firmware header to find load address,
compression type, uncompressed size, and entry point. The header
documents how the blob expands at runtime — possibly explaining
where in virt address space the fault PC 0x0d1102f4 lives.

PSP $PS1 header layout (from PSPReverse community + AMD public docs):
  +0x00  16 bytes  signature (random/per-version)
  +0x10  4 bytes   "$PS1" magic
  +0x14  4 bytes   size_remaining? (body bytes after header)
  +0x18  ... varies. PSPReverse documented header layout:
    +0x0   uint32   id_seed
    +0x4   uint32   ?
    +0x8   uint32   image_size
    +0x10  uint32   image_size minus header
    +0x18  uint32   load_addr (where image loads to)
    +0x1C  uint32   entry_addr
    +0x20  uint32   psp_module_address
    +0x24  uint32   psp_module_size
    +0x40+ ... actual signature + further data
    +0x100 ... compressed body OR ELF / raw blob
"""
import struct, sys
from pathlib import Path

OUT_DIR = Path("/home/pantelis/Desktop/Projects/Work/MLRift/captures")
TARGETS = [
    ("sys_drv", "sys_drv_subblob.bin"),
    ("sos",     "sos_subblob.bin"),
    ("kdb",     "kdb_subblob.bin"),
    ("soc_drv", "soc_drv_subblob.bin"),
    ("intf_drv", "intf_drv_subblob.bin"),
    ("dbg_drv", "dbg_drv_subblob.bin"),
    ("spl",     "spl_subblob.bin"),
]

# PSPReverse documented field offsets within the $PS1 header.
# The $PS1 magic itself lives at +0x10 of our sub-blob; from PSPReverse's
# format perspective, "header start" is the $PS1 magic location.
# All field offsets below are RELATIVE TO THE MAGIC.

HEADER_FIELDS = [
    ("magic",                  0x00, 4),  # $PS1
    ("size_signed",            0x04, 4),  # full image size including signature
    ("encryption_options",     0x08, 4),  # bit fields
    ("unknown_0x0C",           0x0C, 4),
    ("size_uncompressed",      0x14, 4),  # uncompressed body size
    ("compression_options",    0x18, 4),
    ("unknown_0x1C",           0x1C, 4),
    ("size_compressed",        0x20, 4),
    ("compressed_image_offset",0x24, 4),
    ("unknown_0x28",           0x28, 4),
    ("size_uncompressed_2",    0x2C, 4),
    ("load_addr",              0x30, 4),  # virt addr where it loads
    ("rom_size_used",          0x34, 4),
    ("unknown_0x38",           0x38, 4),
    ("unknown_0x3C",           0x3C, 4),
    ("version",                0x40, 4),
    ("entry_point",            0x44, 4),  # virt addr of entry
    ("unknown_0x48",           0x48, 4),
    ("memmap_options",         0x4C, 4),
    ("unknown_0x50",           0x50, 4),
    ("sig_fingerprint",        0x54, 32), # SHA-256 hash maybe
    ("unknown_0x74",           0x74, 4),
    ("unknown_0x78",           0x78, 4),
]


def dump_header(name, path):
    data = path.read_bytes()
    print(f"=== {name}: {path.name} ({len(data)} bytes) ===")
    magic_off = data.find(b"$PS1")
    if magic_off < 0:
        print("  No $PS1 magic")
        return
    print(f"  $PS1 @ offset 0x{magic_off:x}\n")
    for fld, off, sz in HEADER_FIELDS:
        absolute = magic_off + off
        if absolute + sz > len(data):
            continue
        chunk = data[absolute:absolute+sz]
        if sz == 4:
            v = struct.unpack_from("<I", chunk, 0)[0]
            extra = ""
            if fld == "magic":
                extra = f"  (ASCII: '{chunk.decode('latin-1', errors='replace')}')"
            elif 0x01000000 <= v <= 0xFFFFFFFF and v != 0:
                extra = f"  ({v} = {v/1024:.1f} KB)" if v < 0x10000000 else ""
            print(f"  +0x{off:02x}  {fld:30s} = 0x{v:08x}{extra}")
        else:
            hex_str = chunk[:32].hex()
            print(f"  +0x{off:02x}  {fld:30s} = {hex_str}...")
    print()


for name, fn in TARGETS:
    p = OUT_DIR / fn
    if not p.exists():
        print(f"=== {name}: SKIP (not extracted) ===\n")
        continue
    dump_header(name, p)
