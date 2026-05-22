#!/usr/bin/env python3
"""
η.3m diagnostic — dump the indirectioaccess data table from /tmp/mlrift_vbios.rom
to see why our _atom_index_iio walker finds zero IIO_START entries.

Reproduces atom_get_data_table_list_base + atom_data_table_offset_by_index(23).
"""
import struct, sys
from pathlib import Path

VBIOS = Path("/tmp/mlrift_vbios.rom")
data = VBIOS.read_bytes()
print(f"VBIOS size: {len(data)} bytes (0x{len(data):x})\n")

# AMD ATOM ROM header is at byte offset 0x48 — it stores u16 pointer
# at 0x48+0x0E to the AtomRomTable (BIOS info / MasterCommand table /
# MasterData table). Inside that table:
#   +0x16 = u16 master command table list base
#   +0x18 = u16 master data table list base
# (See amdgpu/atom.c amdgpu_atom_parse)

rom_header_ptr_off = 0x48
print(f"--- ROM header pointer @ 0x{rom_header_ptr_off:x} ---")
rom_hdr = struct.unpack_from("<H", data, rom_header_ptr_off)[0]
print(f"  rom_header = 0x{rom_hdr:04x}")

# amdgpu uses ctx->cmd_table = CU16(idx + 32-2) and
#               ctx->data_table = CU16(idx + 32) where idx = rom_hdr
master_cmd  = struct.unpack_from("<H", data, rom_hdr + 32 - 2)[0]
master_data = struct.unpack_from("<H", data, rom_hdr + 32)[0]
print(f"  master_command_table @ 0x{master_cmd:04x}")
print(f"  master_data_table    @ 0x{master_data:04x}\n")

# Master data table structure:
#   +0..3 = atom_common_table_header (size, format_rev, content_rev)
#   +4..  = list of u16 offsets, one per data-table index
data_list_base = master_data
hdr = struct.unpack_from("<HBB", data, data_list_base)
print(f"--- master_data_table header @ 0x{data_list_base:x} ---")
print(f"  structuresize = {hdr[0]}, format_rev = {hdr[1]}, content_rev = {hdr[2]}\n")

# What our atom_data_table_offset_by_index(23) returns:
# data_list_base + 4 + 23*2 = data_list_base + 50 = data_list_base + 0x32
# That position holds the u16 offset to indirectioaccess table.
iio_ptr_off = data_list_base + 4 + 23 * 2
iio_tbl_off = struct.unpack_from("<H", data, iio_ptr_off)[0]
print(f"--- indirectioaccess pointer @ 0x{iio_ptr_off:x} (= data_list_base + 0x32) ---")
print(f"  iio_table_offset = 0x{iio_tbl_off:04x}\n")

if iio_tbl_off == 0:
    print("indirectioaccess table NOT PRESENT in this VBIOS (table-pointer is 0).")
    sys.exit(0)

# Dump first 64 bytes of indirectioaccess table.
print(f"--- indirectioaccess table head @ 0x{iio_tbl_off:x} (64 bytes hex+ascii) ---")
for off in range(0, 64, 16):
    chunk = data[iio_tbl_off + off : iio_tbl_off + off + 16]
    hex_str = ' '.join(f'{b:02x}' for b in chunk)
    ascii_str = ''.join(chr(b) if 32 <= b < 127 else '.' for b in chunk)
    print(f"  +{off:04x}: {hex_str:<48} {ascii_str}")

# Parse the table header (atom_common_table_header is 4 bytes)
tbl_hdr = struct.unpack_from("<HBB", data, iio_tbl_off)
print(f"\n  iio table header: structuresize = {tbl_hdr[0]} (0x{tbl_hdr[0]:x})"
      f", format_rev = {tbl_hdr[1]}, content_rev = {tbl_hdr[2]}")

# After 4-byte header, IIO sub-programs start. Each begins with [START][slot_id].
# ATOM_IIO_START = 1
post_hdr_off = iio_tbl_off + 4
first_byte = data[post_hdr_off]
print(f"\n  byte @ iio_tbl + 4 = 0x{first_byte:02x}  "
      f"({'IIO_START!' if first_byte == 1 else 'NOT IIO_START — walker exits immediately'})")

# Try other header sizes as a fallback
for try_hdr in (1, 2, 3, 4, 5, 6, 7, 8):
    b = data[iio_tbl_off + try_hdr]
    marker = "  <<< IIO_START" if b == 1 else ""
    print(f"  byte @ iio_tbl + {try_hdr} = 0x{b:02x}{marker}")

# Walk the table assuming standard 4-byte header; identify all IIO_START + slot pairs
print(f"\n--- attempting full IIO walk from offset 0x{iio_tbl_off + 4:x} ---")

ATOM_IIO_NOP, ATOM_IIO_START, ATOM_IIO_READ, ATOM_IIO_WRITE = 0, 1, 2, 3
ATOM_IIO_CLEAR, ATOM_IIO_SET, ATOM_IIO_MOVE_INDEX = 4, 5, 6
ATOM_IIO_MOVE_ATTR, ATOM_IIO_MOVE_DATA, ATOM_IIO_END = 7, 8, 9
LEN = {0:1, 1:2, 2:3, 3:3, 4:3, 5:3, 6:4, 7:4, 8:4, 9:3}

cur = iio_tbl_off + 4
slot_count = 0
while True:
    if cur >= len(data) or data[cur] != ATOM_IIO_START:
        break
    slot = data[cur + 1]
    print(f"  slot {slot:3d} (= 0x{slot:02x})  body @ 0x{cur+2:04x}", end="")
    cur += 2
    body_start = cur
    body_ops = 0
    while cur < len(data) and data[cur] != ATOM_IIO_END:
        op = data[cur]
        cur += LEN.get(op, 1)
        body_ops += 1
    body_len = cur - body_start
    print(f"  ops={body_ops}  body_len={body_len} bytes")
    cur += 3
    slot_count += 1

print(f"\n  total slots parsed: {slot_count}")
