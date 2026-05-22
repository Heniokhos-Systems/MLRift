#!/usr/bin/env python3
"""
η.3o-1 — dump VRAM_USAGE_BY_FIRMWARE data table (idx 11) from /tmp/mlrift_vbios.rom.

This is the VBIOS-declared list of VRAM regions reserved for firmware.
On a real boot, amdgpu reads this and reserves matching regions in VRAM
before letting userspace allocate. SOS firmware presumably expects these
regions to be PRE-EXISTING when it starts up — and may probe them by
accessing them. If our SOS crash at virt 0x83f00f80 is SOS trying to
access one of these regions, this table will tell us where.

ATOM_VRAM_OPERATION_FLAGS_MASK is the high bits of start_address_in_kb
that encode the operation type (e.g. SR-IOV reservation marker).
"""
import struct
from pathlib import Path

VBIOS = Path("/tmp/mlrift_vbios.rom")
data = VBIOS.read_bytes()

# Replay the atom_get_data_table_list_base flow.
rom_hdr = struct.unpack_from("<H", data, 0x48)[0]
master_data = struct.unpack_from("<H", data, rom_hdr + 32)[0]

# vram_usagebyfirmware is data-table idx 11.
# Byte offset within master_data: 4 (hdr) + 11*2 = 26 = 0x1A
ptr_off = master_data + 4 + 11 * 2
tbl_off = struct.unpack_from("<H", data, ptr_off)[0]
print(f"vram_usagebyfirmware @ data table idx 11")
print(f"  ptr at master_data + 0x1a = 0x{ptr_off:x}")
print(f"  tbl_off                  = 0x{tbl_off:04x}\n")

if tbl_off == 0:
    print("Table not present.")
    raise SystemExit(0)

# Print table header
size, fmt_rev, content_rev = struct.unpack_from("<HBB", data, tbl_off)
print(f"  structuresize = {size} (0x{size:x})")
print(f"  format_rev    = {fmt_rev}")
print(f"  content_rev   = {content_rev}\n")

# Dump first 64 bytes
print("  raw hex (first 64 bytes):")
for off in range(0, min(64, size), 16):
    chunk = data[tbl_off + off : tbl_off + off + 16]
    hex_str = ' '.join(f'{b:02x}' for b in chunk)
    ascii_str = ''.join(chr(b) if 32 <= b < 127 else '.' for b in chunk)
    print(f"    +{off:04x}: {hex_str:<48} {ascii_str}")

# Decode based on rev
ATOM_VRAM_OPERATION_FLAGS_SHIFT = 30
ATOM_VRAM_OPERATION_FLAGS_MASK  = (3 << ATOM_VRAM_OPERATION_FLAGS_SHIFT)
ATOM_VRAM_BLOCK_SRIOV_MSG_SHARE_RESERVATION = 1
ATOM_VRAM_BLOCK_NEEDS_NO_RESERVATION = 2
ATOM_VRAM_BLOCK_PRIVATE_DRIVER_RESERVATION = 3

def decode_addr(start_addr_kb):
    op = (start_addr_kb >> ATOM_VRAM_OPERATION_FLAGS_SHIFT) & 3
    addr_kb = start_addr_kb & ~ATOM_VRAM_OPERATION_FLAGS_MASK
    addr_b  = addr_kb << 10
    op_name = {
        0: "no_flag (relative VRAM offset)",
        1: "SRIOV_MSG_SHARE_RESERVATION",
        2: "NEEDS_NO_RESERVATION (don't reserve)",
        3: "PRIVATE_DRIVER_RESERVATION",
    }.get(op, f"unknown_{op}")
    return addr_kb, addr_b, op, op_name

print()
if fmt_rev == 2 and content_rev == 1:
    start_kb, fw_kb, drv_kb = struct.unpack_from("<IHH", data, tbl_off + 4)
    addr_kb, addr_b, op, op_name = decode_addr(start_kb)
    print(f"  === vram_usagebyfirmware_v2_1 ===")
    print(f"  start_address_in_kb (raw) = 0x{start_kb:08x}")
    print(f"    operation flags = {op} ({op_name})")
    print(f"    addr            = {addr_kb} KB = 0x{addr_b:x} bytes ({addr_b/(1024*1024):.1f} MiB)")
    print(f"  used_by_firmware_in_kb    = {fw_kb} KB = 0x{fw_kb<<10:x} bytes")
    print(f"  used_by_driver_in_kb      = {drv_kb} KB = 0x{drv_kb<<10:x} bytes")
    # show the fw region span
    print(f"\n  FW region span: 0x{addr_b:x}..0x{addr_b + (fw_kb<<10):x}")
    print(f"                  ({addr_b/(1024**3):.3f} GiB .. {(addr_b + (fw_kb<<10))/(1024**3):.3f} GiB)")
elif fmt_rev == 2 and content_rev == 2:
    fw_start_kb, fw_kb, _rsv, drv_start_kb, drv_kb = \
        struct.unpack_from("<IHHII", data, tbl_off + 4)
    fw_addr_kb, fw_addr_b, op_f, opn_f = decode_addr(fw_start_kb)
    dr_addr_kb, dr_addr_b, op_d, opn_d = decode_addr(drv_start_kb)
    print(f"  === vram_usagebyfirmware_v2_2 ===")
    print(f"  fw_region_start_in_kb (raw)        = 0x{fw_start_kb:08x}")
    print(f"    operation flags = {op_f} ({opn_f})")
    print(f"    addr            = {fw_addr_kb} KB = 0x{fw_addr_b:x} bytes ({fw_addr_b/(1024**3):.3f} GiB)")
    print(f"  used_by_firmware_in_kb              = {fw_kb} KB = 0x{fw_kb<<10:x} bytes ({fw_kb/1024:.1f} MiB)")
    print(f"  driver_region0_start_in_kb (raw)   = 0x{drv_start_kb:08x}")
    print(f"    operation flags = {op_d} ({opn_d})")
    print(f"    addr            = {dr_addr_kb} KB = 0x{dr_addr_b:x} bytes ({dr_addr_b/(1024**3):.3f} GiB)")
    print(f"  used_by_driver_region0_in_kb        = {drv_kb} KB = 0x{drv_kb<<10:x} bytes ({drv_kb/1024:.1f} MiB)")
    print(f"\n  FW region span: 0x{fw_addr_b:x}..0x{fw_addr_b + (fw_kb<<10):x}")
    print(f"                  ({fw_addr_b/(1024**3):.3f} GiB .. {(fw_addr_b + (fw_kb<<10))/(1024**3):.3f} GiB)")
    print(f"  DRV region span: 0x{dr_addr_b:x}..0x{dr_addr_b + (drv_kb<<10):x}")
    print(f"                  ({dr_addr_b/(1024**3):.3f} GiB .. {(dr_addr_b + (drv_kb<<10))/(1024**3):.3f} GiB)")
else:
    print(f"  Unknown rev {fmt_rev}.{content_rev} — raw dump above")

# Cross-check: where does our SOS crash live?
sos_fault_virt = 0x83f00f80
print(f"\n=== Cross-check vs SOS crash address ===")
print(f"  SOS fault virt   = 0x{sos_fault_virt:x} ({sos_fault_virt/(1024**3):.3f} GiB)")
print(f"  System aperture  = [0x40000000 (LOW), 0x83FEFC0000 (HIGH)]")
print(f"                     ({0x40000000/(1024**3):.3f} GiB .. {0x83FEFC0000/(1024**3):.3f} GiB)")
print(f"  Inside aperture? {0x40000000 <= sos_fault_virt <= 0x83FEFC0000}")
print()
print("  If fw region is far from 0x83f00f80, the fault isn't a vram_usage region.")
print("  Look in: vram_usage span, driver region span, or far above (high system aperture).")
