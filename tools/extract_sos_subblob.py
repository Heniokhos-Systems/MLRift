#!/usr/bin/env python3
"""
Extract the SOS sub-blob from psp_13_0_10_sos.bin for reverse-engineering.

The .bin uses AMD's v2 firmware container format:
  +0  uint32 magic_or_size
  +4  uint32 header_size      = ucode_array_offset (start of sub-blobs)
  +8  uint32 fw_version       (e.g. 0x00320d17)
  +12 uint32 fw_type
  +16 uint32 ucode_array_offset_bytes  (sub-blob region start)
  +20 ...

Then a list of v2 sub-headers starting at the parent's
ucode_array_offset_bytes. Each sub-header:
  +0  uint32 fw_offset       (relative to ucode_array start)
  +4  uint32 fw_size_bytes
  +8  uint32 fw_type         (matches PSP_FW_TYPE_* enum)
  +12 uint32 fw_version

PSP_FW_TYPE for SOS is 1.

This script extracts the SOS sub-blob to a file plus prints metadata
useful for RE setup.
"""

import struct
import subprocess
from pathlib import Path

SOS_BIN     = Path("/tmp/mlrift_fw/psp_13_0_10_sos.bin")
OUT_DIR     = Path("/home/pantelis/Desktop/Projects/Work/MLRift/captures")
OUT_BLOB    = OUT_DIR / "sos_subblob.bin"

# PSP_FW_TYPE values (from std/amdgpu_firmware.mlr)
FW_TYPE_NAMES = {
    1:  "SOS",
    2:  "SYS_DRV",
    3:  "KDB",
    6:  "SPL",
    8:  "SOC_DRV",
    9:  "INTF_DRV",
    10: "DBG_DRV",
    11: "RAS_DRV",
}

def main():
    data = SOS_BIN.read_bytes()
    print(f"Container file size: {len(data)} bytes")

    # common_firmware_header: size at +0, header_size at +4, version at +8,
    # ucode_version at +12, ucode_size at +16, ucode_array_offset at +20
    # struct common_firmware_header layout:
    #   uint32 size_bytes              +0
    #   uint32 header_size_bytes       +4
    #   uint16 header_ver_major        +8
    #   uint16 header_ver_minor        +10
    #   uint16 ip_version_major        +12
    #   uint16 ip_version_minor        +14
    #   uint32 ucode_version           +16
    #   uint32 ucode_size_bytes        +20
    #   uint32 ucode_array_offset      +24    <-- 16-bit fields shifted this!
    #   uint32 crc32                   +28
    common_size       = struct.unpack_from("<I", data,  0)[0]
    common_header_sz  = struct.unpack_from("<I", data,  4)[0]
    common_hdr_ver_maj, common_hdr_ver_min = struct.unpack_from("<HH", data, 8)
    ucode_version     = struct.unpack_from("<I", data, 16)[0]
    ucode_size_bytes  = struct.unpack_from("<I", data, 20)[0]
    ucode_array_off   = struct.unpack_from("<I", data, 24)[0]
    common_hdr_ver    = (common_hdr_ver_maj << 16) | common_hdr_ver_min
    print(f"common header: size={common_size}, header_size={common_header_sz}, ver=0x{common_hdr_ver:x}")
    print(f"ucode_version=0x{ucode_version:x}, ucode_size_bytes=0x{ucode_size_bytes:x}, ucode_array_off=0x{ucode_array_off:x}")

    # psp_v2 header: count at offset 32, descriptors START AT offset 36
    # (16 bytes each: fw_type, fw_version, offset_bytes, size_bytes).
    psp_bin_count = struct.unpack_from("<I", data, 32)[0]
    print(f"psp_fw_bin_count = {psp_bin_count}")
    print()

    desc_base = 36
    print(f"{'idx':<4} {'fw_type':<10} {'name':<10} {'offset':<10} {'size':<10} {'version':<10}")
    for idx in range(psp_bin_count):
        d = desc_base + idx * 16
        if d + 16 > len(data):
            print(f"  (descriptor {idx} truncated)")
            break
        fw_type, fw_ver, rel_off, fw_size = struct.unpack_from("<4I", data, d)
        name = FW_TYPE_NAMES.get(fw_type, f"?{fw_type}")
        # rel_off is relative to ucode_array_off, OR sometimes absolute
        # from the file start? In our blob KDB had rel_off=0 size=7488 and
        # the BL chain copied 7488 bytes starting at the ucode array region,
        # so rel_off IS relative to ucode_array_off.
        abs_off = ucode_array_off + rel_off
        print(f"{idx:<4} {fw_type:<10} {name:<10} 0x{rel_off:<8x} {fw_size:<10} 0x{fw_ver:08x}  abs=0x{abs_off:x}")

        if fw_type == 1:  # SOS
            OUT_DIR.mkdir(parents=True, exist_ok=True)
            OUT_BLOB.write_bytes(data[abs_off:abs_off+fw_size])
            print(f"   --> extracted SOS sub-blob to {OUT_BLOB} ({fw_size} bytes)")

    if not OUT_BLOB.exists():
        print(f"WARNING: SOS sub-blob (type 1) not found in container.")
        return 2

    print()
    print(f"=== SOS sub-blob extracted, analyzing ===")
    blob = OUT_BLOB.read_bytes()
    print(f"Size: {len(blob)} bytes")
    print(f"First 64 bytes (header):")
    for i in range(0, 64, 16):
        chunk = blob[i:i+16]
        hex_str = ' '.join(f'{b:02x}' for b in chunk)
        ascii_str = ''.join(chr(b) if 32 <= b < 127 else '.' for b in chunk)
        print(f"  {i:04x}: {hex_str:<48} {ascii_str}")

    # File magic detection
    print()
    print(f"file(1) on sub-blob:")
    try:
        r = subprocess.run(["file", str(OUT_BLOB)], capture_output=True, text=True)
        print(f"  {r.stdout.strip()}")
    except FileNotFoundError:
        pass

    # Entropy snapshot
    print()
    chunk_size = 1024
    entropies = []
    for i in range(0, len(blob), chunk_size):
        chunk = blob[i:i+chunk_size]
        if len(chunk) < 16: continue
        # cheap entropy = unique byte count / total bytes
        u = len(set(chunk))
        entropies.append(u)
    avg_u = sum(entropies) / max(1, len(entropies))
    high_e = sum(1 for u in entropies if u > 200)
    print(f"Avg unique-bytes / 1KiB chunk: {avg_u:.1f} / 256")
    print(f"Chunks with >200 unique bytes (likely compressed/encrypted): {high_e}/{len(entropies)}")

    if avg_u > 200:
        print("HIGH ENTROPY — likely signed+encrypted blob. Header may decode but body is opaque.")
    elif avg_u > 100:
        print("MIXED ENTROPY — code segments (low) + tables/data (medium).")
    else:
        print("LOW ENTROPY — likely raw code/data, accessible for static analysis.")

    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
