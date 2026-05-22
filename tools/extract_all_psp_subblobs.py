#!/usr/bin/env python3
"""
η.3q-1 — extract ALL PSP sub-blobs from psp_13_0_10_sos.bin (not just SOSDRV).

Up to now we've only RE'd the SOSDRV sub-blob (PSP_FW_TYPE=1). The BL
chain also loads KDB(3) + SYS_DRV(2) + SOC_DRV(8) + INTF_DRV(9) +
DBG_DRV(10). The fault PC 0x0d11_02f4 sits at 1.07 MiB virt — way
beyond SOSDRV's 56 KB encrypted body — so the actual faulting code
is in a DIFFERENT sub-blob (most likely SYS_DRV = the PSP kernel).

For each sub-blob, dump:
  - Size
  - First 64 bytes hex
  - Entropy estimate per 1 KiB chunk
  - Whether $PS1 magic present (= encrypted PSP fw with header)
  - ARM/Thumb decode test (does the first few KB disasm cleanly?)
"""
import struct
import subprocess
from pathlib import Path

SOS_BIN = Path("/tmp/mlrift_fw/psp_13_0_10_sos.bin")
OUT_DIR = Path("/home/pantelis/Desktop/Projects/Work/MLRift/captures")

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
    if not SOS_BIN.exists():
        print(f"ERROR: {SOS_BIN} not found — run tools/restore_eta3h_inputs.sh first")
        return 1
    data = SOS_BIN.read_bytes()
    print(f"Container: {SOS_BIN} ({len(data)} bytes)")
    ucode_array_off = struct.unpack_from("<I", data, 24)[0]
    psp_bin_count = struct.unpack_from("<I", data, 32)[0]
    print(f"psp_fw_bin_count = {psp_bin_count}, ucode_array_off = 0x{ucode_array_off:x}\n")

    for idx in range(psp_bin_count):
        d = 36 + idx * 16
        fw_type, fw_ver, rel_off, fw_size = struct.unpack_from("<4I", data, d)
        name = FW_TYPE_NAMES.get(fw_type, f"?{fw_type}")
        abs_off = ucode_array_off + rel_off

        print(f"=== {name} (fw_type={fw_type}) @ abs 0x{abs_off:x}, size {fw_size} ===")
        sub = data[abs_off:abs_off+fw_size]
        out_path = OUT_DIR / f"{name.lower()}_subblob.bin"
        out_path.write_bytes(sub)
        print(f"  → {out_path}")

        # First 32 bytes
        print(f"  first 32 bytes:")
        for o in range(0, 32, 16):
            chunk = sub[o:o+16]
            hex_str = ' '.join(f'{b:02x}' for b in chunk)
            ascii_str = ''.join(chr(b) if 32 <= b < 127 else '.' for b in chunk)
            print(f"    +{o:03x}: {hex_str:<48} {ascii_str}")

        # Look for $PS1 magic at common offsets
        ps1 = b"$PS1"
        ps1_locs = []
        for try_off in (0x0, 0x10, 0x20, 0x40, 0x100):
            if try_off + 4 <= len(sub) and sub[try_off:try_off+4] == ps1:
                ps1_locs.append(try_off)
        if ps1_locs:
            print(f"  $PS1 magic found at offsets: {ps1_locs}")
        else:
            print(f"  $PS1 magic NOT found in first 0x100 bytes")

        # Entropy estimate (unique bytes per 1 KiB chunk)
        chunks_e = []
        for i in range(0, len(sub), 1024):
            c = sub[i:i+1024]
            if len(c) < 16: continue
            chunks_e.append(len(set(c)))
        if chunks_e:
            avg = sum(chunks_e) / len(chunks_e)
            high = sum(1 for u in chunks_e if u > 200)
            low  = sum(1 for u in chunks_e if u < 100)
            print(f"  entropy: avg {avg:.1f}/256 unique per KB, {high}/{len(chunks_e)} high-entropy chunks (>200), {low} low (<100)")
            if avg > 200:
                print(f"  --> LIKELY FULLY ENCRYPTED")
            elif avg > 150:
                print(f"  --> MIXED: header + encrypted body, or compressed")
            else:
                print(f"  --> LIKELY PLAINTEXT (or mostly-plaintext)")
        print()

if __name__ == "__main__":
    raise SystemExit(main() or 0)
