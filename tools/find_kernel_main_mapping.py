#!/usr/bin/env python3
"""
θ.3-B — find the correct SYS_DRV load address.

The SOS Reset handler jumps to PSP virt 0x2ee0 (Thumb) for kernel main.
Under load=0/body=0, file 0x2ee0 is mid-function (in an I/O routine at
file 0x2ec4). That's suspicious — kernel main should be a clean entry.

Try multiple (load_addr, body_start) combos and check whether the
file offset for virt 0x2ee0 contains a function prologue.
"""
from pathlib import Path
from capstone import Cs, CS_ARCH_ARM, CS_MODE_THUMB

SYS_DRV = Path("/home/pantelis/Desktop/Projects/Work/MLRift/captures/sys_drv_subblob.bin").read_bytes()
md = Cs(CS_ARCH_ARM, CS_MODE_THUMB); md.skipdata = True

# Find ALL function-start offsets (push.w with lr)
all_insns = list(md.disasm(SYS_DRV, 0))
fn_starts = set()
for ins in all_insns:
    if ins.mnemonic in ("push", "push.w") and "lr" in ins.op_str:
        fn_starts.add(ins.address)

print(f"SYS_DRV: {len(SYS_DRV)} bytes")
print(f"Total function starts found: {len(fn_starts)}\n")

# Check various (load_addr, body_start) configs for virt 0x2ee0
# file_off = (virt - load_addr) + body_start
TARGET_VIRT = 0x2ee0

CONFIGS = [
    (0x0,    0x0),   (0x0,    0x100),  (0x0,    0x400),  (0x0,    0x1000),
    (0x1000, 0x0),   (0x1000, 0x100),  (0x1000, 0x1000),
    (0x2000, 0x0),   (0x2000, 0x100),  (0x2000, 0x1000),
    (0x4000, 0x0),   (0x4000, 0x1000),
    (0x8000, 0x0),   (0x8000, 0x1000),
    (0x10000, 0x0),  (0x10000, 0x1000),
    (0x20000, 0x0),  (0x40000, 0x0),
]

print(f"Searching for mapping where virt 0x{TARGET_VIRT:x} = clean function start...")
print(f"{'load':>10}  {'body':>10}  {'file_off':>10}  is_fn_start  surrounding")
print("-" * 80)
for load, body in CONFIGS:
    file_off = (TARGET_VIRT - load) + body
    if file_off < 0 or file_off + 8 > len(SYS_DRV):
        continue
    is_fn = file_off in fn_starts
    # Show first 4 bytes
    chunk = SYS_DRV[file_off:file_off+8].hex()
    flag = " ← FN START!" if is_fn else ""
    print(f"  0x{load:08x}  0x{body:08x}  0x{file_off:08x}  {'YES' if is_fn else 'no':<11}  {chunk}{flag}")

# Cross-check: also test SVC dispatcher target 0x50c4 (= SOS SVC's blx 0x50c5)
print(f"\nCross-check virt 0x50c4 (SOS SVC vector calls this):")
print(f"{'load':>10}  {'body':>10}  {'file_off':>10}  is_fn_start  surrounding")
print("-" * 80)
TARGET_SVC = 0x50c4
for load, body in CONFIGS:
    file_off = (TARGET_SVC - load) + body
    if file_off < 0 or file_off + 8 > len(SYS_DRV):
        continue
    is_fn = file_off in fn_starts
    chunk = SYS_DRV[file_off:file_off+8].hex()
    flag = " ← FN START!" if is_fn else ""
    print(f"  0x{load:08x}  0x{body:08x}  0x{file_off:08x}  {'YES' if is_fn else 'no':<11}  {chunk}{flag}")

# A mapping is CONSISTENT if both kernel main and SVC dispatcher are
# function starts under the same (load, body) config.
print("\nConsistency check: mapping where BOTH targets are clean fn starts:")
for load, body in CONFIGS:
    km_off = (TARGET_VIRT - load) + body
    svc_off = (TARGET_SVC - load) + body
    if km_off < 0 or svc_off < 0: continue
    if km_off + 8 > len(SYS_DRV) or svc_off + 8 > len(SYS_DRV): continue
    if km_off in fn_starts and svc_off in fn_starts:
        print(f"  load=0x{load:x}, body=0x{body:x}  ← BOTH CONSISTENT")
