#!/usr/bin/env python3
"""
θ.3-C — search ALL plaintext drivers for the kernel-main entry at
PSP SRAM virt 0x2ee0. Test each (load, body) combo per driver.

Also test other key SOS-virt addresses:
  - 0x2ee0  (kernel main from SOS Reset bx ip)
  - 0x50c4  (SVC dispatcher target from SOS SVC vector blx)
  - 0xcbc   (exception logger target from SOS common-logger blx)
  - 0xe2c   (IRQ user handler target from SOS IRQ vector blx)
"""
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

TARGETS = {
    "kernel_main": 0x2ee0,
    "svc_dispatcher": 0x50c4,
    "exception_logger": 0xcbc,
    "irq_user_handler": 0xe2c,
}

CONFIGS = [(0x0, 0x0), (0x0, 0x100), (0x0, 0x400), (0x0, 0x1000)]

for name, fn in BLOBS:
    p = Path("/home/pantelis/Desktop/Projects/Work/MLRift/captures") / fn
    if not p.exists(): continue
    data = p.read_bytes()
    all_insns = list(md.disasm(data, 0))
    fn_starts = set()
    for ins in all_insns:
        if ins.mnemonic in ("push", "push.w") and "lr" in ins.op_str:
            fn_starts.add(ins.address)

    print(f"\n=== {name} ({len(data)} bytes, {len(fn_starts)} fn starts) ===")
    for tgt_name, tgt in TARGETS.items():
        for load, body in CONFIGS:
            fo = (tgt - load) + body
            if fo < 0 or fo + 8 > len(data): continue
            is_fn = fo in fn_starts
            if is_fn:
                # Print 5 insns
                addr_to_idx = {ins.address: i for i, ins in enumerate(all_insns)}
                if fo in addr_to_idx:
                    insns = all_insns[addr_to_idx[fo]:addr_to_idx[fo]+6]
                    print(f"  {tgt_name} (virt 0x{tgt:x}) → file 0x{fo:x} ← FN START "
                          f"(load=0x{load:x}, body=0x{body:x})")
                    for ins in insns:
                        print(f"      0x{ins.address:08x}: {ins.bytes.hex():<10} {ins.mnemonic:<10} {ins.op_str}")
                    break  # only print first matching config per target
