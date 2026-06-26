# Phase 3 — Cold Boot Research Report

Comprehensive findings from the cold-boot investigation for the AMD Radeon
RX 7800 XT (Navi 32, `psp_13_0_10`). This document captures the GPU's
internal architecture as we mapped it, every block of data we extracted
from the silicon, and the final state of the wall analysis.

Status: **REOPENED (2026-06-11).** The prior "investigation closed —
silicon wall" conclusion was built on a premise that is now **proven
false**. See §14 for the full re-audit. In brief:

- The worker thread the report claimed was unreadable ("almost certainly
  from the encrypted SOS body") is **plaintext firmware we already
  possess**. It is `INTF_DRV` (the `LOAD_INTFDRV` BL component), loaded
  at PSP SRAM base `0x00200000`. There is **no encrypted body** — the
  region the report pointed at (`captures/sos_encrypted_body.bin`,
  57,344 B) is 99.9 % zeros (entropy 0.02).
- The "wall" at `svc #0xe` / virt `0x0d126e9e` was a Thumb-alignment
  **misread**. The real instruction at that address is `bl 0xd12700c`.
  The actual `svc #0xe` sites (virt `0x0d12655c`, `0x0d126d9e`) are SOS's
  normal **host-command idle loops** (`wait → svc #0x40 poll → loop`),
  not a one-shot worker-wait. SOS halts in **init**, before SoL — not
  parked in a dead wait.
- The DRAM-training mechanism that §7.5 / η.3m called "genuinely UNKNOWN…
  inside SOS encrypted body / boot-ROM silicon" is **plaintext code in
  INTF_DRV**: a mailbox handshake plus `0xaa55aa55` memory-training writes
  to UMC/PHY registers via kernel register-write services. The training
  signature is present only in INTF_DRV and absent from the entire SOS
  blob. It is a readable RDNA3 driver, not silicon ROM. (One detail is
  still open — the exact runtime call path into the trainer, an
  INTF_DRV-vs-SOS-`$PS1` overlay question; see §14.3. It does not affect
  the headline: the trainer is plaintext and in our possession.)

**Still genuinely open:** *why* a from-cold MLRift BL chain fails to reach
SoL while amdgpu cold-boots this exact card on every power-on. That is a
software delta (a precondition PS1_1 needs, or MLRift not reaching the
PS1_1 spawn), now tractable by finishing the RE of the plaintext
INTF_DRV path — not a fab-rooted silicon wall. The earlier "30+
hypotheses → silicon wall" closure does **not** hold.

The warm-hybrid path (amdgpu loads PSP firmware once, then hands control
to MLRift via VFIO) remains the shipping deliverable for the LLM mission:
Mistral-7B mega-kernel at +24% over PyTorch bf16, Qwen3 at 200+ tok/s,
bit-exact validation across 7 model families. Cold-boot is a parallel
research track, now reopened.

---

## 1. Hardware target

| Field | Value |
|---|---|
| Chip | AMD Navi 32 (gfx1100, gc_11_0_3, RDNA3) |
| Card | PowerColor RX 7800 XT |
| Memory | 16 GiB GDDR6 |
| PCI ID | `1002:747e` |
| Subsystem | Tul Corp. / PowerColor `1482:2427` |
| PSP version | `psp_13_0_10` (retail consumer, secure-fused) |
| SMU version | `smu_v13_0_7` |
| Host | Linux 6.17.0-29-generic, vfio-pci default-bound to dGPU |

## 2. PCIe view of the GPU

```
                           HOST CPU (x86_64)
                                 |
                         +-------+-------+
                         |  IOMMU group  |
                         |      14       |
                         +-------+-------+
                                 | PCIe Gen4 x16
                                 v
       +----------------------------------------------------------+
       |     PCI 0000:03:00.0   AMD Navi 32 (1002:747e)           |
       |  +-----------------------------------------------------+ |
       |  |  PCIe Config Space                                  | |
       |  |   * VSEC 0x100 (body all zeros - empty on retail)   | |
       |  |   * AER, PASID enabled, Resizable BAR               | |
       |  |   * BIST not capable (config 0x0F = 0x00)           | |
       |  +-----------------------------------------------------+ |
       |                                                          |
       |  BAR0   16 GB   prefetchable  ->  VRAM aperture          |
       |  BAR2  256 MB   prefetchable  ->  Doorbell aperture      |
       |  BAR4  256 B    I/O ports     ->  legacy                 |
       |  BAR5    1 MB   non-pref      ->  SOC15 register block * |
       |  EROM  128 KB   (disabled)    ->  VBIOS image            |
       +----------------------------------------------------------+
              * our primary control surface (used by all probes)
```

Region numbering note: PCIe BARs 0-5 are 6 slots. A 64-bit BAR consumes TWO
consecutive slots (low 32 + high 32). So "BAR1" and "BAR3" are not missing;
they hold the upper halves of BAR0 (16 GB VRAM) and BAR2 (256 MB doorbells).

## 3. Chip internal topology

```
                           SMN (System Management Network)
                           0x00xxxxxx - 0x07xxxxxx
   +-------------------------------------------------------------------+
   |                                                                   |
   |  +-------------+    +-------------+    +-------------+            |
   |  |  MP0 (PSP)  |<-->|  MP1 (SMU)  |    |   GC (gfx)  |            |
   |  |  0x0320xxxx |    |  0x0301xxxx |    |             |            |
   |  |             |    |             |    |  +--------+ |            |
   |  | ARM Cortex-A5    | Microcode   |    |  |  MEC   | |            |
   |  | 64+ KB SRAM |    | controller  |    |  |(compute) |          |
   |  | * ENCRYPTED |    |             |    |  +--------+ |            |
   |  |   FW inside |    |             |    |  +--------+ |            |
   |  |             |    |             |    |  |  MES   | |            |
   |  | C2PMSG_0..  |    | C2PMSG_66/  |    |  |(sched) | |            |
   |  | ..127 mbox  |    | 82/90 mbox  |    |  +--------+ |            |
   |  +------+------+    +------+------+    |  +--------+ |            |
   |         | commands         | msgs      |  |  RLC   | |            |
   |         v                  v           |  |        | |            |
   |   controls all firmware loads          |  +--------+ |            |
   |   + DRAM training trigger              +------+------+            |
   |                                               |                   |
   |  +--------------+  +--------------+  +--------+-----+             |
   |  |  GMC/MMHUB   |  |   OSSSYS     |  |    SDMA      |             |
   |  |   page table |  |     (IH)     |  |  DMA engine  |             |
   |  |   walker     |  |  Interrupt   |  |              |             |
   |  |              |  |   Handler    |  |              |             |
   |  +------+-------+  +--------------+  +--------------+             |
   |         |                                                         |
   |  +------+-------+                                                 |
   |  |   UMC v8.10  |   --- connects to --->  +--------------------+  |
   |  | memory ctrl  |                         |  GDDR6  (16 GiB)   |  |
   |  | + PHY        |                         |  * needs training  |  |
   |  +--------------+                         | RCC_CONFIG_MEMSIZE |  |
   |                                           | reflects trained MB|  |
   |  +--------------+  +--------------+       +--------------------+  |
   |  |     NBIF     |  |     HDP      |                               |
   |  | (PCIe iface) |  | (host data   |                               |
   |  |              |  |  path)       |                               |
   |  +--------------+  +--------------+                               |
   +-------------------------------------------------------------------+
```

### 3.1 IP block SOC15 base offsets (Navi 32, BASE_IDX = 0)

These are byte offsets from BAR5 start. Values populated by
`amdgpu_load_navi32_bases()` in `std/amdgpu_ip_discovery.mlr`.

| IP block | role | typical access |
|---|---|---|
| MP0 (PSP) | secure processor, FW loader | C2PMSG_0..127 mailbox |
| MP1 (SMU) | power management, clocks | C2PMSG_66/82/90 mailbox |
| GC | graphics + compute (MEC, MES, RLC) | requires PSP-loaded FW |
| GMC / MMHUB | memory page-table walker | needs init before PSP DMA |
| OSSSYS | interrupt handler (IH) | ring init + writeback |
| SDMA0 | DMA engine | needs FW |
| HDP | host data path | atomic power gate |
| NBIF | PCIe interface, FB enable | BIF_FB_EN gating |
| SMUIO | system management I/O, fuses | chip ID readout |

## 4. PSP processor internals (what we decoded)

The PSP is an ARMv7-A Cortex-A5 class processor with:
- 32-bit, little-endian, ARM mode + Thumb-2 interworking
- ~64 KiB on-die SRAM (externally inaccessible per theta.2)
- On-die ROM containing boot code + AMD public key
- MMU with TTBR0 (PT base = 0xc000, TTBR0 = 0xc048, DACR = 0x55555555)
- Banked stacks: 0x9000 (abort), 0x9100 (IRQ), 0x9300 (SVC)

### 4.1 SOS internal address layout

When SOS loads, it runs at virt base `0x0d110000`:

```
Virt addr               | Content
------------------------|--------------------------------------------------
0x0d110000              | Reset vector  -> b 0x0d110088
0x0d110004              | Undef vector  -> b 0x0d1101f8 (r0=3)
0x0d110008              | SVC vector    -> b 0x0d110228 (kernel-call dispatch)
0x0d11000c              | PrefAbort     -> b 0x0d1102ac (r0=1)
0x0d110010              | Data Abort    -> b 0x0d1102f0 (r0=2) <-- fault hits here
0x0d110018              | IRQ vector    -> b 0x0d110344 (calls 0xe2c)
0x0d110028 - 0x0d110500 | ARM-mode handlers (plaintext)
0x0d110500 - 0x0d116d00 | R1 user-mode Thumb-2 code (plaintext)
0x0d117000 - 0x0d124fff | (was thought to be encrypted - ACTUALLY ZEROS)
0x0d125000 - 0x0d127000 | R2 user-mode Thumb-2 code (plaintext, including
                        |   kernel MP0/MP1 init fn @ virt 0x0d126eb4)
```

### 4.2 Decoded SOS internals

| Construct | Location | Description |
|---|---|---|
| DEBUG_STATUS helper | virt 0x0d111d50 | Writes SMN `0x032000d8` with bit 31 marker |
| State 0x22 halt | virt 0x0d112fc4 | After SMU-register snapshot + cache flush |
| Command dispatcher | virt 0x0d115260 | Thumb `tbb [pc, ip]` at 0x5294, 79-entry table |
| Common exception logger | virt 0x0d110288 | Calls SRAM 0xcbc (SOC_DRV exception report) |
| MP0/MP1 mapping init | virt 0x0d126eb4 | `svc #0x42` for 0x03200000 / 0x03010000 |

### 4.3 SYS_DRV (the PSP "kernel/OS")

61,696 bytes plaintext Thumb-2 code, loaded at PSP SRAM offset 0.

| Function | File offset | Role |
|---|---|---|
| SVC dispatcher | 0x50c4 | Called by SOS SVC vector via `blx 0x50c5` |
| Kernel main entry target | 0x2ee0 | Called by SOS Reset via `bx 0x2ee1` |
| (jumped mid-function) | | SOS deliberately enters at offset 0x1c into fn @ 0x2ec4 |

Notable: ZERO CP15 (MMU/cache) instructions anywhere in SYS_DRV. The kernel
never manipulates the MMU after boot ROM sets it up.

### 4.4 SOC_DRV (the exception bridge)

33,024 bytes plaintext. Contains the exception report function at PSP SRAM
0xcbc that issues `svc #0xf6` to forward fault info to SYS_DRV kernel.

### 4.5 SVC numbering (kernel API surface)

| SVC # | Purpose | Sites in plaintext |
|---|---|---|
| 0x0c | open/lookup session (paired with 0x4d) | many |
| 0x42 | `map_physical(addr, ?, flags)` | 14 in sub-blob #2 |
| 0x4d | copy 8 bytes from user space | 13 in SYS_DRV |
| 0xf6 | report event to kernel | 102 in SOC_DRV |

## 5. Cold boot flow (where it works, where it breaks)

```
   Power-On Reset
        |
        v
   +---------------------+
   |  PSP Boot ROM       |   * on-die silicon ROM, AES-encrypted storage
   |  (NEVER VISIBLE)    |   * signature-verifies all loaded firmware
   |                     |   * sets up initial page table (including 0x83f00xxx)
   |                     |   * installs unhandled-exception trap
   +----------+----------+
              |
              v
   +----------------------------------------------------+
   | BL chain (we replicate this fine)                  |
   |                                                    |
   |  1. LOAD_KDB        (key database)        rc=0  OK |
   |  2. LOAD_SPL        (secure platform tbl) rc=0  OK |
   |  3. LOAD_SYS_DRV    (PSP "kernel/OS")     rc=0  OK |
   |  4. LOAD_SOC_DRV                          rc=0  OK |
   |  5. LOAD_INTF_DRV                         rc=0  OK |
   |  6. LOAD_DBG_DRV                          rc=0  OK |
   |                                                    |
   |  Each command verified by BL signature check       |
   +----------+-----------------------------------------+
              |
              v
   +----------------------------------------------------+
   | LOAD_SOSDRV                                        |
   |   PSP verifies signature                           |
   |   Jumps to SOS Reset @ virt 0x0d110000             |
   +----------+-----------------------------------------+
              |
              v
   +----------------------------------------------------+
   | SOS Reset handler (ARM mode, PLAINTEXT)            |
   |   * Self-relocates                                 |
   |   * Sets TTBR0 = 0xc048 (PT base 0xc000)           |
   |   * Banked stacks                                  |
   |   * Enables MMU                                    |
   |   * bx to virt 0x2ee0 (kernel main in SYS_DRV)     |
   +----------+-----------------------------------------+
              |
              v
   +----------------------------------------------------+
   | SOS kernel main (Thumb-2, plaintext SYS_DRV)       |
   |   * Goes through init, dispatches commands         |
   |   * At some point, internal RPC writes state 0x32  |
   |     to DEBUG_STATUS (we observe 0x80320d17 in      |
   |     C2PMSG_58)                                     |
   |                                                    |
   |  Tracing marker - NOT the halt                     |
   +----------+-----------------------------------------+
              |
              v
   +----------------------------------------------------+
   | Code path requires virt 0x83f00f80 access          |
   |   - PTE for that virt was set by boot ROM          |
   |   - Maps to physical region that needs VRAM        |
   |     trained (or similar precondition)              |
   |   - On cold boot, VRAM untrained -> Data Abort     |
   |                                                    |
   |  Boot ROM's unhandled-exception trap fires:        |
   |    C2PMSG_60 = 4 (composite fault type)            |
   |    C2PMSG_61 = 0x83f00f80 (fault virt addr)        |
   |    C2PMSG_62 = 0x0d1102f4 (handler self-mark PC)   |
   |                                                    |
   |   *** THE WALL ***                                 |
   +----------------------------------------------------+
```

## 6. Warm boot flow (the working path for the LLM mission)

```
   amdgpu kernel module loads
        |
        v
   +---------------------------------------------------------+
   | Same BL chain + LOAD_SOSDRV as cold (we don't override) |
   | PSP signature-verifies + loads everything correctly     |
   | SOS reaches BOOT_STEADY:                                |
   |    C2PMSG_81 = 0x016ed035 (warm SoL marker)             |
   |    VRAM trained (RCC_CONFIG_MEMSIZE = 16384 MiB)        |
   |    TMR allocated at 0x83e0000000 (167 MB)               |
   |    MEC/MES/RLC/SDMA firmware loaded by PSP              |
   +----------+----------------------------------------------+
              |
              v
   +---------------------------------------------------------+
   | amdgpu->vfio handoff (PSP aux power survives)           |
   |   echo 03:00.0 > /sys/.../amdgpu/unbind                 |
   |   echo 03:00.0 > /sys/.../vfio-pci/bind                 |
   |   PSP state PERSISTS (no software reset clears PSP)     |
   +----------+----------------------------------------------+
              |
              v
   +---------------------------------------------------------+
   |  MLRift takes over (no PyTorch, no ROCm, no hipcc)      |
   |   * Configures doorbells, MQDs                          |
   |   * Submits PM4 packets                                 |
   |   * MEC executes kernels (already PSP-loaded)           |
   |                                                         |
   |  Shipping: Qwen3 200+ tok/s, Mistral-7B mega-kernel     |
   |            +24% over PT bf16                            |
   +---------------------------------------------------------+
```

## 7. Captured data from the GPU (durable reference values)

### 7.1 PSP mailbox registers (read via BAR5 SOC15 path)

All offsets are in MP0 register block, accessed as
`BAR5 + (amdgpu_ip_mp0_base_0 + dword) * 4`.

All values **empirically verified** in this session — earlier columns of
this table were partly wrong; see corrections in §13.

| Reg | dword | Cold (pre-anything) | Post-LOAD_SOSDRV stall | Warm SOS | Meaning |
|---|---|---|---|---|---|
| C2PMSG_33 | 0x61 | `0x80000000` | `0x80000000` | varies | KDB ready (bit 31) — see `psp_v13_0_wait_for_vmbx_ready` |
| C2PMSG_35 | 0x63 | `0x80000000` | `0x00000000` | varies | BL ready (bit 31). After SOS handoff goes to 0. |
| C2PMSG_36 | 0x64 | 0 | (last cmd buf addr) | varies | Buffer addr `>> 20` for BL commands |
| C2PMSG_58 | 0x7a | **`0x00000000`** | **`0x00320d17`** | (varies) | **SOS firmware version label** (written by `report_version()` @ virt `0x0d12630c`). NOT a state machine indicator. |
| C2PMSG_59 | 0x7b | 0 | `0x0032041c` | (varies) | Companion fw-version-shaped value, BL/SOS init artifact |
| C2PMSG_60 | 0x7c | **`4`** | `4` (unchanged) | (varies) | **Pre-existing cold-boot value** — written by boot ROM POST self-test, NOT a runtime fault indicator |
| C2PMSG_61 | 0x7d | **`0x83f00f80`** | `0x83f00f80` (unchanged) | (varies) | **Pre-existing** — see §11 correction |
| C2PMSG_62 | 0x7e | **`0x0d1102f4`** | `0x0d1102f4` (unchanged) | (varies) | **Pre-existing** — see §11 correction |
| C2PMSG_81 | 0x91 | 0 | 0 (stays 0 — wall here) | **`0x016ed035`** | SoL marker. amdgpu `psp_v13_0_is_sos_alive` checks `!= 0`. |
| C2PMSG_88 | 0x98 | 0 | 0 | varies | Runtime DB offset |
| C2PMSG_89 | 0x99 | 0 | `0x68693199` | (varies) | BL-side artifact; amdgpu doesn't reference |
| C2PMSG_90 | 0x9a | 0 | `0xcb439e60` | (varies) | BL-side artifact |
| C2PMSG_91 | 0x9b | 0 | `0x0d320049` | (varies) | BL-side artifact, version-shaped |
| C2PMSG_92 | 0x9c | 0 | 0 (!!!) | 0 | "BOOT_STATUS" — UNRELIABLE on this chip |
| C2PMSG_115 | 0xb3 | 0 | `0x80000000` | (varies) | SPI flash update mailbox `MBOX_READY` (bit 31) — NOT a SOS indicator |
| C2PMSG_118 | 0xb6 | 0 | `0x80000000` | (varies) | Companion to C2PMSG_115 in SPI mailbox subsystem |
| **SMN `0x032000d8`** | dw 0x36 | **`0x00000000`** | **`0x00000000`** | (writes during boot) | **THE REAL DEBUG_STATUS register.** SOS helper @ virt `0x0d111d50` would write it — but SOS *actively clears it* at virt `0x0d126eec` and the helper is gated behind the command dispatcher which only runs after the worker thread posts the IPC event. |

**Critical reframe**: The amdgpu source says C2PMSG_92 = `AMDGPU_RAS_BOOT_STEADY_STATUS` should read `0xBA` on warm-steady. On `psp_13_0_10` it reads 0 even when SOS is fully booted. Use `C2PMSG_81 != 0` as the SoL marker instead.

### 7.2 SMUIO chip-info dump (theta.2 capture, warm state)

Read via BAR5 SMUIO IP block. The first 0x80 dwords:

```
dw[0x040] = 0x00000863
dw[0x041] = 0x00000062
dw[0x042] = 0x00000055
dw[0x043] = 0x00000001
dw[0x045] = 0x00000780
dw[0x046] = 0x00000820
dw[0x047] = 0x0000003c
dw[0x048] = 0x00000082
dw[0x049] = 0x00000006
dw[0x04a] = 0x00000008
dw[0x04b] = 0x00000011
dw[0x04c] = 0x000048ff
dw[0x04d] = 0x00000011
dw[0x050] = 0x00000001
dw[0x05b] = 0x00000001
dw[0x05c] = 0x00000006
dw[0x05f] = 0x00000050
dw[0x065] = 0x00000064
dw[0x066] = 0x00000001
dw[0x067] = 0x00000001
dw[0x068] = 0x00000014
dw[0x069] = 0x00000001
dw[0x07d] = 0x0017178e   <- numeric timestamp / counter
dw[0x07e] = 0x3230322a   <- LE bytes "*202" (fuse/lot string)
dw[0x07f] = 0x44570140   <- LE bytes "@\x01WD"
```

The `*202` string suggests a fab lot code from 2022.

### 7.3 PCIe extended config space (theta.2 sudo lspci -xxxx)

| Cap offset | Type | Status |
|---|---|---|
| 0x100 | VSEC ID=0x0001 Rev=1 Len=0x10 | **body all zeros** (no debug interface on retail) |
| 0x150 | AER | UnsupReq+ sticky (set by theta.1's blanket BAR scan that rebooted us) |
| 0x200 | Resizable BAR | BAR0 16GB (resize options up to 16GB), BAR2 256MB |
| 0x240 | Power Budgeting | present |
| 0x2a0 | ACS | present |
| 0x2d0 | PASID | Enable+, Max width 10 bits |
| 0x320 | LTR | max 1048576 ns |
| 0x410 | Physical Layer 16.0 GT/s | present |
| 0x450 | Lane Margining | present |

### 7.4 PSP firmware structure

The file `/lib/firmware/amdgpu/psp_13_0_10_sos.bin.zst` (165,703 B zstd-compressed)
decompresses to 360,352 bytes and contains **16 sub-blobs** delimited by `$PS1`
magics. **None of the sub-blobs are encrypted** on this consumer dGPU - max
entropy across all of them is 6.98 (Thumb-2 code density). Compare to
`psp_13_0_6/12/14` which show entropy 7.95+ (true AES-encrypted) at the same
offsets.

| # | offset | size | role |
|---|---|---|---|
| 0 | 0x00100 | 7,488 B | KDB (key database) |
| 1 | 0x01e40 | 61,696 B | SYS_DRV variant A (the PSP kernel/OS we decoded) |
| 2 | 0x10f40 | 94,448 B | SOS (ARM vectors + R1 + R2 user code) |
| 3 | 0x28030 | 2,232 B | small driver fragment |
| 4 | 0x288e8 | 12,536 B | small driver |
| 5 | 0x2b9e0 | 11,392 B | driver |
| 6 | 0x2e660 | 50,612 B | SYS_DRV variant B (different chip) |
| 7 | 0x3ac14 | 14,388 B | DBG_DRV variant |
| 8 | 0x3e448 | 5,180 B | small driver |
| 9 | 0x3f884 | 12,892 B | driver (lots of padding) |
| 10 | 0x42ae0 | 17,000 B | DBG_DRV variant |
| 11 | 0x46d48 | 16,024 B | DBG_DRV variant |
| 12 | 0x4abe0 | 28,928 B | INTF_DRV variant |
| 13 | 0x51ce0 | 2,304 B | small driver |
| 14 | 0x525e0 | 16,640 B | DBG_DRV variant |
| 15 | 0x566e0 | 6,336 B | small driver |

The file packages drivers for multiple Navi 3x chip variants in one firmware
image. Sub-blob #2 (SOS) is the primary target of our RE work.

### 7.5 Sub-blob #2 (SOS) layout

```
+0x00000-+0x00100   signature + $PS1 header (SHA-256 hash at +0xD0..+0xEF)
+0x00100-+0x00500   ARM exception vector table + handlers (PLAINTEXT)
+0x00500-+0x06D00   R1 user-mode Thumb-2 code (PLAINTEXT)
+0x07000-+0x14FFF   *** ALL ZEROS - empty padding ***
+0x15000-+0x17000   R2 user-mode Thumb-2 code (PLAINTEXT, ~16 KB)
+0x17000-+0x170d0   trailing data
```

Hypothesis-from-eta.3 ("57 KB encrypted body holds the DRAM trainer") is
**REFUTED**. There is no encrypted body — the `+0x07000..+0x15000` region
is zero padding (`captures/sos_encrypted_body.bin` is 99.9 % zeros).

> **CORRECTION (2026-06-11):** the follow-on claim "the DRAM trainer lives
> in PSP boot ROM (on-die silicon)" is **also wrong**. The DRAM trainer is
> **plaintext code in `INTF_DRV`** (bring-up routine at offset `0x2f80`),
> loaded by the `LOAD_INTFDRV` BL command at SRAM `0x00200000`. The training
> signature (`0xaa55aa55`, UMC reg `0x81041e54`) is present only in INTF_DRV
> and absent from the entire SOS blob. See §14.

### 7.6 VBIOS / Expansion ROM

The Expansion ROM at `0xf6c00000` (size 128 KB) contains the ATOM-format
VBIOS image. We dumped + parsed it in eta.3a. Key findings:

- ATOM v2.1 image header at start
- `ATOM_CMD_INIT` (AsicInit, index 0) = 256-byte stub with 3 BIF writes,
  no PHY work
- `firmware_info` data table: contains capability bits including
  `ATOM_FIRMWARE_CAP_ENABLE_2STAGE_BIST_TRAINING = 0x400`
- `umc_info`, `vram_info` tables present but consumed only by encrypted
  boot ROM, not by ATOM bytecode
- `indirectioaccess` table: 8 bytes (effectively empty) on retail Navi 32
- No ATOM bytecode trainer for DRAM (demoted RDNA2 -> RDNA3 to boot ROM)

### 7.7 ARM exception handler bytes (the fault PC region)

At SOS sub-blob file offset `0x3f0` (virt `0x0d1102f0`), the Data Abort
handler entry:

```
0x0d1102f0: 40 00 0c f1   cpsid f                ; disable FIQ during handler
0x0d1102f4: d8 d0 9f e5   ldr  sp, [pc, #0xd8]   ; <-- THE FAULT PC we observe
0x0d1102f8: 0d 00 5e e1   cmp  lr, sp
0x0d1102fc: 00 00 00 1a   bne  #0xd110304
0x0d110300: 00 f0 5e e2   subs pc, lr, #0        ; exception return
0x0d110304: 08 e0 4e e2   sub  lr, lr, #8        ; recover faulting PC
0x0d110308: 80 d0 9f e5   ldr  sp, [pc, #0x80]
0x0d11030c: 00 d0 9d e5   ldr  sp, [sp]
0x0d110310: 03 00 2d e9   stm  sp, {r0, r1}
...
```

The reported PC `0x0d1102f4` is the handler's own entry PC (self-mark), NOT
the faulting instruction. The handler reports its own location to indicate
"a fault was caught at this handler."

## 8. Hypotheses tested, all with empirical results

| Hypothesis | Test slice | Result |
|---|---|---|
| `PSP_BL__DRAM_LONG_TRAIN` trains DRAM | eta.3k-1 | NO - ack'd as stub in 1 ms |
| Same with proper `C2PMSG_36` buffer offset | kappa.1 | NO - identical no-op stub |
| `ATOM_CMD_INIT` trains DRAM | eta.3m | NO - 256-byte stub, no PHY work |
| Any SMU MSG triggers training | eta.3k SMU audit + iota.E | NO - no such MSG exists for v13_0_7 |
| Direct UMC v8.10 MMIO programming | iota.A | NO - PHY regs not in public headers |
| UEFI POST trains DRAM | eta.3f | NO - display init only |
| NBIO BIF FB gating helps | eta.3e-6 | NO - identical SOS stall |
| Replay all amdgpu pre-PSP MMIO writes | eta.3i | NO - identical SOS stall |
| IH ring initialization helps | eta.3e-2/3 | NO - SOS doesn't reach IH stage |
| HDP atomic power enable | eta.3e-7 | NO |
| SPL_TABLE / RAS_DRV BL stack | eta.3e-7 | NO |
| Sysmem at IOVA 0x83f00f80 satisfies fault | eta.3o | NO - fault is PSP-MMU, not IOMMU |
| PSP SRAM externally readable | theta.1 (rebooted system!) / theta.2 (safe) | NO |
| Patching plaintext SOS firmware | theta.4 | NO - SOS self-verifies, silent halt |
| Public PSP_v13 decryption key exists | iota.C | NO - only Zen CPU PSP attacked publicly |
| GFX bypass cold init (no PSP) | iota agent | NO - MEC IMEM gated by PSP-mediated security cage |
| `fw_load_type = DIRECT (0)` bypasses PSP | iota.D | NO - still needs PSP for TMR + secure pages |
| `fw_load_type = RLC_BACKDOOR_AUTO (3)` | iota.D | NO - PSP still triggers via GFX_CMD_ID_AUTOLOAD_RLC |
| All PSP_BL command space brute force | eta.2 / eta.3 / iota.D | NO - only 8 commands documented + working |
| PCIe VSEC has PSP debug interface | iota.D + sudo lspci | NO - body all zeros on retail |
| PCIe BIST | iota.D | NOT CAPABLE - config 0x0F = 0 |
| FLR / SBR / D3cold / hot reset | feedback memory | NO - PSP aux power survives all software resets |
| SMU-Mode1 reset bypass | iota.F | UNTESTED (speculative; same risk as theta.1) |
| Encrypted SOS body decryption | lambda.1 | N/A - no encrypted body exists on this chip |

19+ decisive negatives. The wall consistently resolves to "PSP boot ROM"
which is on-die silicon ROM signed by AMD at fab.

## 9. Where the wall actually is (ξ.1 — **SUPERSEDED by §14, 2026-06-11**)

> **CORRECTION (2026-06-11):** This entire section is unreliable. Its
> central claims — that SOS halts at `svc #0xe` (virt `0x0d126e9e`) and
> that the worker thread's code is "NOT in any plaintext sub-blob we
> possess… almost certainly from the encrypted SOS body" — are both
> **false**:
>
> 1. The address `0x0d126e9e` does not hold `svc #0xe`; it holds
>    `bl 0xd12700c` (Thumb-alignment misread). The real `svc #0xe` sites
>    are `0x0d12655c` and `0x0d126d9e`, both host-command idle loops.
> 2. There is no encrypted SOS body (it is 99.9 %-zero padding). The
>    worker / DRAM-training code is plaintext in `INTF_DRV` at SRAM
>    `0x00200000`.
>
> Read §14 instead. The text below is retained only for provenance.

The previous "PSP boot ROM unhandled-exception trap" hypothesis is **WRONG**.
Section 8's "Sysmem at IOVA 0x83f00f80 satisfies fault" line is also a
misattribution — the fault triplet was never a runtime fault, it's a
pre-existing cold-boot value (see §11 correction table).

### The actual wall, identified by static RE of the full 109,216-byte SOS:

```
SOS execution after LOAD_SOSDRV:

  virt 0x0d126e84  main loop entry
       |
       v
  virt 0x0d126eb4  main_init():
                    svc #0x42  map MP1 SMN region (0x03010000, 0xb10)
                    svc #0x42  map MP0 SMN region (0x03200000, 0xb48)
                    svc #0xb4  get_chip_tier byte
                    bl    0x0d12630c  report_version()
                              |
                              v
                       writes 0x00320d17 to MP0+0x9e8 (C2PMSG_58)
                              |  ← we observe this within ~50 ms
                              v
                    clears DEBUG_STATUS at MP0+0xd8 (SMN 0x032000d8)
                              ← why SMN_DEBUG_STAT reads 0 post-stall
                    svc #0x21  alloc base-handle
                    svc #0xb   x 4  alloc 4 IPC/event channels
                              -> ctrl[+0xc/+0x10/+0x14/+0x18]
                    svc #0x42  map 32 KiB blob at (handle + 0x85000)
                    bl    0x0d126fd0  PLAIN MEMCPY of tagged sub-records
                                      from sysmem -> PSP SRAM cursor
                                      (ξ.2 / ξ.3: confirmed not decompression,
                                       not decryption — just paged memcpy.
                                       Per ξ.2 disasm: map_physical/
                                       memcpy/unmap_virtual loop, no crypto.)
                    svc #1     create_thread(entry=virt 0x00202460, ...)
                              ← THE REAL WORKER ENTRY (corrected from
                                prior 0x00202f80 misattribution; ξ.3
                                literal pool revealed actual entry).
                                Worker code lives in record that fills
                                SRAM 0x00200000..0x00202f28 (~12 KiB
                                BEFORE PS1_1 lands). That record is
                                NOT in any plaintext sub-blob we possess.
                    ret
       |
       v
  virt 0x0d126e9e  svc #0xe   ← THE WALL.  wait_for_event() — blocks here
       |          [identical wait-then-yield pattern is also visible inside
       |           PS1_2 at SRAM 0x00203d3c: svc #0xe + svc #0x20
       |           polls status != 0x50 forever]
       |
       v
   (never returns; never posts SoL; never trains DRAM)
```

The event SOS is waiting for would be posted by the **worker thread at virt
`0x00202460`** — the actual DRAM training code. That thread is spawned
by PS1_2 (an SOS-plaintext dispatcher at SRAM `0x002037dc`) which itself
calls into SOS plaintext, but the worker code at `0x00202460` is NOT in
any of the four inner sub-records we extracted (`$KDB`, `$PS1` ×2, `$XF9`,
`$TOC`). It must come from a sibling record sourced by the lookup helper
(`0x0d128724`) that we never directly extracted — almost certainly from
the encrypted SOS body. On warm boot that worker succeeds and posts the
IPC event; on cold boot it doesn't. We cannot see its code.

### What PS1_1 and PS1_2 actually are (post-ξ.3)

The two plaintext sub-records we *did* extract turned out to be **plumbing**, not the trainer:

| Record | File offset | SRAM range | Purpose | DRAM-relevance |
|---|---|---|---|---|
| **PS1_1** | `+0x17100` | `0x00202f28..0x002037dc` | Resource-setup helper. Allocates event channels (`svc #0xb` ×5), thread handles (`svc #7`), kernel mem (`svc #0x25`). | **None.** Zero MMIO writes. Just kernel-object allocation. |
| **PS1_2** | `+0x179b8` | `0x002037dc..0x00204618` | Dispatcher/parent. Maps 8 phys regions, spawns 2 worker threads (`svc #1`), waits via `svc #0xe`+`svc #0x20`. | **None.** Zero MMIO writes. Orchestrates other threads but doesn't touch UMC/PHY. |

PS1_2's literal pool at virt `0x00204030..0x0020403c` contains the spawn
parameters as a 4-tuple `(entry1, stack1, entry2, stack2)`:

```
entry1 = 0x00202f31  (Thumb-bit; entry = 0x00202f30 = PS1_1 helper)  ← we have
stack1 = 0x00206f78
entry2 = 0x00202461  (Thumb-bit; entry = 0x00202460 = THE WORKER)    ← we DON'T have
stack2 = 0x002062a0
```

### Why no external unblock is possible (ν.9-cold-fresh confirmed)

Three independent attempts to find a host-side write channel into the
PSP-internal state required to unblock the wait — all decisively closed:

1. **Fake host doorbell → IRQ → wake `svc #0xe`**. Event channel IDs are
   runtime-allocated by `svc #0xb` and stored in PSP SRAM `ctrl[+0xc..
   +0x18]`. We have no host-side handle to those channel IDs.
2. **Write to PSP SRAM `[0x00205044]`** (the event status byte that PS1_2
   polls). PSP SRAM is not host-mapped at any phase (θ.2).
3. **Write to MP0 `SMN 0x032000d8`** (the real DEBUG_STATUS register).
   **ν.9 (wedged-state) + ν.9-cold-fresh both confirm**: the register is
   silently write-masked from host MMIO. Mask is silicon/BL-level, NOT
   SOS-set (cold-fresh test verified — the mask is present BEFORE SOS
   ever runs). Same write-mask behavior as `C2PMSG_60/61/62`.

### Implications (finalized)

- **The wall is one byte.** `[0x00205044]` flipping from `0x50` to `0x00`
  would unblock PS1_2's wait → wake SOS → assert SoL → train DRAM.
- That byte lives in **PSP private SRAM**, on the GPU die, behind silicon-
  level write isolation from the host PCIe surface.
- Hardware fault injection is the only path that reaches it. The minimum-
  cost shot would be voltage-glitching the GPU's `VDDCR_SOC` rail to flip
  the `cmp r0, #0x50` result at PS1_2's poll loop (SRAM `0x00203d44`).
  Practical cost: ~$300 ChipWhisperer + 3–6 months bench work + risk of
  bricking the card. Out of scope for this project.
- Software-only path: **definitively exhausted across 30+ tested
  hypotheses** through μ/ν/ξ slices. The closure is grounded in concrete
  RE-derived knowledge, not exhaustion.

## 10. Repository assets produced by this investigation

### 10.1 RE tools (in `tools/`)

| Tool | Purpose |
|---|---|
| `extract_all_psp_subblobs.py` | Original 9-sub-blob extractor (superseded) |
| `reextract_all_subblobs.py` | Correct 16-sub-blob extractor ($PS1-magic-based) |
| `lambda_full_plaintext_search.py` | Constant search across all sub-blobs |
| `lambda3_find_offset_stores.py` | Disasm-based store-offset search for C2PMSG writers |
| `lambda4_decode_new_debug_writers.py` | Decode R2 region kernel init code |
| `lambda5_find_all_map_physical.py` | Enumerate SVC #0x42 map_physical sites |
| `walk_sys_drv.py` | SYS_DRV characterization |
| `disasm_sys_drv.py` | SYS_DRV plaintext disassembly |
| `decode_sos_dispatcher.py` | SOS command-table tbb decoder |
| `decode_sos_kernel_entry.py` | Reset/SVC/Common-logger disassembly |
| `disasm_sos_exception_vectors.py` | ARM vector table decoder |
| `find_sram_0xcbc_owner.py` | Locates SOC_DRV exception report fn |
| `find_kernel_main_driver.py` | Driver-mapping solver for SOS jumps |
| `decode_svc_42.py` | SVC #0x42 (map_physical) call-site decoder |
| `walk_r2_svc.py` | SVC API enumeration from sub-blob R2 region |
| `find_svc_f6_handler.py` | SVC #0xf6 handler search |
| `find_psp_page_table.py` | Exhaustive search for PT-shape data (zero hits) |
| `extract_sos_smu_msgs.py` | SMU MSG enumeration |
| `find_kernel_exception_handler.py` | Locate C2PMSG write sites |
| `decode_psp_header.py` | `$PS1` header field decoder |

### 10.2 Live-GPU smokes (in `examples/`)

| Smoke | Purpose |
|---|---|
| `phase3_eta3k1_dram_long_train.mlr` | First DRAM_LONG_TRAIN attempt |
| `phase3_eta3k2_sos_state_probe.mlr` | Read-only SOS state probe |
| `theta2_ip_bounded_psp_probe.mlr` | Safe IP-bounded BAR5 probe (validates safety methodology) |
| `theta4_patched_sos_load.mlr` | Patched-firmware silent-reject test |
| `kappa1_dram_long_train_with_buf.mlr` | DRAM_LONG_TRAIN with correct buffer payload |
| `theta1_psp_sram_probe.mlr.CAUSED_REBOOT_DO_NOT_RUN` | DO NOT REBUILD - blanket BAR scan caused MCE |
| `phase3_eta3e7_full_chain_load_sos.mlr` | Complete BL chain + LOAD_SOSDRV reference |

## 11. Reproduction of the investigation

Each slice in the table below produces a deterministic, recoverable result.
Cold-cycle the system (PSU off > 15 s) before any cold-boot test. Warm-state
tests need amdgpu owning the dGPU - temporarily disable
`/etc/modprobe.d/vfio-dgpu.conf` to allow that.

### 11.1 Read-only cold-state probe

```bash
# After cold cycle, with vfio-pci default-bound:
sudo ./build/theta2_ip_bounded_psp_probe   # safe IP-bounded BAR5 scan
sudo ./build/phase3_eta3k2_sos_state_probe # read MP0 mailbox state
```

### 11.2 Warm handoff probe

```bash
# Disable vfio default-bind, cold-cycle, boot
sudo mv /etc/modprobe.d/vfio-dgpu.conf{,.disabled}
# reboot, PSU off > 15s

# After warm boot, dGPU on amdgpu, SOS running
readlink -f /sys/bus/pci/devices/0000:03:00.0/driver  # expect amdgpu
sudo dmesg | grep -iE "psp|bootloader" | tail

# Hand off to vfio (PSP state survives software unbind)
sudo modprobe vfio-pci
echo "1002 747e" | sudo tee /sys/bus/pci/drivers/vfio-pci/new_id
echo 0000:03:00.0 | sudo tee /sys/bus/pci/drivers/amdgpu/unbind
echo 0000:03:00.0 | sudo tee /sys/bus/pci/drivers/vfio-pci/bind

sudo ./build/theta2_ip_bounded_psp_probe   # expect warm SoL = 0x016ed035

# Restore default vfio-bind for next reboot
sudo mv /etc/modprobe.d/vfio-dgpu.conf{.disabled,}
```

### 11.3 Firmware extraction + analysis (no GPU needed)

```bash
# Decompress firmware to /tmp
mkdir -p /tmp/mlrift_fw
zstd -d -k -f /lib/firmware/amdgpu/psp_13_0_10_sos.bin.zst \
    -o /tmp/mlrift_fw/psp_13_0_10_sos.bin

# Extract all 16 sub-blobs
python3 tools/reextract_all_subblobs.py

# Full plaintext analysis
python3 tools/lambda_full_plaintext_search.py
python3 tools/lambda3_find_offset_stores.py
python3 tools/lambda4_decode_new_debug_writers.py
python3 tools/lambda5_find_all_map_physical.py
```

## 12. Conclusion (**revised 2026-06-11**)

The previous conclusion — "the cold-boot wall is real, located at
`svc #0xe`/`0x0d126e9e`, and software-only unattackable" — does **not
hold**. Both load-bearing facts were wrong (Thumb misread; no encrypted
body). See §14. The corrected position:

- The DRAM-training / hardware bring-up code is **plaintext and in our
  possession** — thread PS1_1 inside `INTF_DRV`. It is fully analyzable.
- Cold-boot is therefore **not** gated by silicon-rooted security or by
  inaccessible firmware. The open question is purely a software delta
  between MLRift's from-cold BL chain and amdgpu's (which boots this card
  cold every power-on).
- Hardware fault injection / AMD cooperation / SRAM exfiltration are **no
  longer required** to make progress; finishing the static RE of the
  INTF_DRV path is.

The warm-hybrid path still satisfies the LLM mission independently:
amdgpu does the PSP-mediated firmware loading once, MLRift takes over via
VFIO handoff. MLRift's execution path contains no PyTorch, no ROCm, no
hipcc. Shipping deliverable, competitive performance: Mistral-7B
mega-kernel at +24% over PT bf16, Qwen3 200+ tok/s, bit-exact validation
across 7 model families.

The research artifacts (20+ RE tools, complete SOS plaintext decode,
firmware-cap audit, durable reference values for warm/cold state, full
μ/ν/ξ campaign infrastructure, plus the `psp_fuzz/redis.py` capstone
harness) remain independently valuable, and now anchor an active —
not closed — cold-boot track.

---

## 13. μ/ν/ξ campaign (final closure work)

After §1-§12 reached "wall is real," a follow-up campaign was run to
verify nothing was missed. It produced one major correction set and
the definitive halt-point identification in §9.

### 13.1 μ.1-μ.E — `$PS1` parser fuzz campaign

Built `psp_fuzz/` infrastructure (gitignored): 415-case corpus, smokes,
runner scripts, ranker. Findings:

| Slice | Result |
|---|---|
| μ.A | BL fails silently on malformed input. No diagnostic in any C2PMSG. Wedge after one malformed cmd; recovery requires PSU cycle. |
| μ.baseline | Cold-boot register baseline established. C2PMSG_60/61/62 = 4/0x83f00f80/0x0d1102f4 IS the *default cold-boot state*, set pre-our-smoke (UEFI POST or boot-ROM POR). Not a runtime fault report. |
| μ.C | Triplet C2PMSG_60/61/62 is PSP-write-only via the host-MMIO path (write attempts silently dropped). |
| μ.D | Malformed LOAD_SOSDRV → same silent-wedge pattern as malformed LOAD_KDB. Uniform BL fail-closed behavior across command types. |
| μ.E | Sub-millisecond timing channel. Across 8 cases / 5 mutation classes: 15-22 µs spread (3 clusters: hash bitflip ~15 µs, intermediate ~18 µs, length/magic/truncation ~21 µs). Graded but not exploitable. |

### 13.2 ν series — directed re-verification

| Slice | Result |
|---|---|
| ν.1 | Replay with all 3 previously-skipped UMC writes re-included → BL chain GREEN, SOS stalls bit-identically. 3 UMC writes are NOT the missing piece. Skip filter now reduced. |
| ν.2 | amdgpu source audit found VBIOS `ENABLE_2STAGE_BIST_TRAINING` bit (firmware_capability bit 6) **OFF** on this card — so `psp_mem_training(PSP_MEM_TRAIN_COLD_BOOT)` is never invoked by amdgpu either. |
| ν.3 | Manual SMU MSGs pre-BL (TestMessage, GetSmuVersion, PowerUpDF, EnableAllSmuFeatures): **all timeout**. SMU at cold boot is in ROM mode and does not accept new mailbox MSGs until PSP loads SMU FW (which happens *during* SOS init). |
| ν.5 | 5-agent independent audit. 4/5 confirmed wall is real; 1 (doc audit) flagged real diagnostic errors. |
| ν.6 | Read SMN `0x032000d8` (the *real* DEBUG_STATUS register): reads `0x00000000`. SOS helper @ `0x1d50` never writes it cold. |
| ν.7 | SPL_TABLE re-test in modern (ν.1-baseline) chain: SPL load GREEN, BL chain GREEN, SOS still stalls bit-identically. SPL is not the missing piece. |
| ν.8 | Full 128-register MP0 SMN sweep. Found `C2PMSG_59/89/90/91/115/118` populated — amdgpu source review showed these are BL-side artifacts (`C2PMSG_115` is SPI flash update mailbox `MBOX_READY`), not SOS-runtime indicators. |
| **ξ.1** | **Static RE of full 109,216-byte SOS** (prior captures were truncated to 94,448 B). Claimed exact halt at virt `0x0d126e9e` (`svc #0xe`). **RETRACTED 2026-06-11 (§14):** that address holds `bl 0xd12700c`, not `svc #0xe` — a Thumb-alignment misread. |

### 13.3 Corrections to earlier doc claims (audit-driven)

| Earlier claim | Correction |
|---|---|
| "Cold C2PMSG_58 = `0x80320d17`, warm = `0x00320d17` (bit 31 = SOS wrote)" | **Wrong.** Cold = `0x00000000`. Post-LOAD_SOSDRV stall = `0x00320d17`. Bit 31 framing was a doc fabrication. The value is simply SOS's firmware version label written by `report_version()`. |
| "SOS hits a Data Abort at virt `0x83f00f80`, dumps triplet, halts" | **Wrong.** Triplet is pre-existing cold-boot state. SOS does not fault here — it executes through `report_version()`, completes init, then blocks at `svc #0xe`. |
| "η.3o decisively proved the fault is PSP-CPU-side MMU" | **Status changed**: untestable. η.3o experiment (map sysmem at IOVA `0x83f00f80`) saw bit-identical triplet — but that was because the triplet pre-existed; the experiment proved nothing. |
| "θ.4 unpatched cold writes `0x80320d17` in ms" | **Wrong baseline.** Unpatched cold writes `0x00320d17`. Qualitative conclusion (patched stays 0, unpatched advances to `0x00320d17`) still holds. |
| "16 sub-blobs in `psp_13_0_10_sos.bin`" | **Wrong.** TOC has 9 real entries. Our `reextract_all_subblobs.py` had 7 false positives (incidental `$PS1` bytes inside larger blobs). True 9 entries: KDB / SYS_DRV-NV32 / SOS-NV31 / RL / SOC_DRV / INTF_DRV / TOC / DBG_DRV / SPL. |
| "Sub-blob #2 (SOS) is 94,448 bytes" | **Wrong size.** Per TOC, SOS is **109,216 bytes**. `captures/sos_full_109216B.bin` has the corrected extraction. Smokes were always loading the right 109,216 B via `psp_v2_find`; only the captured file was truncated. |
| "MP0_SMN C2PMSG_58 = DEBUG_STATUS" | **Wrong.** `C2PMSG_58` = SOS fw version. The real DEBUG_STATUS is at SMN `0x032000d8` (dword 0x36), distinct from C2PMSG_58. SOS *clears* DEBUG_STATUS at virt `0x0d126eec` early in init. |

### 13.4 Final hypothesis tally

| Cumulative hypotheses tested / ruled out | Count |
|---|---|
| Pre-μ series | 19+ |
| μ-series (fuzz observability) | +3 (silent halt, constant time, no exploit channel) |
| ν-series (directed re-verification) | +8 |
| Total | **30+** |

### 13.5 Repository additions (μ/ν/ξ era)

All gitignored — local only.

```
psp_fuzz/
  gen_corpus.py            μ.1 — corpus generator (--sos for SOS-derived)
  rank_cases.py            μ.rank — priority-ranked test queue
  audit_replay_blob.py     classifies amdgpu_replay.bin writes per IP
  audit_firmware_capability.py  parses ATOM firmware_info for ENABLE_2STAGE_BIST_TRAINING bit
  audit_vram_info.py       parses VBIOS vram_info data table
  corpus/                  413 KDB-derived test cases + manifest
  sos_corpus/              415 SOS-derived test cases + manifest + ranked.txt
  smoke/
    mu_a_reusability_probe.mlr
    mu_baseline_probe.mlr
    mu_c_triplet_provenance.mlr
    mu_d_sos_parser_fuzz.mlr
    mu_e_sos_parser_timing.mlr
    nu1_full_umc_replay.mlr
    nu3_smu_msgs_pre_bl.mlr
    nu6_smn_debug_status_probe.mlr
    nu7_spl_in_modern_chain.mlr
    nu8_full_c2pmsg_scan.mlr
  scripts/
    stage_fuzz_inputs.sh / stage_sos_case.sh / next_case.sh
  results/                 timestamped run logs

captures/
  sos_full_109216B.bin     CORRECT full SOS extraction (per TOC), not truncated
```

### 13.6 Final ranked candidate exhaustion (last 3 evaluated)

Per Gemini's 4-angle framing earlier in session, plus 5-agent audit:

| Angle | Status |
|---|---|
| SMU power-gate prerequisite | Ruled out (ν.3 — SMU in ROM mode cold, MSGs timeout) |
| IOMMU / physical address trap | Ruled out (BL chain GREEN proves PSP DMA works) |
| ATOM table host execution | Ruled out (η.3m — AsicInit is 256-byte stub; ν.2 — firmware_cap gate OFF) |
| Polling window timing | Ruled out (μ.E sub-µs timing trace, 50 ms granularity poll across 15 s) |
| SPL_TABLE pre-SYSDRV (agent #1 lead) | Ruled out (ν.7 — loads GREEN, no effect on stall) |
| Wrong SOS sub-blob | Ruled out (TOC parse — `psp_v2_find` returns correct entry; NV31 label is internal) |
| Reading wrong DEBUG register | Verified (ν.6 — real reg is `0x032000d8`, reads 0 because SOS clears it) |
| Static RE of post-version-write code | **RETRACTED (§14, 2026-06-11)**: the ξ.1 "`svc #0xe` at `0x0d126e9e`" reading was a Thumb misread; the worker/DRAM-trainer is plaintext `INTF_DRV` (PS1_1 @ `0x00202f80`), not an inaccessible `$KDB` thread |
| Early-attach mmiotrace (ν.4) | Superseded by ξ.1 |

---

## 14. Re-audit (2026-06-11) — the wall premise is false; trainer is plaintext

A fresh, independent static re-audit of the §9/ξ.1 closure found that the
two facts the "software-only unattackable / silicon wall" conclusion rests
on are both wrong. All work below is static (no GPU, no cold cycle, zero
risk), reproduced from `captures/` with `psp_fuzz/redis.py` (capstone 5.0.7).

### 14.1 There is no encrypted body

`captures/sos_encrypted_body.bin` (57,344 B = the SOS `+0x07000..+0x15000`
region §9 attributes the worker to) is **99.9 % zeros, entropy 0.02** — it
is padding, not encrypted code. §7.4/§7.5/§8 already say no encrypted body
exists on this consumer part; §9's "almost certainly from the encrypted SOS
body" contradicts the rest of the document and is unsupported.

### 14.2 `svc #0xe` at `0x0d126e9e` is a Thumb-alignment misread

Disassembling the full 109,216-B SOS (virt→file: `V − 0x0d110000 + 0x100`):

| virt | actual instruction | §9 claim |
|---|---|---|
| `0x0d126e9e` | `bl 0xd12700c` | "`svc #0xe` — THE WALL" (wrong) |
| `0x0d126eac` | `svc #1` (create_thread, entry `0x00202f80`) | — |

SOS issues exactly **one** `svc #1` (create_thread), spawning entry
`0x00202f80`. The only real `svc #0xe` sites are `0x0d12655c` and
`0x0d126d9e`, and both are **host-command idle dispatch loops**:

```
0x0d12655c: svc #0xe        ; wait for event
0x0d12655e: svc #0x40       ; poll for a posted host command -> r0
0x0d126560: cmp r0, #0
0x0d126562: beq 0x0d12655c  ; nothing posted -> keep waiting
```

i.e. SOS waiting on the **host** after init completes — not a one-shot wait
on a worker. Since cold-boot leaves SoL (C2PMSG_81) at 0, SOS is halting
**in init, before SoL**, not parked in this loop.

### 14.3 The DRAM trainer is plaintext in `INTF_DRV`

**Proven (byte-level):** the hardware bring-up / DRAM-training code is
plaintext in **`INTF_DRV`** (`captures/intf_drv_subblob.bin` == subblob_12,
28,928 B; the `LOAD_INTFDRV` = `0x000D0000` component the BL chain already
loads — Step 4 of `phase3_eta3e7_full_chain_load_sos.mlr`). Loaded at PSP
SRAM base `0x00200000`, INTF_DRV decodes as a coherent driver. The training
signature is present **only in INTF_DRV and absent from the entire SOS
blob**: the pattern `0xaa55aa55` and UMC register `0x81041e54` appear in
INTF_DRV's literal pool (offsets `0x3008` / `0x3004`); neither byte sequence
occurs anywhere in `sos_full_109216B.bin`. INTF_DRV contains **no** `svc #1`
/ `svc #0xe` / `svc #0x40` — it is pure driver code, not a thread/wait owner.

The decoded bring-up routine (INTF_DRV offset `0x2f80`) does:

```
; mailbox handshake
mov.w r1,#0x80000000 ; str r1,[r4]
loop: ldr r0,[r4] ; ubfx r0,r0,#0x10,#8 ; cmp r0,#0xe ; bne loop   ; wait state==0xe
; register writes via svc #0xa5  (r0=addr, r1=val, r2=size)
svc #0xa5  0x0000d8b8 = 0x00000000
svc #0xa5  0x0c9100c4 = 0x00000040
svc #0xa5  0x81041e54 = 0xaa55aa55      ; <- classic memory-training pattern
svc #0xa5  0x8100009c = 0xaa55aa55      ; <- UMC/DF/PHY register
; plus svc #0xc1/#0xc4/#0xc5/#0xb8 (HW control), svc #0xa7 (SMN r/w)
```

A companion helper (INTF_DRV offset `0x2460`) triggers an op via `svc #0x92`
and polls a done-flag at `[0x00d06004 + 3]` up to `r6` iterations (`svc #0x91`
delay between), returning timeout `0xffff3024` on failure. `0x00d06004` is the
op's byte-field control struct (`+1` = op id, `+2` = enable, `+3` = done).

This is precisely the mechanism §7.5/η.3m declared "genuinely UNKNOWN…
inside SOS encrypted body / boot-ROM silicon." It is a readable RDNA3
driver — that conclusion stands independent of any remaining detail below.

**Open detail (not yet resolved — do not over-read §9's "PS1_1 = plumbing"):**
SOS's single `svc #1` (create_thread, at `0x0d126eac`) spawns thread entry
`0x00202f80`, which maps to INTF_DRV offset `0x2f80` (the bring-up routine
above). However, the SOS-tail `$PS1` record at file `0x17100` is *different*
code (it does not contain the training writes) and, per §9, targets the
overlapping SRAM range `0x00202f28..0x002037dc`. So there is an unresolved
overlay question: at runtime, does SRAM `0x00202f80` execute INTF_DRV's
trainer, or an SOS `$PS1` record copied on top? §9 characterised that SRAM
as "plumbing, zero MMIO writes" — but it derived that from disassembling the
SOS-tail `$PS1` record, a *different blob* than INTF_DRV. Resolving which
bytes win at runtime (and thus the exact call path into the trainer) needs
tracing the SOS init memcpy (`bl 0xd126fd0`) destination/order vs the BL
load of INTF_DRV. Either way, the trainer code itself is plaintext and ours.

### 14.4 Corrected model and what is still open

```
LOAD_SOSDRV -> SOS init -> report_version (C2PMSG_58 = 0x00320d17)
            -> svc #1 create_thread(entry 0x00202f80 = INTF_DRV PS1_1)
                 PS1_1: mailbox handshake (poll state==0xe)
                        + UMC/PHY training writes (0xaa55aa55 ...)
                        + svc #0x92 ops w/ completion-poll (timeout 0xffff3024)
            -> (warm) training completes -> SOS asserts SoL (C2PMSG_81)
                 -> reaches host-command idle loop (svc #0xe @ 0x0d12655c)
            -> (cold, MLRift) SoL never set -> halt is INSIDE PS1_1 bring-up
```

**Open question (genuinely unresolved):** why PS1_1's bring-up completes
under amdgpu (which cold-boots this card every power-on) but not under
MLRift's from-cold BL chain. Leading candidates, both software and testable:

1. A precondition PS1_1 needs before its mailbox reaches state `0xe`
   (e.g. SMU/clock state — ν.3 found SMU is in ROM mode cold and rejects
   MSGs until PSP loads SMU FW during SOS init).
2. MLRift never actually reaching the PS1_1 spawn (a wrong/missing sub-blob
   fed to an earlier `LOAD_*`, so init faults before create_thread).

**Next static step:** finish RE of PS1_1 and the INTF_DRV functions it
calls to recover the full register/SMN bring-up sequence, and resolve the
`svc #0xa5`/`#0x92`/`#0xa7` kernel handlers (SYS_DRV dispatcher @ file
`0x50c4`, wrapped in a secure-channel: key-derive via SMN `0x30000004` +
paired AES/hash calls). That yields the literal cold-boot register script,
which MLRift can replay from `std/psp.mlr` or use to pinpoint the missing
precondition.

### 14.5 Specific corrections to earlier text

| Location | Earlier claim | Correction |
|---|---|---|
| Status, §9, §12 | "Wall at `svc #0xe`/`0x0d126e9e`; software-only unattackable; silicon wall." | False on both facts (Thumb misread; no encrypted body). Cold-boot is a software RE task, reopened. |
| §9 | Worker code "NOT in any plaintext sub-blob… almost certainly encrypted SOS body." | Worker/trainer is plaintext `INTF_DRV` at SRAM `0x00200000`. |
| §7.5 | "DRAM trainer lives in PSP boot ROM (on-die silicon)." | DRAM trainer is plaintext code in INTF_DRV (bring-up routine @ offset `0x2f80`); signature only in INTF_DRV, absent from SOS. |
| §9 | "SOS spawns worker `0x00202460` directly and waits on its IPC event." | SOS spawns one thread, entry `0x00202f80` (→ INTF_DRV bring-up). `0x00202460` is a downstream helper. The `svc #0xe` loops are host-command idle, not worker-waits. |
| §9 | "PS1_1 = plumbing, zero MMIO writes." | Derived from the SOS-tail `$PS1` record, a different blob than INTF_DRV; INTF_DRV's code at the same SRAM offset is the MMIO trainer. Runtime overlay still open (§14.3). |
| §13.6 | "η.3m: DRAM training mechanism genuinely UNKNOWN." | Mechanism is known and readable (INTF_DRV register/SMN bring-up sequence). |
