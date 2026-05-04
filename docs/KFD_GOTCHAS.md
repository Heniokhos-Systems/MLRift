# AMDKFD Direct-Dispatch Gotchas

Notes captured during Stage 2a (first MLRift kernel dispatched on gfx1100 via
raw AMDKFD ioctls, zero ROCm userspace) and revised during Stage 2b (back-to-back
multi-dispatch on a single queue). Each item below was a hard-won
discovery — none of them appear in AMD's user-facing documentation or the
`kfd_ioctl.h` comments. They came from reading the kernel driver source
(`drivers/gpu/drm/amd/amdkfd/kfd_mqd_manager_v11.c` and `kfd_queue.c`),
reading the tinygrad bare-KFD backend (`tinygrad/runtime/ops_amd.py`),
LD_PRELOAD-tracing `libhsa-runtime64`, and reading `journalctl -k` GPU-fault
logs after each hang.

If you're implementing your own KFD-direct path or porting to a new GPU
generation, start here.

**Tested target:** gfx1100 (RDNA3, Radeon 7900 XTX-class) on Linux 6.17 with
KFD v1.18. Items 1-3, 6, 7, 8 apply to gfx10/11 in general; items 4-5 are
gfx11-specific. Item 8 is AQL-specific (PM4 queues use a different
doorbell convention). Items 9-10 were uncovered during Stage 2b
(launcher fleet — see `docs/AMDGPU_NATIVE.md`).

---

## 1. `ALLOC_MEMORY_OF_GPU` requires a pre-reserved host `va_addr`

The `va_addr` field is documented as "to KFD" — i.e. the user-supplied virtual
address where the allocation lives. Passing `va_addr=0` lets the kernel pick.

**Reality:** passing `va_addr=0` is silently accepted and `ALLOC` returns a
handle, but the subsequent `MAP_MEMORY_TO_GPU` then fails with `EINVAL`.

**Fix:** reserve a host-side VA range before each ALLOC, pass the returned
address as `va_addr`:

```c
void *va = mmap(NULL, size, PROT_NONE,
                MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
// Pass (uint64)va as the va_addr field of kfd_ioctl_alloc_memory_of_gpu_args.
```

The `MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE` combination pins the address
range so unrelated allocations don't grab it; KFD then claims the same VA on
the GPU side via `MAP_MEMORY_TO_GPU`.

**Where:** `_kfd_reserve_va()` in `std/kfd.mlr`.

---

## 2. Host-visible `mmap` uses the DRM render-node fd, not `/dev/kfd`

The KFD docs imply that the `mmap_offset` returned by `ALLOC_MEMORY_OF_GPU`
should be passed to `mmap(/dev/kfd, ..., offset=mmap_offset)` to get a
host pointer.

**Reality:** mmapping with the KFD fd returns `ENODEV`. The mmap target is
the DRM render-node fd (`/dev/dri/renderD128`), not `/dev/kfd`.

```c
void *host = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED,
                  drm_render_fd,        // <-- NOT kfd_fd!
                  mmap_offset);
```

This is confirmed by reading ROCT-Thunk's `queues.c` — but the kfd_ioctl.h
docstring just says "mmap a render node" without making clear which fd.

**Where:** `dev_alloc_host_visible()` in `std/kfd.mlr`.

---

## 3. wp/rp must live INSIDE an `amd_queue_t` struct (not standalone BOs)

The `kfd_ioctl_create_queue_args` documentation says the kernel returns
`write_pointer_address` and `read_pointer_address` as "from KFD" out-fields.
The natural interpretation is that these are pointers into the AQL ring,
or — second-natural — that they are the start of dedicated 4 KiB BOs.

**Reality:** for AQL queues the CP firmware on gfx10/11 sets
`cp_hqd_pq_wptr_poll_addr_lo/hi = q->write_ptr` and
`cp_hqd_pq_rptr_report_addr_lo/hi = q->read_ptr` with
`SLOT_BASED_WPTR=2` (see `kfd_mqd_manager_v11.c::update_mqd`). The CP
treats those addresses as `&amd_queue_t.write_dispatch_id` and
`&amd_queue_t.read_dispatch_id` respectively — fields at fixed offsets
within an `amd_queue_t` struct (per `ROCR-Runtime/src/inc/amd_hsa_queue.h`).

**Symptom of getting this wrong:** the *first* dispatch on the queue
appears to work — CP picks up packet 0 because the doorbell-write path
primes the HQD with the initial wp value. But the *second* and any
subsequent dispatch on the same queue hangs forever in busy-poll: CP
re-polls `wptr_poll_addr` looking for a new write_dispatch_id and reads
garbage from a standalone BO that has no semantic meaning to firmware.

**Fix:** allocate one host-visible BO for an `amd_queue_t` struct
(0x100 bytes; we use a 4 KiB page). Initialize the fields the firmware
reads, and pass field-offsets as the wp/rp pointers:

```c
// Layout (offsets stable across gfx10/11/12):
//   amd_queue_t.hsa_queue                       at 0x00 (size 0x28)
//   amd_queue_t.write_dispatch_id (u64)         at 0x38   <-- wp_va
//   amd_queue_t.max_cu_id (u32)                 at 0x48
//   amd_queue_t.max_wave_id (u32)               at 0x4C
//   amd_queue_t.read_dispatch_id  (u64)         at 0x80   <-- rp_va
//   amd_queue_t.read_dispatch_id_field_base     at 0x88
//   amd_queue_t.queue_properties (u32)          at 0xB4

uint64_t qdesc_va = dev_alloc_host_visible(4096);  // GTT|COHERENT|UNCACHED
uint8_t* qd = host_ptr(qdesc_va);
*(uint32_t*)(qd + 0xB4) = 0x0A;     // IS_PTR64(2) | ENABLE_PROFILING(8)
*(uint32_t*)(qd + 0x48) = cu_count - 1;
*(uint32_t*)(qd + 0x4C) = waves_per_cu - 1;
*(uint32_t*)(qd + 0x88) = 0x80;     // read_dispatch_id offset
// Pass these to CREATE_QUEUE:
args.write_pointer_address = qdesc_va + 0x38;
args.read_pointer_address  = qdesc_va + 0x80;
```

`AMD_QUEUE_PROPERTIES_IS_PTR64` is **2**, not 1 — easy to mis-read; setting
this field to a wrong bit pattern causes silent firmware misbehavior.

Recent kernels' `kfd_queue_buffer_get` only checks the BO covers
`[user_addr, user_addr + 8)`; non-page-aligned offsets into a single BO
are accepted. (Older kernels may have differed; the Stage-2a-era claim
that wp/rp had to be *separate* page-sized BOs was empirically derived
from a different failure mode and was wrong.)

**Where:** `kfd_create_queue` in `std/kfd.mlr` — `qdesc_va`/`qdesc_host`
locals carry the descriptor; `wp_va`/`rp_va` are computed as offsets.

---

## 4. EOP / context-save / control-stack sizes are EXACT, topology-derived

The kernel docs imply you can pass any reasonable size for `eop_buffer_size`,
`ctx_save_restore_size`, and `ctl_stack_size`. ROCm headers show "typical"
values like 16 KiB EOP and 96 KiB ctx-save.

**Reality:** for gfx1100 (`gfx_target_version=110001`), the kernel computes
expected sizes from the GPU topology and rejects anything else with `EINVAL`:

| Field | gfx1100 wave32 value |
|---|---|
| `eop_buffer_size` | 4096 (4 KiB) |
| `ctl_stack_size` | 0x6000 (24 KiB) |
| `cwsr_size` (ctx-save data part) | 0x1B72000 (28,778,496 ≈ 27.4 MiB) |
| Underlying ctx-save BO size | `ALIGN((cwsr_size + debug) * num_xcc, PAGE_SIZE) = 28,839,936` |

Get any of these wrong → `CREATE_QUEUE` returns `EINVAL` with no further
indication of which size was wrong.

**Where:** `KFD_EOP_BYTES`, `KFD_CTX_SAVE_BYTES`, `KFD_CTL_STACK_BYTES`
constants in `std/kfd.mlr`.

These will differ on other GPU generations; for gfx10/gfx9 read the kernel's
`kfd_topology.c` and `kfd_chardev.c` (look for the queue-size validation).

---

## 5. Ring + EOP + kernel image ALL need `EXECUTABLE` flag

The `KFD_IOC_ALLOC_MEM_FLAGS_EXECUTABLE` bit is documented for the kernel
code segment. Naturally, you'd think only the `.text` allocation needs it.

**Reality:** GPU page-faults at runtime on three separate allocations if any
of them lacks the EXECUTABLE bit:

| Allocation | Why it needs EXECUTABLE |
|---|---|
| AQL ring | Command Processor Fetcher reads packets with execute permission |
| Kernel code (`.text` + `.kd`) | Shader instruction cache (SQC) requires it |
| EOP buffer | CP firmware writes end-of-pipe events with execute permission |

The faults are silent at the ioctl level — `CREATE_QUEUE` and dispatch both
return success. The only signal is in `journalctl -k`:

```
amdgpu: VM_L2_PROTECTION_FAULTS: VMID=11 PERMISSION_FAULTS=0x8 ... CLIENT=CPF
amdgpu: VM_L2_PROTECTION_FAULTS: ... CLIENT=SQC (inst)
```

The host-side observation is just a hung dispatch — the busy-poll never sees
the signal change.

**Canonical flag combos** (LD_PRELOAD-traced from `libhsa-runtime64`):

| Allocation | Flags |
|---|---|
| AQL ring | `0xD6000004` = `USERPTR \| WRITABLE \| EXECUTABLE \| NO_SUBSTITUTE \| COHERENT \| UNCACHED` |
| EOP / kernel image | `0xD0000001` = `VRAM \| WRITABLE \| EXECUTABLE \| NO_SUBSTITUTE` |

**Where:** `dev_alloc_aql_ring`, `dev_alloc_kernel_image`, `dev_alloc_vram_exec`
in `std/kfd.mlr`.

---

## 6. `KFD_IOC_ALLOC_MEM_FLAGS_AQL_QUEUE_MEM` is gfx7/8-only

The flag exists in `kfd_ioctl.h` and looks like the obvious choice for AQL
ring allocations. Documentation suggests it.

**Reality:** on gfx11 setting this flag halves the BO size and double-maps
the VA range to itself, creating a wraparound buffer for legacy hardware.
This causes weird "buffer is half the size you asked for" confusion.

`libhsa-runtime64` does NOT set this flag on gfx10+ — it allocates the AQL
ring as a normal USERPTR or GTT buffer with `EXECUTABLE` set (see #5).

**Fix:** drop `AQL_QUEUE_MEM` from the ring allocation flags on gfx10+.

---

## 7. `__sync_synchronize` is a GCC builtin, not a libc symbol

The HSA AQL packet release pattern says: write the body of the packet, then
do an `atomic_release` write of the header word so the GPU scheduler sees a
fully-formed packet.

The natural way to do this in plain C is `__sync_synchronize()` as a fence,
or `__atomic_store_n(..., __ATOMIC_RELEASE)`.

**Reality:** `__sync_synchronize` is a GCC compiler builtin that gets compiled
inline as an `mfence` instruction — there is no actual symbol named
`__sync_synchronize` in `libc.so.6` or anywhere else linkable. Trying to
declare `extern void __sync_synchronize(void);` and link it gives
`undefined symbol: __sync_synchronize` at runtime.

**On x86 you can skip the fence entirely.** x86 has strong store ordering:
single-core stores retire in-order, so the body stores naturally precede the
header store. Just write the body, then write the header, then write the
doorbell.

```c
// All packet body fields...
*(uint32_t *)(packet + 0x00) = (setup << 16) | header;  // header release
// Doorbell...
*(uint64_t *)(doorbell_host) = new_wp;
```

**On ARM or any weakly-ordered architecture,** you'd need a real `dmb ish`
inline asm before the header write. None of our targets need this today.

**Where:** `_kfd_fence()` in `std/kfd.mlr` is a no-op.

---

## 8. AQL doorbell value is `last_packet_index`, not `new_wp`

After writing an AQL packet at slot `S`, ring the doorbell to wake CP. The
"obvious" doorbell value, and the one ROCT-Thunk uses for **PM4** queues,
is `new_wp` (= `S + 1`, the next slot to be filled). For **AQL** queues
on gfx10/11 this is wrong.

**Reality:** the AQL doorbell on gfx10+ uses *last-valid-packet-index*
semantics (`= new_wp - 1 = S`). Writing `new_wp` tells CP "wait until
packet at index `new_wp` becomes valid" — that packet does not exist
yet, so the queue stalls.

**Symptom:** identical to gotcha #3's symptom (first dispatch works,
back-to-back dispatches hang). On the *first* dispatch after queue
creation, CP appears to ignore the off-by-one (the queue is in a fresh
state; CP processes whatever packets are present up to the index given,
inclusive or exclusive doesn't matter). On the *second* dispatch CP is
running and the off-by-one matters.

**Fix:**

```c
*(uint64_t *)doorbell_host = new_wp - 1;
//                                   ^^^ packet *index*, not the next-slot pointer
```

**Source:** tinygrad's bare-KFD AMD backend (`tinygrad/runtime/ops_amd.py`,
`AMDComputeAQLQueue::_submit` — `signal_doorbell(dev, doorbell_value=put_value-1)`).
HSA-Runtime's `amd_aql_queue.cpp` uses the same convention, hidden behind
the `kAql` queue-format branch.

**Where:** `kfd_dispatch` in `std/kfd.mlr`.

---

## Debug techniques that actually worked

1. **`journalctl -k | tail -30` after a hang.** The single most valuable tool
   for KFD debugging. GPU-internal page faults appear here with VMID,
   offending VA, and the client unit (CPF / SQC / TC). Neither dmesg, strace,
   nor ltrace surface KFD's complaints.

2. **LD_PRELOAD-trace libhsa-runtime64.** Build a small `.so` that intercepts
   `ioctl()` via `dlsym(RTLD_NEXT, "ioctl")`, log every KFD ioctl call with
   its args struct dumped via `xxd`, then run any HSA-using program (e.g.
   `rocm-smi`). You get the canonical ioctl sequence + flag combinations for
   free. This is how we discovered the `0xD6000004` / `0xD0000001` flag
   patterns in #5.

3. **Compare with a working reference C program at each step.** If the KFD
   layer fails, write the equivalent in raw C (with kfd_ioctl.h included),
   run that. If the C version also fails, the bug is in your understanding
   of the API. If it works, the bug is in the MLRift port.

4. **Diff the kernel source.** `linux/drivers/gpu/drm/amd/amdkfd/`. The
   `kfd_chardev.c` is the ioctl entry point; follow the `validate_*`
   functions for size/alignment checks; follow `kfd_queue_buffer_get` for
   wp/rp BO requirements (#3 came from there).

---

## 9. AQL `grid_size_*` is total work items, not num_workgroups

The HSA AQL kernel-dispatch packet has three `grid_size_x/y/z` fields
in addition to three `workgroup_size_x/y/z` fields. The natural
interpretation, especially after years of HIP/CUDA, is that
`grid_size = num_workgroups` (analogous to `gridDim`).

**Reality:** for AQL, `grid_size = num_workgroups × workgroup_size` —
the **total** number of work items. Setting `grid_size_x=1` and
`workgroup_size_x=16` doesn't launch one workgroup of 16 lanes; it
launches at most 1 lane (the CP scales EXEC by `min(grid_size,
workgroup_size)` — and `min(1, 16) = 1`).

**Symptom:** kernel runs without faulting, completion signal fires,
but the kernel produces no output. With `block_x = 1` (every Stage-2a
KFD launcher) the bug is invisible because `1 × 1 == 1`. The
Stage-13 launcher (`block_x = 16` and `block_x = 256`) was the first
to hit it — every dispatch was effectively single-lane and `d_sc`
came back all zeros.

**Fix:** in `kfd_dispatch`, take HIP-style parameters (`grid_x =
num_workgroups`) and multiply for the AQL field:

```mlr
uint32 gx = grid_x * block_x; unsafe { *((packet + 0x0C) as uint32) = gx }
uint32 gy = grid_y * block_y; unsafe { *((packet + 0x10) as uint32) = gy }
uint32 gz = grid_z * block_z; unsafe { *((packet + 0x14) as uint32) = gz }
```

**Source:** tinygrad's `_submit_aql` passes `gl.x*l.x` for the same
reason. HSA-Runtime's `aql_queue.cpp` does the same multiplication
inside `enqueue_kernel`.

**Where:** `kfd_dispatch` in `std/kfd.mlr`.

---

## 10. MLRift Stage-13 kernel ABI: `w_base` is slot, `row_base` is entry

The two CSR-aware Stage-13 kernels disagree on the unit of their
per-step "base" kernarg:

| Kernel              | Address formula              | Encoding                | Unit        |
|---------------------|------------------------------|-------------------------|-------------|
| `mt_csr_e_delayed`  | `bd + (row_base + src) * 8`  | `s_lshl_b64 ..., 3`     | entry index |
| `mt_lif_full`       | `bd + (w_base * N + tid) * 8`| `s_lshl_b64 ..., 7` (= ×N×8 for N=16) | slot index |

Pass each kernel its native unit:
```mlr
unsafe { *((kp_csre_h + 0x30) as uint64) = read_slot * N }   // entry idx
unsafe { *((kp_lif_h  + 0xA8) as uint64) = write_slot }      // slot idx
```

**Symptom of getting it wrong:** under HIP nothing visible — pooled
allocations silently absorb the resulting out-of-bounds writes from
LIF (only step 0 lands in `bd` since `0 * N == 0` makes both formulas
coincide). Under raw KFD the per-buffer page reservation is strict;
step 1's LIF dispatch page-faults on a TCP-write to one page past the
end of `d_bd`.

**Source of confusion:** the b4 launcher (`b4_native_stage13_load.mlr`)
passed `write_slot * N` to both kernels and was "accidentally GREEN"
under HIP for the reason above. The bug went undetected until the
Stage-2b KFD port. The b4 launcher has since been fixed and the new
KFD-native launcher (`b5b_kfd_stage13_load.mlr`) was written with the
correct units from the start.

**Where:** kernel emitters in `src/format_amdgpu.mlr`
(`emit_amdgpu_mt_lif_full_blob` and `emit_amdgpu_mt_csr_e_delayed_blob`);
launchers in `examples/b4_native_stage13_load.mlr` and
`examples/b5b_kfd_stage13_load.mlr`.

---

## 11. `power_dpm_force_performance_level=profile_peak` deadlocks single-queue dispatch

While probing whether the sysfs DPM knob could close the sync-launch
latency gap to HIP runtime (it can't — see `docs/AMDGPU_NATIVE.md`),
we found a specific firmware-side quirk: with the GPU pinned to
`profile_peak` AND only one active KFD compute queue (i.e. the user
queue, with our in-process boost queues disabled via `MLRIFT_BOOST=0`),
`hipDeviceSynchronize` times out:

```
hipDeviceSynchronize: timeout draining queue errno=30000000001
```

The user-queue AQL packet is dispatched cleanly (we see wp advance,
doorbell ring, packet header valid), but the GPU never retires it —
the completion signal stays at its in-flight value forever.

**Workaround:** keep at least one other compute or SDMA queue active
in the same process. Our default `MLRIFT_BOOST=1` path satisfies this
trivially (the dedicated boost AQL queue + SDMA boost queue stay
non-empty), so the deadlock only surfaces if you explicitly disable
the boost queues *and* pin DPM at `profile_peak`. Both `auto` and
`high` modes work fine with a single queue — only `profile_peak`
exposes this.

**Probable cause:** `profile_peak` puts the firmware's CP scheduler
into a mode where it expects multi-queue arbitration to drive
dispatch progress. With only a single queue's wp updates, the
scheduler appears to wait for a second queue's doorbell before
advancing rp. We have not root-caused this in firmware sources.

**Where:** `_hipkfd_lazy_init` in `std/hip_kfd.mlr` always brings up
the boost queues, so the deadlock is unreachable from any default
path. Documented here only because the script
`scripts/mlrift-gpu-perf-mode.sh` makes it easy to set
`profile_peak`, and someone bisecting the boost queues against the
DPM knob would hit it.

## Per-call `alloc(8)` ioctl scratch slots leak whole 4 KiB pages

**Symptom:** under any workload that calls `dev_alloc_*` /
`dev_free` / `dev_host_ptr` / any `raw_kfd_*` wrapper at high
frequency (Noesis NEAT evolution: ~80 such calls per genome × 3
evals/sec), host RSS grows linearly at ~10 MB/s with no obvious
source in the launcher's own `alloc(...)` calls. At 3 minutes the
process is ~2 GB heavy. `dealloc()` works correctly — primitive
verified by a 1 M-iter `alloc(16384) + write + dealloc` round-trip
with flat RSS. Smaps shows thousands of leaked `rw-p` anonymous
mappings of small sizes (12, 16, 32, 48 bytes — all rounded to 4 KiB
pages by the kernel).

**Cause:** every MLRift `alloc(N)` is a `mmap(NULL, N+8, PROT_RW,
MAP_PRIVATE|MAP_ANON)` syscall. The kernel rounds `N+8` up to a page
(4 KiB minimum). An `alloc(8)` for a single ioctl-outparam slot
consumes a full 4 KiB of RSS. When such a slot leaks (no matching
`dealloc()`), we lose 4 KiB per call, not 16 bytes.

The pattern was historically present at every site that needed an
ioctl outparam:

```mlrift
uint64 h_slot = alloc(8); uint64 va_slot = alloc(8); uint64 mm_slot = alloc(8)
uint32 r = raw_kfd_alloc_memory_of_gpu(_kfd_fd, _kfd_gpu_id, n, va, flags,
                                        h_slot, va_slot, mm_slot)
```

That's 12 KiB of RSS leaked per `dev_alloc_*` call. Across the
shim's 17+ ioctl wrappers (`dev_try_alloc_vram`,
`dev_try_alloc_host_visible`, `dev_alloc_qdesc`,
`dev_alloc_kernel_image`, `dev_alloc_aql_ring`,
`dev_alloc_vram_exec`, `dev_register_userptr`, `dev_alloc_is_vram`,
`dev_host_ptr`, `dev_free`, plus every `raw_kfd_*` wrapper in
`std/kfd_raw.mlr`), this dominates residual host-RSS growth once
obvious launcher-side leaks are fixed.

**Fix:** persistent `static uint64[N]` scratch buffers shared across
all ioctl wrappers. Two cover `std/kfd.mlr`'s needs: one for the
ALLOC family (`h_slot`, `va_slot`, `mm_slot`) and one for the FREE
/ lookup family (`h_slot`, `host_slot`). Every `alloc(8)` becomes
`_kfd_scratch_alloc + 0/8/16` (offsets into the shared buffer).
Single-threaded invariant holds: none of these wrappers is
recursively re-entered, and the shim is single-threaded with respect
to KFD ioctls.

Rolled out across:

- `std/kfd.mlr` — `_kfd_scratch_alloc[8]` and `_kfd_scratch_free[8]`
  cover all `dev_alloc_*` / `dev_free` / `dev_host_ptr`.
- `std/kfd_raw.mlr` — `_kfd_raw_scratch[16]` (128 B) covers all 17
  bare ioctl wrappers (`raw_kfd_alloc_memory_of_gpu`,
  `raw_kfd_free_memory_of_gpu`, `raw_kfd_map_memory_to_gpu`,
  `raw_kfd_unmap_memory_from_gpu`, `raw_kfd_wait_events`, etc.).
  The 128-byte slab is partitioned: bytes 0..95 for the primary args
  blob (largest is CREATE_QUEUE at ~88 B); bytes 96..127 for a
  secondary `dev_ids` slot used by MAP/UNMAP that needs two
  simultaneously-live buffers.

End-to-end: Noesis Phase Y NEAT evolution went from **1928 MB at
3 min** (10.6 MB/s leak) to **5.4 MB at 3 min** (~99.7% reduction,
flat from t=1s onward) without any other change to the workload.

**Where to look first when adding a new wrapper:** any new `dev_*`
or `raw_kfd_*` function that needs ioctl-outparam slots MUST use the
persistent scratch, not a fresh `alloc(8)`. When auditing leaks, the
smaps signature is unmistakable: thousands of `rw-p` anonymous
mappings of 4-12 KiB each (one or a few 4 KiB pages), all with
`Anonymous: == Size` (fully committed). The diagnostic recipe is in
`/tmp/_smaps_diff.sh` (size-class delta between two snapshots) plus
`strace -e mmap,munmap -c` to attribute alloc/free counts by size.
DPM knob would hit it.
