# `--target=amdgpu-native` — ROCm-free GPU launchers via raw AMDKFD

## What it does

`--target=amdgpu-native` makes MLRift compile a launcher that dispatches
GPU kernels through **raw AMDKFD ioctls** instead of HIP. The launcher
source itself is unchanged: same `import "../std/hip.mlr"`, same
`hipMalloc` / `hipMemcpy` / `hipModuleLoadData` / `hipModuleGetFunction`
/ `hipModuleLaunchKernel` / `hipDeviceSynchronize` / `hipFree` calls.
Under this flag the binary's `ldd` shows only `libc + ld-linux + vdso`
— no `libamdhip64`, `libhsa-runtime64`, `libdrm`, `libdrm_amdgpu`.

Default behavior (no flag) is unchanged: the launcher links the ROCm
HIP runtime as before.

```
# Default — uses HIP, links libamdhip64.
./build/mlrc --arch=x86_64 examples/m1o_atomic_load.mlr -o /tmp/m1o_hip
ldd /tmp/m1o_hip | grep hip          # → libamdhip64.so.7

# amdgpu-native — uses KFD, no ROCm linkage.
./build/mlrc --target=amdgpu-native --arch=x86_64 \
    examples/m1o_atomic_load.mlr -o /tmp/m1o_kfd
ldd /tmp/m1o_kfd | grep -iE 'hip|hsa|amd|drm'   # → (empty)
/tmp/m1o_kfd                                     # → "M1o gate: GREEN"
```

## How it works

Three pieces:

### 1. Import redirect (`src/main.mlr`)

`import_resolve` and the stdlib lookup branch run every resolved path
through `_maybe_redirect_hip_to_kfd`. When `target_amdgpu_native != 0`
and the path ends in `hip.mlr` (and not already `hip_kfd.mlr`), the
suffix is rewritten in place. The launcher `import "../std/hip.mlr"`
ends up loading `std/hip_kfd.mlr` instead.

### 2. KFD-backed shim (`std/hip_kfd.mlr`)

The shim provides concrete MLR `fn` implementations for the HIP API
surface used by Stage-1 launchers. They delegate to `std/kfd.mlr`
primitives:

| HIP call                      | KFD shim implementation                     |
|-------------------------------|---------------------------------------------|
| `hipSetDevice(0)`             | no-op (returns 0)                           |
| `hipGetDeviceCount(*ct)`      | writes 1                                    |
| `hipMalloc(*slot, n)`         | `dev_alloc_smart_hostvis(n)` (VRAM-host-vis → GTT fallback) — kept for back-compat; silently spills to GTT under BAR1 pressure |
| `hipMallocVramOnly(*slot, n)` | `dev_try_alloc_vram(n)` — VRAM only (kind=0, no host pointer); fails hard if VRAM exhausted.  Use for buffers the CPU never touches after init |
| `hipMallocAuto(*slot, n)`     | `dev_alloc_smart(n)` — VRAM-first, transparent system-RAM (GTT) spill on exhaustion.  Discriminate via `dev_alloc_kind(va)`: 0=VRAM, 2=GTT |
| `hipFree(p)`                  | `dev_free(p)`                               |
| `hipMemcpy(d, s, n, H2D=1)`   | `dev_copy_h2d` (requires host pointer; not safe on `hipMallocVramOnly` allocations) |
| `hipMemcpyHostToVram(d, s, n)`| SDMA `COPY_LINEAR` via shared GTT staging slab; works on any KFD-mapped destination including VRAM-only |
| `hipMemcpy(d, s, n, D2H=2)`   | `dev_copy_d2h` (host-view, fast for VRAM-host-vis) |
| `hipMemcpyVramToHost(d, s, n)`| SDMA `COPY_LINEAR` reverse + memcpy; works on any KFD-mapped source incl. VRAM-only |
| `hipMemcpy(d, s, n, D2D=3)`   | host-view-to-host-view memcpy               |
| `hipMemcpy(d, s, n, H2H/4)`   | plain `memcpy`                              |
| `hipModuleLoadData(*m, blob)` | `kernel_load_from_blob(blob, 0)` → `*m`     |
| `hipModuleGetFunction(*f, m, name)` | wrapper `{kd_va, kernarg_size}` → `*f`|
| `hipModuleLaunchKernel(...)`  | unmarshal ptr_table → kernarg blob, dispatch + busy-poll wait |
| `hipDeviceSynchronize()`      | spins on the shared signal counter; `hipModuleLaunchKernel` is async so this drains the queue |

A single AQL queue + signal + 4 KiB scratch kernarg buffer are
lazy-initialised on first call and reused across every launch.

`hipModuleLaunchKernel` is the only non-trivial translation. HIP's
`kernel_params` is `void**` — an array of pointers, each pointing to
an 8-byte slot holding one argument value. The shim:

1. Reads `kernarg_size` from the AMDHSA kernel descriptor at offset
   `0x08` (set by `amd_emit_kernel_descriptor` in
   `src/format_amdgpu.mlr`). `kernel_kernarg_size` in `std/kfd.mlr`
   does the lookup.
2. Iterates `kernarg_size / 8` argument slots, dereferences each
   ptr_table entry, and concatenates the values into a contiguous KFD
   kernarg blob — the layout the GPU's `s_load_dword` instructions
   actually expect.
3. Calls `kfd_dispatch` and `kfd_wait` synchronously.

Step 2 assumes every kernarg is 8 bytes. Every MLRift `@kernel` emitter
satisfies this (`args_desc` entries declare `size=8` for both
`GLOBAL_BUFFER` pointer args and `BY_VALUE` `uint64` / `f64` args).

### 3. Duplicate `@dynamic extern` deduplication (`src/analysis.mlr`)

Stage-1 HIP launchers locally declare `@dynamic extern fn puts(...)`
and `@dynamic extern fn exit(...)`. `std/kfd.mlr` (transitively imported
through the shim) declares the same. With strict redefinition checking
this collides. The semantic analyser now silently dedupes when **both**
declarations have flag bits 3 + 5 set (`extern` + `@dynamic`) — they
name the same `dlopen` symbol, so dropping one is correct.

## Supported launchers

Verified GREEN under `--target=amdgpu-native` with zero ROCm linkage:

| Launcher                                | Kernel emit flag                          |
|-----------------------------------------|-------------------------------------------|
| `m1o_atomic_load.mlr`                   | `--emit-amdgpu-atomic-demo=`              |
| `m1p_f64atomic_load.mlr`                | `--emit-amdgpu-f64-atomic-add=`           |
| `m1q_csr_scatter_load.mlr`              | `--emit-amdgpu-csr-scatter-demo=`         |
| `m1r_csr_i_deliver_load.mlr`            | `--emit-amdgpu-csr-i-deliver=`            |
| `m1s_csr_e_deliver_load.mlr`            | `--emit-amdgpu-csr-e-deliver=`            |
| `m1t_lif_integrate_load.mlr`            | `--emit-amdgpu-lif-integrate=`            |
| `m1t2_lif_spike_load.mlr`               | `--emit-amdgpu-lif-spike=`                |
| `m1t3_lif_stp_load.mlr`                 | `--emit-amdgpu-lif-stp=`                  |
| `m1t4_lif_full_load.mlr` / `b1_lif_full_loop.mlr` | `--emit-amdgpu-lif-full=`       |
| `m1u_mt_lif_integrate_load.mlr`         | `--emit-amdgpu-mt-lif-integrate=`         |
| `m1u2_mt_lif_full_load.mlr`             | `--emit-amdgpu-mt-lif-full=`              |
| `m1v_mt_csr_delivery_load.mlr`          | `--emit-amdgpu-mt-csr-i-deliver=`, `--emit-amdgpu-mt-csr-e-delayed=` |
| `b4_decay_relax_fill_load.mlr`          | `--emit-amdgpu-decay-step=`, `--emit-amdgpu-relax-step=`, `--emit-amdgpu-fill-step=` |
| `b4_native_stage13_load.mlr`            | all six Stage-13 emit flags               |

Standard build cycle:

```
./build/mlrc --emit-amdgpu-<flag>=/tmp/<name>.co
./build/mlrc --target=amdgpu-native --arch=x86_64 \
    examples/<launcher>.mlr -o /tmp/<launcher>
/tmp/<launcher>
```

`--target=amdgpu-native` on a launcher source with no `@kernel`
functions does NOT error any more — for these the redirect at import
time is what makes the flag mean something; no `.co` is emitted from
the launcher itself (the user is expected to emit kernels separately
via `--emit-amdgpu-<flag>=`).

## Semantics differences

* **`hipModuleLaunchKernel` is synchronous** under the shim. HIP
  proper is async + you sync via `hipDeviceSynchronize`; here the
  busy-poll wait runs inside the launch call and `hipDeviceSynchronize`
  is a no-op. This matches the b5b "single queue, wait between every
  dispatch" pattern. Launchers that rely on enqueueing many launches
  before syncing will run at HIP-async cadence under HIP and at
  serialised cadence under amdgpu-native — same final result, slower.

* **Graph API not implemented** in the shim. `hipGraphCreate`,
  `hipGraphLaunch`, etc. would need a graph→AQL-batch translator.
  Three launchers use it — `gpu_graph_smoke.mlr`,
  `gpu_stage13_graph.mlr`, and the noesis 60M graph variants — they
  fall through to undefined-symbol errors at compile time. The non-graph
  Stage-13 launcher (`b4_native_stage13_load.mlr`) works.

* **Buffer alignment is strict**. HIP allocates from pooled memory and
  silently absorbs out-of-bounds writes within the pool. KFD's
  `dev_alloc_host_visible` uses one mmap per allocation, so OOB writes
  page-fault. This caught a real bug in
  `b4_native_stage13_load.mlr` — see "Stage-13 kernel ABI gotcha"
  below.

## Gotchas

### #9: AQL `grid_size_*` is total work items, not num_workgroups

HIP/CUDA `gridDim` means num_workgroups; HSA AQL `grid_size` means
num_workgroups × workgroup_size. With `grid=1, block=1` the bug is
invisible (1×1 == 1), but with `block_x = N` the CP launches at most 1
lane per dispatch and the kernel does nothing useful (no fault, just
zero output). `kfd_dispatch` in `std/kfd.mlr` takes HIP-style params
(`grid = num_workgroups`) and multiplies internally for the AQL field —
the shim inherits this so HIP launchers work unchanged.

### #10: MLRift Stage-13 kernel ABI — `w_base` is slot, `row_base` is entry

`mt_lif_full` does `bd + (w_base * N + tid) * 8` (`s_lshl_b64` by 7
for `N=16`), so `w_base` must be a slot index `0..BUF_SIZE-1`.
`mt_csr_e_delayed` does `bd + (row_base + src) * 8` (`s_lshl_b64` by
3), so `row_base` must be `read_slot * N` (an entry index).

The original `b4_native_stage13_load.mlr` passed `write_slot * N` to
**both** kernels and was "accidentally GREEN" under HIP because:
- step 0's writes happened to be correct (`w_base = 0` makes both
  formulas coincide), and
- HIP's pooled allocations silently absorbed the out-of-bounds writes
  from steps 1..3.

KFD's per-buffer page reservations don't, so under
`--target=amdgpu-native` the launcher page-faulted on step 1's LIF
dispatch. The current `b4_native_stage13_load.mlr` passes
`write_slot` (slot index) to LIF — works under both HIP and KFD.
`b5b_kfd_stage13_load.mlr` documents the same fix.

For the full set of KFD-runtime gotchas (queue MQD layout, doorbell
off-by-one, EOP / ctx-save sizes, etc.) see `docs/KFD_GOTCHAS.md`.

## VRAM allocators and the GTT fallback

`std/kfd.mlr` exposes a small allocator family. The "smart" entry point
is the GPU analogue of CUDA's system-memory fallback: try VRAM first,
spill to host-mapped GTT if the card is short.

| Function                              | Backing | Failure  | Use                                  |
|---------------------------------------|---------|----------|--------------------------------------|
| `dev_alloc_vram(n)`                   | VRAM    | `kfd_die`| Mandatory VRAM allocation            |
| `dev_try_alloc_vram(n)`               | VRAM    | returns 0| Probe / caller handles fallback      |
| `dev_alloc_host_visible(n)`           | GTT     | `kfd_die`| Always-mappable, slower (PCIe)       |
| `dev_try_alloc_host_visible(n)`       | GTT     | returns 0| Probe                                |
| `dev_alloc_smart(n)`                  | VRAM→GTT| returns 0| Default for "fits if possible"       |
| `dev_alloc_is_vram(va) -> 0/1`        |   —     |    —     | Inspect what backing won out         |

`dev_alloc_smart` is the recommended default for ML workloads: weights
that fit in VRAM stay in VRAM (HBM bandwidth), oversized models spill
to system RAM (PCIe bandwidth, but always works as long as RAM does).
Both branches return the same kind of device VA — the GPU sees a
contiguous mapping in either case.

### Stress-test results (gfx1100, 16 GiB card)

`examples/b5d_kfd_vram_stress.mlr` measured the limits on this dev box:

* **Maximum single VRAM allocation: 16,176 MiB ≈ 15.8 GiB** — about 98%
  of total VRAM. The KFD ioctl path imposes a small overhead beyond
  the kernel's own bookkeeping but does not artificially cap chunk
  size.
* **Smart-fallback verified end to end**: held 15 separate 1 GiB VRAM
  blocks, then `dev_alloc_smart(1 GiB)` successfully spilled to GTT;
  the spilled buffer's host pointer round-tripped a marker write.

Run it yourself:
```
./build/mlrc --arch=x86_64 examples/b5d_kfd_vram_stress.mlr -o /tmp/b5d_stress
/tmp/b5d_stress
```

The stress test is deterministic on a quiet GPU; if other processes
hold VRAM the cap it reports drops accordingly.

## File map

* `std/hip_kfd.mlr` — the HIP-API shim (~150 lines).
* `std/kfd.mlr` — KFD library: `kfd_init`, `dev_alloc_*` family
  (incl. `dev_alloc_smart` / `dev_try_alloc_*` / `dev_alloc_is_vram`),
  `kernel_load` / `kernel_load_from_blob` / `kernel_kernarg_size`,
  `kfd_create_queue`, `kfd_dispatch`, `kfd_wait`,
  `kfd_signal_alloc`/`reset`/`value`, `dev_copy_h2d` / `dev_copy_d2h`
  / `dev_free`, `dev_host_ptr`.
* `std/kfd_raw.mlr` — bare AMDKFD ioctl wrappers.
* `src/main.mlr` — `_maybe_redirect_hip_to_kfd`, the
  `--target=amdgpu-native` flag handler.
* `src/analysis.mlr` — duplicate `@dynamic extern` dedup.
* `examples/b5a_*.mlr`, `examples/b5b_*.mlr`, `examples/b5c_*.mlr`,
  `examples/b5d_kfd_vram_stress.mlr` — KFD-native launchers (import
  `std/kfd.mlr` directly). Useful as references for the KFD library
  API. The b5c set ports every Stage-1 milestone (M1p..M1v, B1, B4a)
  to KFD-native form alongside the shim path.

## Inventory of HIP-using example sources

After the shim landed, every existing HIP-API launcher works under
`--target=amdgpu-native` with no source change. The `examples/`
directory still ships these so you can compare HIP-direct vs
KFD-via-shim on the same code.

* **Tested under the shim and ldd-clean**: `m1o_atomic_load`, `m1p_*`,
  `m1q_*`, `m1r_*`, `m1s_*`, `m1t_*`, `m1t2_*`, `m1t3_*`, `m1t4_*`,
  `m1u_*`, `m1u2_*`, `m1v_*`, `b1_lif_full_loop`,
  `b4_decay_relax_fill_load`, `b4_native_stage13_load`. (Note: b4
  carried a real bug — `write_slot * N` to LIF's `w_base` instead of
  `write_slot` — that was "accidentally GREEN" under HIP and now
  page-faults under strict KFD mappings; the fix is in tree.)
* **Stay HIP-only** (use APIs the shim doesn't implement, mostly
  `hipGraph*`): `gpu_graph_smoke`, `gpu_stage13_graph`,
  `n1_launch{,_direct}`, `noesis_*` (graph variants),
  `qwen3_*` (long-running launchers that benefit from `KFD_WAIT_EVENTS`,
  not yet implemented). Compile these with the default path
  (`./build/mlrc examples/...`) — they link `libamdhip64`.
* **Stay HIP-only — kernel sources for `--target=hip-amd`**:
  `gpu_atomic.mlr`, `gpu_csr.mlr`, `gpu_csr_i.mlr`, `gpu_decay.mlr`,
  `gpu_lif.mlr`, `gpu_ring.mlr`, `gpu_stage13.mlr`. These ship
  `@kernel` definitions to be compiled by `hipcc`. They predate the
  native AMDGPU emitter and rely on HIP-runtime atomics / shapes the
  emitter does not yet recognise. Their paired launchers
  (`gpu_*_launch.mlr`) link `libamdhip64`.
* **Removed**: `m1a_load_nop.mlr`, `m1b_load_sentinel.mlr` — trivial
  bring-up tests (NOP kernel + sentinel writer) superseded by
  `b5a_kfd_init_smoke` + `b5a_kfd_m1o_load`. Their kernel emit flags
  (`--emit-amdgpu-nop=`, `--emit-amdgpu-sentinel=`) still exist in
  the compiler for any future regression coverage.

## Testing

```
make build && make bootstrap        # rebuild compiler + verify fixed point
make test                            # 439/439

# Per-launcher gate (under shim):
./build/mlrc --emit-amdgpu-atomic-demo=/tmp/m1o.co
./build/mlrc --target=amdgpu-native --arch=x86_64 \
    examples/m1o_atomic_load.mlr -o /tmp/m1o
/tmp/m1o
ldd /tmp/m1o | grep -iE 'hsa|hip|amd|drm'   # → (empty)
```

## Sync-launch latency vs HIP runtime

Real workloads with sustained dispatch (noesis_60m, batched gemv, LLM
decode) hit HIP-runtime parity out of the box — the in-process boost
queues in `std/hip_kfd.mlr` (a compute boost on a dedicated AQL ring,
plus an SDMA boost streaming VRAM↔VRAM copies) keep both sclk and the
memory subsystem warm while the user queue stays non-empty.

Numbers on the gfx1100 reference (RX 7800 XT, gemv f32 M=K=1024):

| pattern             | shim, default | HIP runtime |
|---------------------|--------------:|------------:|
| sync-launch         |       ~850 µs |    ~410 µs  |
| batched (≥8 deep)   |       ~100 µs |     ~70 µs  |
| noesis_60m, 2000 st |        29.0 s |     28.9 s  |

The shim's residual 2× on sync-launch is **not** closeable from
userspace alone — see the next section for what we tried.

### Why the sysfs DPM knob doesn't help (and the script is mostly diagnostic)

`scripts/mlrift-gpu-perf-mode.sh` writes
`/sys/class/drm/cardN/device/power_dpm_force_performance_level`. We
expected `high` (or `profile_peak`) to pin both sclk and mclk at peak,
removing the post-sync ramp-up cost. Measured outcome on RDNA3 / RX
7800 XT:

| mode                 | mclk pinned? | sclk pinned? | sync-launch |
|----------------------|:------------:|:------------:|------------:|
| `auto` (default)     |   no         |   no         |    ~850 µs  |
| `high`               |  **yes**     |   no         |    ~940 µs  |
| `profile_peak`       |  **yes**     |   no         |    ~830 µs  |

Both `high` and `profile_peak` cap mclk at state 3 (1218 MHz) and
fclk at state 7 (2301 MHz) as expected, but `pp_dpm_sclk` stays
reading `1: 0Mhz *` — sclk DPM stays gated unless there's active
shader work, regardless of the force level. HIP runtime achieves
~157 µs under `profile_peak`, so it must pin sclk through a
non-sysfs path (likely an internal libhsa-runtime ioctl or firmware
hint we haven't yet identified).

Worse: `profile_peak` + `MLRIFT_BOOST=0` (i.e. user disables our
boost queues) **deadlocks** with `hipDeviceSynchronize: timeout
draining queue`. Under pinned mclk, the user-queue dispatch never
retires unless a second queue is also active — a firmware-side
quirk we haven't root-caused.

The script ships anyway as a diagnostic tool — useful for
reproducing the table above, or as a starting point if you find the
sclk-pin path HIP uses. **For normal use, leave the GPU in `auto`
and rely on the in-process boost queues.**

```
sudo scripts/mlrift-gpu-perf-mode.sh profile_peak   # diagnostic
sudo scripts/mlrift-gpu-perf-mode.sh auto           # restore default (do this!)
```
