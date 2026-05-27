# MLRift GPU — path to zero dependencies

## Stage-1 progress (live)

Milestones landed on the native AMDGPU emitter path (no hipcc,
no clang; the bytes written to `.co` come from
`src/format_amdgpu.mlr`'s encoder directly). Each is byte-
identical to clang's gfx1100 output for the equivalent HIP
source, verified at commit time with `cmp` against the hipcc-
produced reference and at runtime with a HIP-launch gate.

| Milestone | What it adds | Gate |
|-----------|--------------|------|
| M1a | Raw code-object ELF, `s_endpgm` no-op | hipModuleLoadData accepts |
| M1b | `.args` metadata + one 64-bit kernarg | Host-sent `0xDEADBEEF` echoes back |
| M1c | gfx1100 encoder (SOPP, SMEM, VOPD, FLAT) | Sentinel kernel built structurally |
| M1d | First AST→AMDGPU: `@kernel` → `.co` | `--target=amdgpu-native` wired |
| M1e | Two-kernarg dispatch (ptr + u32 by-value) | Host forwards value |
| M1f | Multi-kernel `.co` (N kernels per emit) + `tid_x()` | 2 kernels / 1 module; 64-thread per-lane writes |
| M1g | Bounds guard (`if tid_x() < n`) via `s_and_saveexec` + `s_cbranch_execz` | 64 threads / 40 slots, no stray writes |
| M1h | f64 arithmetic: `v_fma_f64`, `global_load_b64`, `global_store_b64`, natural-alignment kernarg offsets | M3 decay single-block: max ULP = 0 vs CPU ref |

Not yet done in Stage 1:

- **Grid-wide thread ID** (`global_tid = blockIdx.x * blockDim.x
  + threadIdx.x`). Gates `--target=amdgpu-native` retiring hipcc
  from `examples/gpu_decay.mlr` and everything downstream of it
  (the stage-13 examples launch multi-block). Clang's reference
  uses `v_mad_u64_u32` + a dispatch-ptr load for `blockDim.x` +
  `v_lshlrev_b64` / `v_add_co_u32` for 64-bit address compute,
  plus hidden kernarg slots — each of those is a new encoder.
  Target for M1j.
- **VOP3P / richer arithmetic**. Stage-13 needs atomic_add_f64,
  conditional-select, STP/STDP updates — more ops but the
  encoder skeleton from M1h covers the format.
- **Multi-kernel sharing of a single dispatch-ptr header**. When
  many small kernels run back-to-back, the native path can
  arrange for one dispatch ptr read to serve many kernels.
  Not on the critical path.

For full single-block workloads M1h is already the replacement
for the hipcc build step: `examples/m1h_decay_kernel.mlr`
through `--target=amdgpu-native` produces the same bytes hipcc
would. `examples/gpu_decay.mlr`'s multi-block launcher path
stays on `--target=hip-amd` until M1j lands.

---

Status (original kickoff): phase-1 backend (M0-M4.h) is complete. The self-hosted
MLRift compiler emits `@kernel` functions as HIP C++ source,
shells out to `hipcc` to produce a code object, then at runtime
dynamically links against `libamdhip64.so.7` to load + launch
kernels. This works end-to-end on an RX 7800 XT (stage-13,
200 000 neurons × 5 000 steps, 614 ms wall) but depends on three
external artifacts:

1. **`hipcc`** — ROCm's HIP compiler driver (`/opt/rocm/bin/hipcc`).
   Invoked at MLRift build time. Uses LLVM/clang under the hood
   with `--offload-arch=gfx1100`. Produces a clang offload
   bundle — *not* a bare ELF.
2. **`libamdhip64.so.7`** — the HIP runtime userspace library.
   Linked at load time via `DT_NEEDED` on the output ELF. Provides
   `hipSetDevice`, `hipMalloc`, `hipModuleLaunchKernel`,
   `hipGraph*`, etc.
3. **`ld-linux-x86-64.so.2`** — the glibc dynamic loader.
   Required because our ELF has `PT_INTERP` and needs runtime
   relocation / `__libc_start_main` bootstrap.

The goal over time is to erase all three, so an MLRift binary
that uses the GPU is a single statically-linked ELF that talks
to the kernel directly. Two stages:

## Stage 1 — drop `hipcc`

**Scope.** MLRift emits a ready-to-load AMD code-object ELF
directly from the IR. Still uses `libamdhip64` at runtime (via
`hipModuleLoadData` on the emitted blob). No `hipcc` shell-out.
No clang / LLVM dependency. No ROCm-devel install required on
build machines — only ROCm runtime on run machines.

**What we need to build.**

1. **AMD GCN / RDNA3 assembler.** An in-tree assembler that
   turns AMDGPU ISA text (or equivalently, a textual IR we pick)
   into the packed 32/64-bit encodings documented in AMD's "RDNA3
   Instruction Set Architecture Reference Guide". For gfx1100
   that's VOP1/VOP2/VOP3/VOPC/VOPD, SOP1/SOP2/SOPP, SMEM, VMEM
   (FLAT/MUBUF/DS/BUFFER), and the EXEC/SCC/VCC/SGPR/VGPR
   machinery. A trimmed subset is enough for our kernels: int +
   fp arithmetic, atomic adds, global loads/stores, thread-id
   intrinsics, bounds-checked branches. The existing `format_hip`
   body translator already factors bodies down to these
   primitives — we translate to ISA instead of HIP C++.
2. **GPU IR lowering.** Today `format_hip.mlr` takes an
   `@kernel` AST + calls `hipcc`. Stage 1 adds an IR pass that
   lowers the kernel IR to AMDGPU instructions directly, runs
   register allocation over the VGPR/SGPR files (separate from
   our x86 allocator), and emits textual ISA or packed bytes
   into a `.text` section buffer.
3. **Code-object ELF emitter.** A new format module that wraps
   the kernel text + metadata into the AMD ROCm code-object v5
   container (`EI_OSABI = 64` AMDGPU_HSA, `e_machine = EM_AMDGPU`,
   specific `.note.hsa` entries, `.rodata` kernel descriptor blobs,
   a `.dynsym` / `.dynstr` pair for the exported kernel symbols,
   and `.rela.dyn` / `.hash` so the loader can find them). The
   format is well-specified; `/opt/rocm/include/hsa/amd_hsa_*.h`
   has the struct layouts. Our existing dynamic-ELF emitter in
   `src/format_elf_dyn.mlr` is not applicable — code objects are
   a different ELF flavour — but the byte-layout discipline
   carries over.
4. **Kernel descriptor + resource registers.** Each kernel gets
   a 64-byte kernel descriptor (`amd_kernel_code_t` in v3 or
   the newer flat-layout in v5) telling the hardware how much
   VGPR/SGPR, LDS, private memory, and scratch it uses, plus
   its entry PC. We compute these from our reg-alloc output.
5. **Validation bridge.** Keep `hipcc` output parked alongside
   ours in a test harness, so for every kernel we can
   byte-diff the emitted `.text` or, more realistically,
   spike-count-diff a run of the same kernel loaded from our
   blob vs `hipcc`'s. Spike counts should remain bit-exact
   under the existing determinism contract.

**Rough sizing.** AMD's v5 code-object layout is ~50 pages of
spec; RDNA3 ISA ref is ~600 pages but we only need 1–2 dozen
opcodes for stage-13. Bringing up assembler + ELF + descriptor
is roughly 15–25 focused sessions, one milestone per subsystem.
The first milestone is a "hello" kernel: write `d[tid] = tid * 2`
into device memory and verify D2H. Once that loads, the existing
translator can be redirected from HIP C++ to ISA incrementally,
one intrinsic at a time.

**What it buys.**

- Remove `hipcc` from MLRift's critical path. Build box only
  needs the MLRift source tree + a C compiler for `/lib/ld`.
- Faster kernel iteration. Current hipcc invocation is ~2–3 s;
  our emitter will be microseconds.
- Full control over the emitted kernel layout, VGPR budget, and
  wavefront size. Opens the door to MLRift-specific
  optimisations (e.g. cooperative deterministic atomics) that
  clang won't do.
- Prerequisite for Stage 2 — a self-emitted code object is
  something we can submit to the GPU without ever calling into
  `libamdhip64`'s loader.

**What it does NOT buy.**

- No wall-clock speedup on compute-bound workloads like our
  200k-neuron stage-13 run. Kernel launch path is still through
  `libamdhip64`.

## Stage 2 — drop `libamdhip64` (and probably `ld-linux-*` too)

**The user's real goal.** Talk to the GPU directly over the KFD
(Kernel Fusion Driver) character-device interface that ROCm's
userspace runtime sits on top of. MLRift's GPU binary becomes a
single static ELF that `open("/dev/kfd")`'s, creates a queue,
writes AQL packets into it, rings a doorbell, and polls / waits
for completion — no HIP symbols, no `dlopen`, no `DT_NEEDED`.

**What the KFD interface looks like.**

- `/dev/kfd` + `ioctl()` for topology, memory pinning, queue
  creation, signal objects, eviction/suspend control, and some
  debugger hooks. The public uapi header is
  `/usr/include/linux/kfd_ioctl.h` (upstream Linux).
- **Queues.** Each queue is a pair of ring buffers (submit +
  completion) mapped into your process via `mmap`. You push
  AQL (Architected Queuing Language) packets — 64-byte records
  — and hit a "doorbell" MMIO page also mapped via `mmap` that
  tells the hardware queue processor a new packet is ready.
- **AQL dispatch packet.** 64-byte struct with kernel object
  address, kernarg address, grid/workgroup size, header bits.
  Same fundamentals as hipModuleLaunchKernel but pushed as a
  memory record rather than a library call.
- **Memory.** `kfd_ioctl_alloc_memory_of_gpu` + `map_memory_to_gpu`
  give you VRAM / GTT buffers usable by the queue. MLRift host
  code mutates them by writing to the mapped host-visible
  pointer (for USWC/WC buffers) or uses `hipMemcpy`-equivalent
  SDMA queues for H2D/D2H.
- **Completion.** AMD uses `amd_signal_t` structs (64 bytes,
  cache-line aligned) plus optional kernel-mode `eventfd`
  handles. Polling the signal's `value` field from the host is
  the simplest model; interrupt-driven waits go through
  `kfd_ioctl_wait_events`.

**What we need to build.**

1. **KFD ioctl wrappers.** A `src/kfd.mlr` that mirrors
   `kfd_ioctl.h`: `AMDKFD_IOC_GET_VERSION`,
   `AMDKFD_IOC_CREATE_QUEUE`, `AMDKFD_IOC_ALLOC_MEMORY_OF_GPU`,
   `AMDKFD_IOC_MAP_MEMORY_TO_GPU`, etc. All via direct
   `syscall(ioctl, fd, nr, arg)` — no libc needed beyond the
   syscall we already emit.
2. **AQL packet writer.** Build the 64-byte dispatch packet in
   MLRift given (kernel descriptor addr, kernarg ptr, grid,
   block, shared mem, signal). Barrier-and / barrier-or packets
   for cross-queue ordering.
3. **Queue / doorbell setup.** At process start: open `/dev/kfd`,
   topology discovery (find the 7800 XT node id), allocate a
   device queue of N entries, map the doorbell page, set up
   `amd_signal` pool. All before `main()` or in a runtime init
   function emitted by the compiler.
4. **Memory manager.** VRAM allocator on top of
   `AMDKFD_IOC_ALLOC_MEMORY_OF_GPU`. Simple bump allocator to
   start; pool by size class later. H2D / D2H via the SDMA
   queue (a second KFD queue of kind `SDMA`) rather than
   synchronous blit, to match `hipMemcpy` semantics.
5. **Static ELF.** Drop `PT_INTERP`. The emitted binary starts
   at `_start`, sets up its own stack / TLS, runs `main`,
   calls `exit` via raw syscall. No glibc, no ld.so. The
   existing `src/format_elf.mlr` (non-dyn) already knows how to
   do this for CPU-only binaries; Stage 2 reuses it.
6. **AQL equivalents of `hipGraph*`.** The AQL spec has
   barrier packets and multiple in-flight packets per queue,
   which gives us graph-style batching natively — submit N
   packets, sync on a tail signal. No host-side overhead per
   launch beyond writing the packet bytes.
7. **Error / abort paths.** MMU faults and SQ traps raise
   interrupts the kernel surfaces via KFD events. Handling is
   a nice-to-have; for phase-1 of Stage 2, fatal aborts on any
   hardware trap are fine.

**Rough sizing.** Larger than Stage 1. The KFD uapi is stable
but thinly documented outside AMD's ROCT repo
([github.com/ROCm/ROCT-Thunk-Interface](https://github.com/ROCm/ROCT-Thunk-Interface))
and the ROCr-Runtime sources
([github.com/ROCm/ROCR-Runtime](https://github.com/ROCm/ROCR-Runtime)).
Expect 30–50 focused sessions, possibly more if the topology /
signal / doorbell paths have surprises. Can be staged:

- **Stage 2a.** Reimplement `hipSetDevice` + `hipMalloc` + `hipMemcpy`
  + a single-kernel launch, by-passing libamdhip64 for those
  calls only. Keep `hipModuleLoadData` (libamdhip64) for now.
  Validates: KFD queue setup, AQL writer, SDMA copy,
  amd_signal polling end-to-end. Smallest self-contained win.
- **Stage 2b.** Replace `hipModuleLoadData` + `hipModuleGetFunction`
  with our own code-object loader (easy once Stage 1 lands —
  we already know the ELF). libamdhip64 is now gone.
- **Stage 2c.** Drop `ld-linux-*`. Emit a fully-static ELF.
  Touches `src/format_elf.mlr`, `src/main.mlr` driver, and the
  `@dynamic extern` machinery (unused now). Confirms:
  `ldd binary` → `not a dynamic executable`.

**What Stage 2 buys.**

- Zero userspace dependencies. `scp` the ELF to any Linux box
  with an RDNA3 GPU + the open AMDGPU kernel driver, run it.
  Deterministic across machines.
- Faster kernel launch. Rough numbers from ROCm maintainers'
  benchmarks: libamdhip64's `hipModuleLaunchKernel` costs
  ~5–10 µs. Direct AQL doorbell write is ~100–300 ns. For our
  50 000 launch raw-variant that's 50 000 × ~7 µs = 350 ms of
  pure CPU overhead removed. Re-run on 5 000-launch graph
  variant ≈ 35 ms saved. Meaningful on latency-bound small-
  workload benchmarks; invisible on the 200k compute-bound
  one.
- Room to build MLRift-native GPU primitives: deterministic
  atomic chains, explicit wavefront-cooperative collectives,
  tight kernarg packing. None of these are possible through
  the public HIP API.
- Fulfils the project-wide vision of "no dependencies beyond
  the kernel and the instruction set." MLRift produces
  bare-metal ELFs for CPU already; this extends it to GPU.

**What Stage 2 does NOT buy on our current workloads.**

- The 200k stage-13 benchmark is GPU-compute-bound. Stage 2
  will move launch overhead from ~7 µs/call to ~300 ns/call
  but that's a fraction of a percent of total runtime. Real
  win comes from future MLRift-native primitives, not from
  stripping libamdhip64 per se.

## Order of operations

Stage 1 first. Concretely:

1. Hand-assemble and load a minimal gfx1100 code object from
   MLRift — just the ELF wrapper around a no-op kernel loaded
   via `hipModuleLoadData`. Prove the format.
2. Add an AMDGPU assembler for a handful of VOP2/VMEM opcodes
   sufficient for the decay kernel. Make the M3 decay kernel
   a round-trip: MLRift IR → our ISA bytes → code object →
   hipModuleLoadData → launch → bit-exact match with the
   hipcc-produced blob.
3. Incrementally cover stage-13's full kernel set (decay,
   relax, fill, csr_e_deliver_delayed, csr_i_deliver,
   lif_full). Delete `hipcc` from `src/main.mlr` build path.
4. Stage 2a: KFD wrappers + AQL writer, replace
   `hipModuleLaunchKernel` on the M3 decay test. Compare
   wall-time per launch against libamdhip64.
5. Stage 2b: replace `hipModuleLoadData` — by now our Stage 1
   code objects are self-loaded.
6. Stage 2c: statically link the test binary, confirm `ldd`
   reports no dependencies.

Each step keeps the prior path working, so regressions are
localized and the CI spike-count diffs keep catching drift.
