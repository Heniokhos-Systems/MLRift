# MLRift GPU backend — kickoff spec

Status: **historical kickoff document.** Stage 1 (HIP source emission)
and Stage 2 (raw KFD direct-dispatch) have both shipped. For the
current GPU paths see:

* `docs/AMDGPU_NATIVE.md` — `--target=amdgpu-native`, the
  KFD-backed runtime that produces ROCm-free launcher binaries.
* `docs/KFD_GOTCHAS.md` — 10 undocumented AMDKFD ioctl gotchas
  collected while bringing up the direct-dispatch path on gfx1100.
* `std/kfd.mlr`, `std/kfd_raw.mlr`, `std/hip_kfd.mlr` — the runtime
  library + HIP-API shim.
* `examples/b5a_*.mlr`, `examples/b5b_*.mlr`, `examples/b5c_*.mlr` —
  KFD-native launchers (the "b5c" set are direct ports of every
  Stage-1 HIP launcher to the KFD library API).

The kickoff spec below is preserved for context.

---

## Priority & scope

**First ship: AMD 7800XT via ROCm/HIP.** Do not write a single line
targeting NVIDIA until an AMD build of stage 13 or stage 15 runs
end-to-end on the 7800XT with results that match the CPU reference
within the documented tolerance.

**Phase 2: NVIDIA 5080 via HIP-on-CUDA.** Same `.hip` source
re-compiled with `hipcc --amdgpu-target=sm_120`-style flags. Expect
~95% of the CPU codegen logic to carry over unchanged. Phase 2 is
starts after phase 1 is stable; do not attempt both in parallel.

Rationale: AMD's 7800XT is the desktop primary target, it's the GPU
most of Noesis already runs on (ROCm), and HIP is AMD's native
interface so the AMD path is the reference. HIP on NVIDIA is a
transpilation layer — stable, but second-class in error messages,
debugger support, and profiling tooling. Debugging against a
transpilation target while you're still learning the IR→HIP pipeline
is masochism; debug against the reference target first.

## Why Option C (source emission)

Decided in the design discussion (see conversation history). For
spiking / sparse ML workloads, the performance-ceiling gap between
Option A (SPIR-V/Vulkan compute) and Option C (HIP source emission)
is 15-25%. But for future dense-ML work in MLRift the gap is 5-10×
because option C inherits vendor tensor-core libraries (cuBLAS,
rocBLAS). We pay an install-time toolchain dependency (hipcc at
build time) to keep that door open.

MLRift's zero-runtime-dependency property is preserved: the binary
`dlopen`s `libamdhip64.so` at runtime (like how our Windows output
goes through the PE IAT). Users install the HIP runtime once; the
MLRift binary itself ships as a self-contained ELF.

## Architecture

```
  ┌────────────┐     frontend (existing)     ┌──────────┐
  │  *.mlr     │  ─────────────────────────▶ │  KRIR    │
  └────────────┘                             └────┬─────┘
                                                  │
                        ┌─────────────────────────┼────────┐
                        │                         │        │
                  ┌─────▼─────┐          ┌────────▼───┐    │
                  │ ir.mlr     │          │ format_hip │    │
                  │ x86/arm   │          │ .mlr (new)  │    │
                  │ (CPU)     │          │            │    │
                  └─────┬─────┘          └────┬───────┘    │
                        │                     │            │
                  ┌─────▼──────┐         ┌────▼──────┐     │
                  │  ELF/PE    │         │  prog.hip │◀────┘
                  │  binary    │         │  (text)   │
                  └────────────┘         └────┬──────┘
                                              │
                                    external: hipcc
                                              │
                                    ┌─────────▼─────────┐
                                    │  prog.hipbin or   │
                                    │  libprog_gpu.so   │
                                    └─────────┬─────────┘
                                              │
                                         embedded as
                                         resource blob
                                              │
                                    ┌─────────▼─────────┐
                                    │  MLRift host bin  │
                                    │  + hip_runtime.mlr │
                                    │  (dlopen HIP)     │
                                    └───────────────────┘
```

Host code runs from a regular MLRift ELF. When it hits a GPU kernel
launch, it calls into `hip_runtime.mlr`, which has dlopen'd
`libamdhip64.so` at startup. The kernel binary (compiled from the
emitted `.hip` source at MLRift-build-time) is embedded in the host
binary as a data blob and loaded via `hipModuleLoadData`.

## New IR ops

Proposed additions to `src/ir.mlr` (opcode numbers tentative — pick
whatever slot is free):

| op              | description                                          |
|-----------------|------------------------------------------------------|
| IR_GPU_ALLOC    | device-side memory alloc — returns device pointer    |
| IR_GPU_FREE     | free device memory                                   |
| IR_GPU_H2D      | host → device memcpy                                 |
| IR_GPU_D2H      | device → host memcpy                                 |
| IR_GPU_D2D      | device → device memcpy                               |
| IR_KERNEL       | marks a function as a GPU kernel (frontend attr)     |
| IR_KERNEL_LAUNCH| dispatch a kernel with (gridX, gridY, gridZ, blockX, blockY, blockZ, arg_ptrs[]) |
| IR_GPU_SYNC     | hipDeviceSynchronize                                 |
| IR_GPU_BARRIER  | __syncthreads() inside a kernel                      |

Kernel body IR is a restricted subset: no heap alloc, no stdout,
no recursion. The `format_hip.mlr` emitter rejects any kernel that
tries to use unsupported ops.

## Runtime interface (`src/hip_runtime.mlr`)

At process start (in main's prologue or first GPU op), dlopen
`libamdhip64.so.6` (or versioned variant). Bind these functions by
name:

```
hipGetDeviceCount, hipSetDevice
hipMalloc, hipFree
hipMemcpy, hipMemcpyAsync
hipModuleLoadData, hipModuleGetFunction, hipModuleLaunchKernel
hipDeviceSynchronize, hipStreamCreate, hipStreamSynchronize
hipGetErrorString, hipGetLastError
hipEventCreate, hipEventRecord, hipEventElapsedTime  (for benchmarking)
```

Each binding is a function pointer loaded via `dlsym`. On missing
library or missing symbol, fail loudly with:

```
mlrc: GPU runtime not available
  looked for: libamdhip64.so.6 in LD_LIBRARY_PATH
  symbol missing: hipMalloc
  install ROCm ≥ 6.0 or run with --target=cpu
```

Host-side coordination code emitted by `format_hip.mlr` calls into
these bindings through normal MLRift extern-call machinery.

## Build pipeline

New `mlrc` flag: `--target=hip-amd`. When set:

1. Frontend parses `.mlr` source → IR (unchanged).
2. IR passes run (unchanged where possible).
3. `format_hip.mlr` walks IR and emits two artefacts:
   - `out.hip` — GPU kernel source
   - `out.hip.hostlib.c` — small C shim that calls into hip_runtime
     (simpler than emitting both sides ourselves)
4. MLRift forks/exec's `hipcc --genco out.hip -o out.co --offload-arch=gfx1100`
   (gfx1100 is 7800XT; gfx1030/1031/1032 for RDNA2, sm_XY for NVIDIA).
5. MLRift emits the host ELF, embedding `out.co` as a data blob via a
   `.gpu_module` section.
6. At runtime, host calls `hipModuleLoadData(blob_ptr)` to load the
   embedded kernel module.

Phase 1 implementation may skip step 5's embedding (write kernel blob
to a side file `out.co`, load via `hipModuleLoad(filename)`); move to
embedded blob in phase 1.5.

## Milestones

**M0 — hello-world (1-3 days after toolchain ready)**
  Write a fixed `.hip` file by hand that does `d_out[i] = i * 2` for
  N=1024 elements. Host code in C (not MLRift yet) that copies up,
  launches, copies down, prints. Run on 7800XT. Confirms ROCm stack
  is alive and you have a working kernel-launch recipe to copy.

**M1 — format_hip.mlr emits a trivial kernel (3-5 days)**
  Single IR pattern: an MLRift fn marked `@kernel` that operates on
  two device arrays. format_hip.mlr emits matching `.hip` source.
  MLRift invokes hipcc. Host side still hand-written C.

**M2 — hip_runtime.mlr in MLRift (5-7 days)**
  dlopen + dlsym binding for hipMalloc, hipFree, hipMemcpy,
  hipModuleLoad, hipModuleLaunchKernel, hipDeviceSynchronize.
  Port the M1 hello kernel's host side to MLRift. End-to-end MLRift
  → .mlr → .hip → .co → MLRift binary → run on GPU.

**M3 — stage-13 decay loop on GPU (1 week)**
  Port ONLY the per-neuron decay sweep (the dense `s_exc decay`,
  `s_inh decay`, etc.) to GPU. Everything else CPU. Host<->device
  transfers per step. Correctness: byte-identical to CPU version
  for this part is not achievable (GPU fp reductions differ), but
  spike-count statistics should match within 0.1%.

**M4 — stage-13 full pipeline on GPU (2 weeks)**
  All hot loops on device: decay, integrate, CSR delivery (with
  atomicAdd into s_exc), STDP weight updates, ring-buffer writes.
  Host only runs the spike-print and summary output. First real
  benchmark: 50k neurons, 5M synapses, 1000 steps.

**M5 — NVIDIA 5080 via HIP-on-CUDA (1 week)**
  Install CUDA toolkit + HIP-nvidia on laptop. Recompile stage-13
  kernels with `hipcc --offload-arch=sm_120`. Run same MLRift
  host binary with a different embedded .co blob. Measure ratio
  vs 7800XT.

Total: 4-6 weeks phase 1 (AMD), +1 week phase 2 (NVIDIA).

## Determinism contract

Vision doc specifies:
- **Bit-exact within backend**: same seed + same GPU + same MLRift
  binary → bit-identical output across runs. ACHIEVABLE by pinning
  launch config and using atomic ops in a fixed order.
- **Bit-exact across backends** (CPU vs GPU): opt-in flag with cost.
  NOT the GPU phase-1 goal. GPU phase-1 documents a tolerance:
  spike counts within 0.1%, mean weights within 1e-6 of CPU values.

Tests: for each ported stage, add a tolerance-diff script (not
byte-diff) that asserts the statistical equivalence. Only
bit-exact at the output boundary for non-reduction paths (spike
step indices, RNG sequences).

## Testing strategy

Mirror the CPU side's Python reference model: for each GPU example,
ship a CPU-MLRift reference binary and a GPU-MLRift benchmark binary.
Compare:
- spike-count vectors: exact match required (spikes are discrete)
- LTP/LTD event counts: within ±0.1% (atomic-add reordering)
- Final weight sums: within ±1e-5 × |value| (fp reduction order)
- Mean u/x STP traces: within ±1e-6

CI (cross-platform.yml) adds a `gpu-amd` job on a self-hosted runner
with a 7800XT once M3 lands. NVIDIA phase 2 adds a second runner.

## Hardware / toolchain prerequisites

**7800XT box (primary)**
- Linux with amdgpu kernel module (check `lspci | grep AMD`)
- ROCm ≥ 6.0: `sudo amdgpu-install --usecase=hip,rocm`
- Verify: `hipconfig --platform` → amd; `hipconfig --check` → ok
- Verify: `/opt/rocm/bin/hipcc --version` succeeds

**5080 laptop (phase 2 only)**
- CUDA Toolkit ≥ 12.0
- HIP-on-CUDA: `apt install hip-dev hip-runtime-nvidia` (or ROCm
  source-built equivalent)
- Verify: `hipconfig --platform` → nvidia; test compile of a simple
  `.hip` with `--offload-arch=sm_120` or whichever sm level Blackwell
  uses.

**Both boxes**
- Build a minimal `hello_hip.hip` kernel by hand, compile, run. This
  is M0. Until hello_hip runs on the respective box, MLRift GPU work
  on that box is blocked.

## Non-goals for phase 1

- Multi-GPU (single device only)
- CUDA Graphs / stream capture (sync after every launch)
- Tensor cores / WMMA (sparse spiking doesn't need them)
- Host-device unified memory (explicit copy only)
- Dynamic parallelism (host launches kernels, kernels don't launch kernels)
- Debugging infrastructure (use hipcc -g + rocgdb externally; no
  built-in MLRift GPU debugger)

These become revisitable after phase 2.

## Open questions to answer in M0-M1

- Kernel ABI: does MLRift's calling convention for `@kernel` fns
  map cleanly to HIP's `__global__` signatures? (Mostly yes — both
  are C-ABI adjacent; verify struct layouts for args passed by value.)
- Name mangling: HIP kernels don't mangle if declared `extern "C"`.
  Emit with `extern "C"` always.
- Error propagation: how does a kernel-side assert fail reach host
  MLRift's error machinery? Phase-1 answer: it doesn't. Kernels run
  and trust the caller to check hipGetLastError after.
- RNG: xorshift32 works fine per-thread on GPU, but seeding per-thread
  so kernel reruns are deterministic is non-obvious. Plan: host
  precomputes seed array, passes device-side array of seeds.

## Related work in the repo

- CPU codegen for reference: `src/ir.mlr` (x86_64), `src/ir_aarch64.mlr`
- Static-data emission (analogous to embedded GPU blob): `codegen.mlr
  emit_static_data()`
- Extern-call machinery (analogous to hip runtime dlopen): see how
  PE IAT calls are lowered in `src/format_pe.mlr` + the Windows path
  in `src/codegen.mlr` around `emit_win_call_iat`
- BSS optimisation (task #86 in conversation history) should land
  before M5 — GPU binaries will embed ~100 MB of kernel blobs, and
  the current output-buffer path will put those in .text. BSS fixes
  that.

## Kickoff actions for the next session

1. On the 7800XT box: install ROCm, verify `hipconfig --check`, write
   and run `hello_hip.hip`. Send output.
2. Read this document end-to-end, then read `src/format_pe.mlr` as the
   template for "emit a binary format one byte at a time" and
   `src/ir.mlr` for IR emission patterns.
3. Propose the IR opcode numbers for the new GPU ops (pick free slots).
4. Stub `src/format_hip.mlr` with a no-op function signature that
   compiles into the MLRift build.
5. Begin M1 (trivial @kernel → .hip emission).

End of spec.
