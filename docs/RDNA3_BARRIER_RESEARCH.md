# RDNA3 / GFX11 cross-workgroup synchronization — research result

**Conclusion: GPU-wide barrier IS available on RDNA3, via the cooperative-launch + software ring-barrier pattern.  Realistic mega-kernel ceiling on this hardware: ~120 tok/s (+30 % over 91 tok/s baseline).  Implementation effort: 2–3 weeks, high deadlock risk.**

## Mechanism

ROCm's `cooperative_groups::this_grid().sync()` resolves to
`__ockl_grid_sync` (file `ockl.bc`).  Disassembled:

```llvm
define hidden void @__ockl_grid_sync() {
  fence syncscope("agent") seq_cst                  ; L2-coherent fence
  call void @llvm.amdgcn.s.barrier()                 ; intra-WG sync
  ; if (lane 0 of WG):
  ;   atomic_add(counter, 1)
  ;   if last to arrive:   atomic_add(counter, 65536 - grid_size)
  ;   else:                spin until (counter >> 16) phase bit advances
  ; s_barrier               ; re-sync within WG
}
```

The "counter" is a single 32-bit word in device memory:
- low 16 bits: workgroups arrived in current phase
- high 16 bits: phase number (every grid-wide barrier flips it)

The hardware doesn't have a dedicated GPU-wide barrier instruction — this
is purely a **software** construct on top of `global_atomic_add` +
agent-scope memory ordering.  L2 is coherent across all CUs on RDNA3,
so atomic ops to global memory are visible everywhere after the fence.

## The deadlock constraint

The barrier requires every workgroup in the grid to call it.  If
grid_size > max concurrent workgroups, queued workgroups never run
(because running ones are spinning), and the kernel hangs forever.

ROCr enforces this via the HSA `cooperative` launch flag: the runtime
checks grid_size ≤ max_active_blocks on the target device and refuses
to launch otherwise.  We'd need the same check in the KFD shim.

### Max concurrent WGs on RX 7800 XT (60 CUs, gfx1100)

| Block size | Max concurrent WGs                      | Implication for qwen3        |
|------------|-----------------------------------------|------------------------------|
| 32         | 60 × 64 = **3 840** (1 wave per WG)     | gate_up N=6144 deadlocks     |
| 128        | 60 × 16 = **960**  (4 waves per WG)     | qkv N=4096 deadlocks         |
| 1024       | 60 × 1  = **60**   (32 waves per WG)    | tight — gemv impossible      |

For a fused mega-kernel that performs ALL Qwen3 phases without leaving
the kernel, every phase needs grid_size ≤ its block-size's concurrency
cap.  Workarounds:

1. **Multi-row WG**: each WG produces N rows of matmul output instead
   of 1.  At block=32, doing 2 rows/WG halves grid size — gate_up
   N=6144 needs grid=3072, fits under 3840.
2. **Heterogeneous block per phase, fixed grid**: pick one grid_size
   (e.g. 2 048) and let each phase scale its work-per-WG.  Each phase
   has different lane-utilization but no deadlock.

## Mega-kernel time budget (revised)

Using software grid_sync as the inter-phase barrier:

| Component                          | Current (ms/step) | Mega-kernel | Δ         |
|------------------------------------|-------------------|-------------|-----------|
| Per-layer sync wait (28 × ~150 µs) | 4.2               | 0           | -4.2      |
| Inter-layer kernel launch (27 × 5 µs) | 0.13           | 0           | -0.13     |
| Barriers (14/layer × 28 × ~1 µs)   | 0                 | 0.4         | +0.4      |
| GPU compute                        | ~12.0             | ~12.0       | 0         |
| Other host work                    | ~0.6              | ~0.6        | 0         |
| **Total**                          | **~17**           | **~13**     | **-4 ms** |

→ ~120 tok/s (+30 % over 91).  Below the original "+50 %" projection
because mega-kernel can't reduce the actual GPU compute time, only
the sync overhead.

## Implementation roadmap (if pursued)

1. **KFD shim cooperative launch path** (~1 week)
   - Add `hipModuleLaunchKernelCooperative` (or env-gated variant).
   - At init: query max_concurrent_wgs from device props.
   - At launch: allocate counter buffer (one device-resident u32),
     fill ABI-v500 implicit-arg slot at offset 88 with pointer to it.
     Reject if grid_size > max_concurrent.
2. **Persistent megakernel emitter** (~1–2 weeks)
   - One ASM blob containing all 15 Qwen3 phases per layer + 28-layer
     loop in scalar SGPRs.
   - Pick (block, grid) shape (likely block=128, grid=960 or so).
   - Each phase scales work-per-WG to fit the global shape.
   - Insert grid_sync between phases (the 10-ish instructions above).
3. **Bit-exact validation** (~3–5 days)
   - Smoke test mega-kernel against the existing chain on identical
     inputs; verify f32 output matches bit-for-bit.
   - Risk: one wrong barrier placement and the GPU hangs (timeout
     30 s wall via existing kfd_die path).

## Should we do it?

**Pros:**
- +30 % perf is a real margin (~120 tok/s = +60 % over PyTorch ROCm bf16
  74 tok/s).
- Builds infrastructure for future fusion experiments (any persistent-
  kernel design needs cooperative launch).

**Cons:**
- 2–3 weeks of focused work for a single 30 % gain.
- High deadlock risk during development.
- We're already +25 % over PyTorch with current architecture.
- Doesn't generalize to other models without re-emitting the persistent
  kernel for each layer shape.

**Recommendation:** Defer.  The +25 % over PyTorch is already a
publishable result; the next 30 % is high-effort, narrow-scope, and
brittle.  Better near-term targets:
- Multi-stream batching (real prefill + concurrent decode)
- 4-bit weight quantization (halves bandwidth pressure → maybe +50 %)
- Apply the techniques to a larger model where the launch overhead
  amortizes better

## Files referenced
- `/home/pantelis/Desktop/Projects/Work/venv/lib/python3.12/site-packages/triton/backends/amd/lib/ockl.bc` — disassembled to confirm grid_sync impl
- `/opt/rocm/include/hip/amd_detail/hip_cooperative_groups_helper.h` — public API
- `docs/MEGAKERNEL_DESIGN.md` — earlier (incorrect) "no GPU-wide sync exists" conclusion
- `docs/SYNC_LAUNCH_GAP.md` — prior sync-launch optimization campaign
