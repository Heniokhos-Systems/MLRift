# Bug: writes to a large `alloc()` block segfault past a certain offset

**Date observed:** 2026-05-08
**Reproducer environment:** mlrc target=amdgpu-native, kfd-amd shim, RX 7800 XT (gfx1100)
**Severity:** silent runtime corruption (segfault delayed to next allocation, no compile error)

## Summary

When growing an existing host-side `alloc()` block from 41 slots (328 B) to 51 slots
(408 B) and writing past the original 328 B boundary, the process segfaults — but
not at the write itself. The segfault appears later, on a subsequent allocation in
the same function, with a backtrace that points to unrelated code.

The same writes succeed when placed in a **separate** `alloc()` of size 80 B.

## Reproduction context

In `/home/pantelis/Desktop/Projects/Work/Noesis/phase_y_nback.mlr`,
`area_new_basic()` allocates a per-area `slot_block` that holds device pointers
plus per-pathway kernarg tables. The block was originally 22 slots, grew to 41
during PhaseT (V_thresh homeostasis added 8 fields + 11 kernarg slots), and
attempted to grow to 51 during PhaseU (lateral inhibition added 3 scalars + 7
kernarg slots).

```mlr
fn area_new_basic(uint64 n, ...) -> uint64 {
    // ... existing allocs (V, ref, Iex, gc, gn, vth, fh, pf, pm, ps, vth_base) ...

    uint64 slot_block = alloc(51 * 8)   // 408 bytes requested

    // ... 30+ existing writes at offsets 0..327 succeed ...

    // BAD: write past offset 327 — even though alloc was 408 bytes
    uint64 a_li_qs = slot_block + 328
    uint64 li_qs_v = 100
    unsafe { *(a_li_qs as uint64) = li_qs_v }    // <— silent corruption
}
```

`area_new_basic` is called once per area in a 16-area genome. The function returns
successfully, but the FIRST area's PROBE prints reach stdout, then the SECOND
call (or later GPU buffer alloc) segfaults.

Concrete failure log:
```
PROBE area0 d_V    kind =0
PROBE area0 d_fh   kind =0
Segmentation fault (core dumped)
```

## What was tried (isolation)

| Variant | Result |
|---|---|
| `alloc(41 * 8)` — original size, no new writes | Works |
| `alloc(51 * 8)` — bigger size, no new writes | Works |
| `alloc(51 * 8)` + ONE uint64 write at offset 328 | Segfault |
| `alloc(51 * 8)` + ONE uint64 write at offset 240 (overwrites existing slot) | Segfault (expected — corrupts existing kp) |
| `alloc(512)` literal + write at offset 328 | Segfault |
| `alloc(4096)` literal + write at offset 328 | Segfault |
| `alloc(80)` SEPARATE buffer + writes at offsets 0..72 | **Works** |

So the failure mode is specifically about writing past offset ~327 on the
*existing* slot_block, regardless of how many bytes were requested. A fresh
separate alloc has no problem accepting writes at any offset.

## Hypothesis (unverified)

Most likely the host-side `alloc()` rounds requests to internal bucket sizes
(e.g. 128 / 256 / 384 / ...). If the bucket boundary lies between 328 (works at
PhaseT) and 408 (PhaseU ask), `alloc(408)` may still return a smaller chunk, and
writes past the actual chunk size clobber another allocation's metadata. The
segfault shows up on the *next* allocator call (which finds corrupted free-list
metadata), not at the offending write.

This is consistent with the "delayed crash with no obvious connection to the
buggy line" symptom.

Less likely alternatives:
- Codegen issue with large constant offsets in `unsafe { *(ptr + N) = ... }`
  (but `(ptr + 327)` works and `(ptr + 328)` doesn't — too narrow a window for
  a codegen issue to be the root cause)
- An alignment requirement that's documented somewhere I haven't found

## Minimal reproducer (suggested)

To confirm whether this is an `alloc()` bucket issue, a minimal program would be
useful:

```mlr
fn main() {
    uint64 a = alloc(51 * 8)        // request 408 bytes
    unsafe { *((a + 400) as uint64) = 42 }   // write at offset 400 (within 408)
    // If alloc bucket is e.g. 384 bytes, the write is past the actual chunk.
    // Either segfault here or on the next alloc/write.

    uint64 b = alloc(80)            // next alloc — likely faults if metadata trashed
    unsafe { *(b as uint64) = 1 }
    puts("ok")
}
```

If this segfaults, the alloc bucket hypothesis is confirmed. If it works, the
issue is more specific to the surrounding code in `area_new_basic` and would
need closer investigation.

## Workaround

Use a separate `alloc()` for new state instead of growing an existing block:

```mlr
// Instead of:
//   uint64 slot_block = alloc(51 * 8)
//   ... writes at offset 328+ in slot_block ...
//
// Use:
uint64 slot_block = alloc(41 * 8)        // keep old size
uint64 li_state   = alloc(80)            // separate alloc for new fields
// ... writes at offset 0+ in li_state — works fine
```

This is what the launcher now does (`phase_y_nback.mlr`, PhaseU). The downside
is one extra small allocation per area (per-eval cost is negligible).

## Possible compiler/runtime fixes

If the bucket-size hypothesis is correct:
1. Make `alloc()` always honor the requested size exactly (return the largest
   bucket that fits, but also `mprotect`/guard pages so over-writes fault
   immediately at the offending write rather than silently corrupting the heap)
2. Document the bucket sizes so users can size their allocations to bucket
   boundaries
3. Add a debug-build assertion that traps writes past `requested_size`

## Files involved

- `/home/pantelis/Desktop/Projects/Work/Noesis/phase_y_nback.mlr` — the launcher
  where the bug surfaced (search for "PhaseU" to see the workaround)
- `/home/pantelis/Desktop/Projects/Work/MLRift/std/mem.mlr` — std memory helpers
  (alloc itself is a primitive, not defined in std)

---

## Root cause analysis (2026-05-08, second pass)

**Status: NOT REPRODUCIBLE in pure MLRift. The bucket-allocator hypothesis
in this doc is wrong. The bug is not in MLRift's `alloc()` primitive or any
runtime layer that MLRift ships.**

### What `alloc()` actually does

`alloc(N)` is a compiler builtin. In `src/ir.mlr` the IR_ALLOC opcode
lowers (per OS) to a single inline syscall:

```
mmap(addr=0, len=N+8, prot=PROT_READ|PROT_WRITE,
     flags=MAP_PRIVATE|MAP_ANONYMOUS, fd=-1, offset=0)
```

Then the size is stashed at the returned page (`*base = N`) and the user
gets `base + 8`. Linux page-rounds the length, so:

- `alloc(408)` → `mmap(0, 416, RW, ...)` → 4 KiB mapping at some page-aligned
  base; user pointer `base+8`; user-visible region `[base+8, base+8+408)`.
- A user write at offset 328 lands at `base+8+328 = base+336`. **That is
  almost exactly the start of the page.** It cannot collide with anything.

There is no bucket allocator, no free-list, no metadata between user
allocations. Every `alloc()` is a fresh `mmap`. There is no way for a
write at offset 328 of an `alloc(408)` block to corrupt heap state, because
there is no heap state to corrupt.

### Reproducer attempts that failed to fault

A pure-MLRift reproducer was added at
`examples/alloc_large_block_repro.mlr`. It builds 4 progressively meaner
patterns:

| Pattern | Description | Result |
|---|---|---|
| `pat_a` | Direct from doc: `alloc(51*8)` + write@400 + alloc(80) + write | OK |
| `pat_b` | Loop `alloc(408+i*8)` for i=0..63, fill all bytes 0xAB, verify last byte | OK |
| `pat_c` | 100 × `alloc(80)`, then `alloc(408)`+writes 0..400, then `alloc(96)` | OK |
| `pat_d` | Full faithful copy of `area_new_basic` slot_block writes (30+) including offsets 328..400, then subsequent `alloc(96)` | OK |

Run on both `--target=linux` and `--target=amdgpu-native`. All four pass on
both. `alloc()` does not corrupt and writes do not fault.

```
$ ./build/mlrc examples/alloc_large_block_repro.mlr -o /tmp/x \
    --arch=x86_64 --target=amdgpu-native --emit=elfexe
$ /tmp/x
OK pat_a
OK pat_b
OK pat_c
OK pat_d
ALL OK alloc-large-block-repro
```

### What the bug is more likely

The Noesis launcher (`phase_y_nback.mlr`) imports `MLRift/std/hip.mlr`,
which under `--target=amdgpu-native` is rewritten to `hip_kfd.mlr` (the
KFD shim) by `_maybe_redirect_hip_to_kfd`. The shim calls `kfd.mlr` which
in turn manages an alloc tracker (`_kfd_alloc_*` 65 K-entry table), a VA
reservation pool (`_kfd_reserve_va` → PROT_NONE mmaps), and a host-mapped
ring buffer per queue. None of those are involved in the pure-MLRift
reproducer above.

Plausible real causes (none directly testable without a Noesis-side
session):

1. **Memcpy or struct-init overshoot in Noesis itself.** Some unrelated
   write in `area_new_basic` or its callees writes a few bytes past the
   end of an `alloc(408)` slot_block, into a neighboring page. The neighbor
   page is a `_kfd_reserve_va()` PROT_NONE mapping — *which would fault
   exactly when written, not later*. Unless the overshoot stays inside the
   first page (offsets 408..4096), in which case it is silent until the
   next `alloc()` happens to land there. Worth re-auditing each unsafe
   store offset in the failing path.

2. **`_kfd_alloc_*` tracker corruption upstream.** Some earlier
   `dev_alloc_smart` / `dev_free` race or off-by-one bumps
   `_kfd_alloc_tab_n` past the tracker capacity, and the next allocation
   walks corrupted bookkeeping. The PROBE failure pattern (probe area0
   d_V/d_fh OK, segfault on next probe) is consistent with the tracker
   getting a bad entry between probes — a `dev_alloc_kind(va)` linear walk
   over corrupt state will fault deterministically. The 2026-05-04
   `AMDGPU_NATIVE_FINDINGS.md` explicitly notes that `dev_free` does NOT
   remove entries (kfd.mlr:501) — the table grows monotonically, but is
   capped at 65,536 long before this matters for one episode.

3. **Driver-side amdkfd VA collision.** `_kfd_reserve_va` uses
   `MAP_NORESERVE` which silently allows over-commit. Writing into a
   reserved-but-not-backed range may transiently succeed and only fault
   when the next `mmap` claims overlapping VA. We have no evidence either
   way; would need an `strace -f -e mmap,munmap` of a failing run.

4. **Noesis launcher emits raw HIP bytes from PhaseU stages that point
   into `slot_block` past offset 327.** The kp tables ARE pointers into
   slot_block; if those pointers are passed to the kernel as kernarg.va
   addresses, the kernel may dereference them in user space (via the
   AMD-IOMMU path) and a stale or aliased entry could fault on dispatch.

### Recommended next steps

This investigation should run on the Noesis side, not the MLRift side:

1. Re-introduce the failing pattern (revert the `li_state = alloc(80)`
   workaround back to writing past offset 327 of the merged
   `slot_block = alloc(408)`).
2. Run under `strace -f -e mmap,munmap,brk -o /tmp/trace.log` to capture
   the actual VAs returned, then check whether the `slot_block` page's
   neighbor is a `MAP_NORESERVE` PROT_NONE region.
3. Run under `valgrind --tool=memcheck` if the binary will load (it might
   not — `--target=amdgpu-native` runs raw KFD ioctls valgrind doesn't
   shim).
4. Try running on `--target=linux` with `hipMalloc` etc. stubbed to plain
   `alloc` and a CPU-side LIF kernel emulator. If the bug **also**
   reproduces there, then the issue is in Noesis. If it does not, the
   issue is in the KFD shim's allocation/tracking layer.

The pure-MLRift `alloc()` primitive is not at fault. Reopen the bug
against Noesis (or the `std/kfd.mlr` shim, if step 4 above pins it there)
once a reproducer is captured.

