# MLRift multi-threading — kickoff spec

Status: **design-approved, implementation not started.** Sister doc to
`docs/GPU_BACKEND.md`. A fresh session picks up from here.

## Goal

Parallelize the per-step simulation phases across CPU cores — 8-16×
wall-clock speedup on tier-1 Noesis workloads, stacking on whatever
AVX2 codegen lands later.

The real prize: **move stage 13's 9.3 s runtime closer to 1 s**, which
puts Noesis tier-1 on a single CPU node under the 5-minute-per-genome
target *without needing GPU at all* for the development loop.

## Why not pthreads

The standard libpthread route requires:
- Dynamic linking (MLRift currently emits static ELFs)
- dlopen infrastructure (doesn't exist in MLRift today)
- libc dependency at runtime (breaks the "self-contained binary" property)

Instead use **raw Linux `clone()` + `futex()` syscalls**. Matches
MLRift's "emit bytes, no runtime deps" ethos and avoids the
linker/loader complexity. Tradeoff: we write the thread-pool and
synchronization primitives ourselves, ~300-500 lines of kr.

macOS doesn't have clone/futex — Mach threads + pthreads-via-libSystem
instead. Phase 2 only. Linux-first.

## Architecture

```
  Main thread                    Worker N (of 4-16)
  ───────────                    ──────────────────
  startup:
    init_thread_pool(n_workers)
      → spawn N workers via clone()
      → each worker enters spin-wait on job_queue[worker_id]
      → workers block on futex when idle
  ...
  per simulation step:
    for phase in (decay, integrate, deliver, stdp):
      dispatch_parallel(fn, buf, n, extra_args)
        → split range 0..n into N chunks
        → for each worker: write (fn, start, end, args) into
          job_queue[worker_id], futex_wake
        → main waits on completion_barrier (atomic counter == N)
  ...
  shutdown:
    signal_shutdown() — workers exit
    join all via clone_exit + tid_futex
```

Pool is STATIC: created once at start, reused forever. No create/join
per barrier — that's what killed the naive fork-join model.

## New primitives needed

In `src/codegen.mlr` or a new `src/thread_runtime.mlr`:

### Syscalls

| syscall        | nr (x86_64) | nr (arm64) | purpose |
|---------------|-------------|------------|---------|
| clone         | 56          | 220        | spawn thread (CLONE_VM \| CLONE_FS \| CLONE_FILES \| CLONE_SIGHAND \| CLONE_THREAD \| CLONE_SYSVSEM) |
| futex         | 202         | 98         | wait / wake (FUTEX_WAIT_PRIVATE / FUTEX_WAKE_PRIVATE) |
| mmap          | 9           | 222        | allocate thread stacks (guard page + stack) |
| exit          | 60          | 93         | thread exit |

### IR ops / builtins

- `thread_pool_init(n_workers) -> uint64` — opaque pool handle
- `thread_pool_submit(pool, fn_ptr, start, end, ctx_ptr)` — submit work to one worker
- `thread_pool_wait(pool)` — barrier: wait for all outstanding jobs
- `atomic_fetch_add_u64(ptr, delta) -> uint64` — for completion counter (needs lock-prefix emit)
- `atomic_store_u32(ptr, val)`, `atomic_load_u32(ptr) -> uint32` — seq-cst stores/loads (plain on x86 with compiler barrier)

All of these can be kr functions that emit raw syscalls or use inline
asm — no external libraries.

### Function-pointer calls from new thread

kr already has `fn_addr("name")` → returns function address, and
`call_ptr(fn_ptr, args...)` → calls through a pointer. The worker
thread entry point receives a `(fn_ptr, ctx_ptr)` pair and does
`call_ptr(fn_ptr, ctx_ptr)` — reuses existing machinery.

## Clone-based thread spawn details

x86_64 clone() syscall return convention:
- rdi = flags
- rsi = child_stack (top of stack, 16-aligned)
- rdx = &parent_tid
- r10 = &child_tid
- r8  = new_tls
- rax = syscall_nr (56)
- after `syscall`: rax = child_tid (parent) OR 0 (child)

In the child, rsp is already set to child_stack. The first thing to do
is pop our entry args from the top of the stack (we'll have placed
`(fn_ptr, ctx_ptr)` there before the syscall), then call through the
fn_ptr.

On child return, syscall(exit) to terminate. Never returns to caller.

This requires **raw assembly bytes** because the child's entry point
is AFTER the clone syscall in parent code, but with a different call
stack. Simplest pattern: inline a small asm blob that does
```
    syscall (clone)
    test rax, rax
    jnz .parent_return
    ; in child
    pop rdi      ; ctx_ptr from top of stack
    pop rax      ; fn_ptr
    call rax
    mov rax, 60  ; exit
    xor rdi, rdi
    syscall
.parent_return:
    ret to caller
```

~30 bytes of x86 machine code. Write as an extern assembly block or
as an emit_byte sequence in codegen.mlr.

## Futex barrier

Simplest form: each worker has a private `state` uint32 (0=idle,
1=has_job, 2=job_done). Main writes job fields + sets state=1 +
`futex_wake(state, 1)`. Worker calls `futex_wait(state, 0)` when idle,
wakes, processes job, sets state=2 + futex_wake. Main barrier:
`for each worker: futex_wait(state[i], 1)` until state==2.

For the "start all at once" dispatch phase, batch the wakes.

## Scope expectations

Phase 2 of the CPU track ships in 3 commits:

**M1 — syscalls + thread pool primitive (1-2 days)**
  clone/futex/mmap bindings, thread_pool_init, thread_pool_submit,
  thread_pool_wait. Toy test: 4-worker pool, each increments a shared
  atomic counter to 100, main waits and verifies.

**M2 — parallel vec_f64_decay_inplace (1 day)**
  `vec_f64_decay_inplace_mt(buf, n, factor, pool)` — submits N/nw-size
  chunks, waits on completion. Benchmark on microbench (65k × 1000
  rounds):
  - target: 3× (close to nw=4) on the microbench
  - expected end-to-end: stage 12 ~1.4× (decay is ~30% of stage 12)

**M3 — parallel integrate + delivery (2-3 days)**
  Integrate is embarrassingly parallel per-neuron. Delivery has
  scatter-add conflicts — use atomic_fetch_add on s_exc[tgt] /
  s_inh[tgt]. Benchmark stage 12 full. Target: 2-3× end-to-end.

Total: ~1 week. Combined with BSS/helpers already shipped, CPU-track
gets stage 12 from 859 ms to maybe 250-350 ms. That's the "is it
close enough to GPU target" data we need.

## What this does NOT do

- **Dynamic scheduling** — static chunk size N/nw, no work-stealing.
  For regular loops (decay, integrate) that's fine. Delivery-phase
  load imbalance (some fired sources have more fan-out) is real but
  tolerable at phase-1 scale.
- **Thread-local storage** — not needed for our workloads; everything
  is in shared static arrays addressed by thread's assigned chunk.
- **False-sharing mitigation** — workers write to disjoint ranges of
  the SAME static array. Cache-line level false sharing could bite on
  the chunk boundaries. Not addressed in M2; if benchmark shows it
  hurting, pad chunk boundaries to 64-byte multiples.
- **Cross-platform** — Linux only initially. macOS needs Mach threads
  or libSystem pthreads; Windows needs CreateThread via kernel32.
  Both are phase-3 concerns.

## Testing strategy

Byte-identical output vs single-threaded kr version across per-neuron
spike counts. Scatter-adds in delivery are atomic, so the sum is
deterministic. Non-determinism could creep in from thread scheduling
affecting float-reduction order in sum aggregations — those are
OUTSIDE the per-step hot loop and not parallelised in M2/M3.

## Open questions for the M0 session to resolve

- Does MLRift's current binary (static ELF) tolerate multiple stacks
  from mmap?  Answer during M0: yes, mmap with MAP_STACK | MAP_GROWSDOWN
  + use the returned address as the new rsp for the cloned thread.
- Register save/restore at the clone boundary: child starts fresh,
  doesn't inherit caller regs. Fn-ptr entry discipline: ctx_ptr in
  rdi (first arg), nothing else relied on. Write the clone asm blob
  accordingly.
- Futex ABI: uint32 futex variables, FUTEX_WAIT returns 0 on success,
  EAGAIN if val already changed (no wait needed).

## Related work already shipped

- `src/codegen.mlr` `emit_win_call_iat`, `exec_process_argv` — patterns
  for emitting carefully-packed syscall sequences by hand
- `std/vec_f64.mlr` — the helpers this effort extends (vec_f64_decay_inplace
  gets an `_mt` counterpart)
- `docs/GPU_BACKEND.md` — sister doc; GPU track runs in parallel with
  this one

## Kickoff actions for the next session

1. Read this doc end-to-end
2. Write `examples/clone_hello.mlr` that uses raw clone()+futex() to
   spawn one worker, have it write to a shared counter, main reads it.
   This is M0 — gate before any IR or stdlib work.
3. Decide where the new primitives live (new file `std/thread.mlr`? new
   section in `std/mem.mlr`? builtin in codegen.mlr?).
4. Begin M1.

End of spec.
