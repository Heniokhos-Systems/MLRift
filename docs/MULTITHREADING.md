# Multi-threading in MLRift

How the thread pool works, how to use it, and what it will not do.

**Platform: Linux x86_64 only.** Built on raw `clone()` + `futex()` syscalls —
no libc, no libpthread, no dynamic linker. macOS and Windows have no
implementation; the pool silently has zero workers there and every `_mt`
helper falls back to its single-threaded path.

Implementation: `std/thread.mlr`. Smoke test: `examples/thread_pool_hello.mlr`.

## Quick start

```
import "std/thread.mlr"

// Single-element array so the bare identifier evaluates to the slot address.
static uint64[1] shared_counter

fn bump(uint64 start, uint64 end, uint64 counter_addr) {
    uint64 i = start
    while i < end {
        atomic_add(counter_addr, 1)
        i = i + 1
    }
}

fn main() {
    thread_pool_init(6)                       // spawn 6 workers, once
    uint64 job     = fn_addr("bump")
    uint64 counter = shared_counter

    uint64 w = 0
    while w < 6 {
        thread_pool_submit(w, job, w * 100, w * 100 + 100, counter)
        w = w + 1
    }
    thread_pool_wait()                        // barrier — 600 increments done

    thread_pool_shutdown()
}
```

That is `examples/thread_pool_hello.mlr` reduced to one round; run it with
`mlrc --arch=x86_64 examples/thread_pool_hello.mlr -o /tmp/tph`.

The submitted function is called as `fn(a0, a1, a2)` inside the worker. Splitting
a range into `[start, end)` chunks is a *caller* convention, not something the
pool imposes — the three arguments are yours to use however you like.

## API

| Function | Notes |
|---|---|
| `thread_pool_init(n)` | Spawn `n` workers. Returns nothing — there is no pool handle; the pool is a process-wide singleton. **Idempotent**: if the pool already has `n` or more workers it returns unchanged. Capped at `TP_MAX = 32`. |
| `thread_pool_submit(wid, fn_ptr, a0, a1, a2)` | Hand a job to worker `wid`. Non-blocking. |
| `thread_pool_wait()` | Barrier on all workers that have an outstanding job. |
| `thread_pool_shutdown()` | Terminate every worker thread and reset the pool to empty. |

`tp_n` holds the current worker count — read it to decide whether to
parallelise at all.

Each worker gets a 2 MB stack (`TP_STACK_SIZE`), matching the glibc default.

## How it works

Each worker owns a state word, and everything is driven by that one value:

| state | meaning |
|---|---|
| 0 | idle |
| 1 | has job |
| 2 | done, awaiting reset by main |
| 3 | shutdown |

**Submit** writes the job descriptor (`fn`, `a0`, `a1`, `a2`) into the worker's
slots, then `atomic_store`s state to 1 and wakes the futex. The atomic store is
a full barrier, so the worker's subsequent reads see the committed descriptor.

**The worker loop** sleeps on the futex whenever state is 0 or 2, waking only
for 1 (job) or 3 (shutdown). Sleeping on 2 matters: a worker that finished
before main got round to resetting it would otherwise spin and burn a core.

**Wait** polls each worker until state reaches 2, then resets it to 0 for the
next round. Workers still at 0 are *skipped* — they were never submitted to
this round, and blocking on them would hang forever since nothing will ever
flip them to 2.

**Shutdown** sets state 3 and wakes each worker; the worker calls `exit` (60)
for itself only, not the process.

### Spawning

`tp_spawn_raw` issues `clone()` directly with

```
CLONE_VM | CLONE_FS | CLONE_FILES | CLONE_SIGHAND | CLONE_THREAD | CLONE_SYSVSEM
```

Before the syscall it stashes `(worker_fn, ctx_ptr)` on the child's stack. After
it, `rax == 0` means we are the child: pop both, call through the function
pointer, and `exit` if it ever returns. The parent falls through and captures
the child TID.

This has to be raw machine-code bytes, because the child's entry point *is* the
instruction after the syscall but running on a different stack.

### Why the state arrays are heap-allocated

They are `alloc`'d in `thread_pool_init`, not declared as `static uint64[32]`.
The static allocator does not reliably 8-byte-align static arrays, and the Linux
futex syscall returns `EINVAL` on a misaligned `uaddr`. `alloc` hands back
page-aligned memory, which satisfies the 4-byte requirement trivially.

The futex watches the low 32 bits of a `uint64`; on little-endian x86_64 that
coincides with the low byte of the value.

## Writing a parallel helper

The `_mt` helpers in `std/vec_f64_mt.mlr` and `std/matmul.mlr` follow one shape.
Two details in it are worth copying deliberately.

**Fall back when it isn't worth it.** Every helper checks the pool size and a
size threshold first:

```
uint64 nw = tp_n
if nw == 0 || n < VEC_MT_THRESHOLD {     // 500000 in vec_f64_mt.mlr
    v_decay(buf, n, factor)              // single-threaded path
    return
}
```

`nw == 0` is what makes these helpers safe on macOS and Windows, and safe for a
caller who never called `thread_pool_init`.

**Let main run the last slice.** `std/matmul.mlr` submits to `num_workers - 1`
workers and runs the final chunk on the calling thread:

```
u64 main_start = (num_workers - 1) * chunk
u64 main_end   = N                       // absorbs the ragged tail
mm_worker_bf16_f32_avx2_naive_2w(main_start, main_end, main_ctx)
thread_pool_wait()
```

Main would otherwise sit blocked in `wait()` doing nothing, and ending the last
slice at `N` handles `N` not dividing evenly by the worker count without a
separate tail case.

### Passing more than three arguments

`thread_pool_submit` takes exactly three. For anything larger, pass a pointer to
a per-worker context block. `std/matmul.mlr` strides these blocks by **64 bytes
— one cache line per worker** so that workers writing their own context do not
false-share:

```
u64 ctx_w = mm_ctx_base + w * 64
thread_pool_submit(w, fn_ptr, start_n, end_n, ctx_w)
```

Globals also work, and the `vec_f64_mt.mlr` helpers use them (`vec_mt_buf`,
`vec_mt_factor`) for values identical across all workers. Anything a worker
*writes* wants the per-worker block instead.

## Gotchas

- **Inline asm in `std/thread.mlr` may only name caller-saved registers**
  (`rax`, `rcx`, `rdx`, `rsi`, `rdi`, `r8`–`r11`). The compiler does not add
  asm-named registers to a function's prologue save set, so naming a
  callee-saved register (`rbx`, `rbp`, `r12`–`r15`) silently destroys a live
  value in the caller. This was a real crash: `flags -> r15` segfaulted the pool
  the moment the register allocator started handing `r15` to callers.
- **`thread_pool_init` is deliberately idempotent.** Two independent components
  initialising the pool used to double-spawn workers onto the same state slots
  and deadlock the futex hand-off.
- **Never block on a worker you did not submit to.** `thread_pool_wait` skips
  state-0 workers for this reason; hand-rolled wait loops must do the same.
- **Workers write to disjoint ranges of the same array.** That is safe, but
  chunk boundaries can false-share a cache line. Pad to 64-byte multiples if a
  benchmark shows it hurting.

## What this does not do

- **No dynamic scheduling** — chunks are a static `n / nw` split, no
  work-stealing. Fine for regular loops; load imbalance is on the caller.
- **No thread-local storage** — everything lives in shared arrays indexed by the
  worker's assigned chunk.
- **No cross-platform support** — Linux x86_64 only. macOS would need Mach
  threads or libSystem pthreads; Windows would need `CreateThread` via
  `kernel32`.
- **No nested parallelism** — workers must not submit to the pool themselves.

## Where things live

| File | Contents |
|---|---|
| `std/thread.mlr` | The pool: spawn, worker loop, submit/wait/shutdown |
| `std/vec_f64_mt.mlr` | Parallel f64 vector ops, with AVX2 variants |
| `std/matmul.mlr` | Column-sharded matmul, the per-worker ctx-block pattern |
| `std/inference.mlr` | `matmul_bf16_weights_f32_mt` — the LLM hot path |
| `examples/thread_pool_hello.mlr` | Smoke test: 6 workers × 10 rounds |
| `examples/clone_hello.mlr` | Minimal single `clone()` spawn, no pool |
| `examples/vec_microbench_mt.mlr`, `examples/mt_avx2_all.mlr` | Benchmarks |
