# Bugs / improvements found in MLRift that KernRift likely needs too

This is the list of compiler-level findings from the MLRift scaling
push that almost certainly apply to upstream KernRift as well — they
touch generic codegen / lexer / IR / language features, not the
MLRift-specific GPU or MT tracks.

Ordered by severity (correctness bugs first, features last).

---

## 1 · 16-argument call cap with silent truncation  — **critical**

`src/ir.mlr` (MLRift commit `4b79dd9`).

Three compounding caps silently truncated calls with more than 16
arguments:

1. `ir_lower`'s Call handler had `uint64[16] call_arg_vregs` and
   `while arg_count < 16` — args 17+ were never lowered.
2. The `IR_ARG` overflow buffer was `alloc(16 * 8)` with `if oi < 16`
   in the emit side — overflow slots ≥ 16 were dropped.
3. `IR_CALL`'s `sub rsp, stack_bytes` only emitted the imm8 form,
   which wraps when `stack_bytes ≥ 128` (~16+ stack args).

**Symptom:** a 22-arg call emitted `sub rsp, 0x50` (10 slots for
what should have been 16) and args 16+ read stale data from the
caller's frame. **Not a crash — silent wrong values**, which is worse.

**Upstream fix:** bump the two array caps to 32, remove the `< 16`
guards, add `imm32` paths for both `sub rsp` and
`mov [rsp+soff], reg` when `soff ≥ 128`.

---

## 2 · `f64 = int_var` emits bit-copy instead of `CVTSI2SD` — **critical**

`src/codegen.mlr` (MLRift commit `a79575c`).

Assigning an int-typed variable to an f64 variable was lowered as a
plain 8-byte load into an xmm register without the int-to-float
conversion. The bit pattern of the int ends up reinterpreted as
garbage-f64.

**Symptom:** anywhere user code does `f64 v = int_val`, the resulting
`v` is NaN / subnormal / wildly wrong. Silent.

**Upstream fix:** insert `CVTSI2SD xmm, r/m64` on assignment when the
LHS is f64 and the RHS vreg is int.

---

## 3 · Writable statics not emitted in the RW segment  — **critical**

`src/codegen.mlr` (MLRift commit `9242cda`).

Before the fix, `static` arrays whose initialisers were non-zero (or
were ever written from main) could land in the RO segment. Writes
from user code would `SIGSEGV`. Additionally, `f64` function-return
fkind wasn't being propagated through the ABI, so `f64 r =
indirect_call()` produced garbage even when the callee returned
properly in `xmm0`.

**Upstream fix:** split statics into RO / RW based on "any assignment
in the program references this offset", and propagate the callee's
f64 return kind through to the caller's vreg fkind table.

---

## 4 · Static arrays don't guarantee 8-byte base alignment  — **correctness**

Worked around in MLRift `std/thread.mlr` (`thread_pool_init` heap-
allocates the state slot arrays via `alloc()`). **Underlying
compiler bug is still present.**

**Symptom:** `static uint64[N] x` — if a preceding static's cumulative
size isn't a multiple of 8, `x`'s base address is mis-aligned.
On x86_64 this usually survives regular loads/stores at a perf hit,
but *breaks* the Linux `futex` syscall — a 4-byte-unaligned `uaddr`
returns `EINVAL` and futex-based synchronisation deadlocks.
`lock cmpxchg` on a mis-aligned cache-line-crossing address also hits
AMD/Intel split-lock detection which some kernels now panic on.

**Repro:** the first MLRift `thread_pool` version stored `tp_state`
as `static uint64[32]` and hung on 5+-worker pools. Fix was to
heap-allocate; root-cause is in the static layout pass.

**Upstream fix:** in `codegen.mlr`'s static allocator, round the
per-variable start offset up to its element alignment (8 for uint64 /
f64, 4 for uint32 / f32, etc.) before bumping `static_data_size`.

---

## 5 · `call_ptr` drops f64 return values  — **correctness**

Worked around in MLRift (`_into`-variant helpers: caller supplies an
out-pointer that the callee writes through). **Compiler bug is still
present.**

**Symptom:** `f64 r = call_ptr(fn_ptr, a, b)` always yields 0.0
(or whatever rax happens to be) regardless of whether the callee
returned a real f64 in xmm0. The IR `IR_CALL_IND` path assumes the
result lives in rax for all return types.

**Verification:** an `add_f64(1.5, 2.5)` smoke test through
`call_ptr` returns 0.0. The direct call (not through fn pointer)
returns 4.0 correctly.

**Upstream fix:** when the vreg receiving `IR_CALL_IND`'s result has
`fkind != 0` (f32/f64), move xmm0→dst instead of rax→dst.

---

## 6 · Local fixed-size arrays heap-allocated per call  — **major perf**

*Not* worked around in MLRift — it's a load-bearing perf trap.

**Symptom:** `uint64[1] scratch` declared as a local inside a function
body lowers to `IR_ALLOC(8)` — i.e. `alloc()` is called on *every
invocation* of that function. In hot paths (e.g. a CAS-loop helper
called millions of times), this is a ~2000× slowdown vs the stack
allocation the syntax implies.

**Repro:** `fn atomic_add_f64(ptr, delta)` using a local
`uint64[1] scratch` for the f64⇔u64 bit-cast ran at 2.27 µs per
call (240 ns per `alloc` + ~20 ns real work), while inlining the
same logic at the call site ran at 15 ns/call — the 151× gap is
all `alloc` overhead.

**Upstream fix:** when an array declaration has a constant size
known at parse time, emit a stack-frame reservation (`sub rsp,
size_bytes` during prologue, reference as `[rsp+offset]`) instead of
`IR_ALLOC`. This is the lowering C, Go, Rust all use for
fixed-size local arrays.

---

## 7 · `asm` block `in(var -> reg)` constraint moves aren't topo-sorted  — **sharp edge**

Worked around in MLRift (pick destination registers the compiler's
register allocator doesn't touch). **Compiler side still has the
race.**

**Symptom:** when two input constraints form a cycle on the
compiler's pinned/spill registers, e.g.

```
in(ctx_ptr -> r13, stack_top -> r12)
```

and the register allocator already has `ctx_ptr` in r12 and
`stack_top` in r13 (spilled params), the constraint moves are
emitted in declaration order:

```
mov r13, r12     ; ctx_ptr → r13 — OK, reads r12 which still holds ctx_ptr
mov r12, r13     ; stack_top → r12 — reads r13, which just got overwritten
```

The second move gets `ctx_ptr` instead of `stack_top`.

**Symptom:** the MLRift `clone()` thread-spawn asm block segfaulted
on the first `mov [r12-8], r14` because r12 had ended up holding 0
(the user's `ctx_ptr` arg) instead of the 2 MB stack top.

**Upstream fix:** build a swap-graph of the source-register set
across all `in(...)` constraints, topo-sort the moves so no
destination overwrites a not-yet-used source. Use a temp register
to break any remaining cycles. Same algorithm register allocators
use for parallel moves.

---

## 8 · Scientific notation float literals missing  — **feature**

`src/lexer.mlr`, `src/parser.mlr`, `src/ir.mlr` (changes partly landed
in MLRift `4b79dd9`; remainder in uncommitted MLRift working tree).

Before the patch, `1e-3` lexed as `1` (IntLit) + `e-3` (Ident-minus-
IntLit), breaking every ML-style constant like `lr = 5e-4` or
`tau = 1e-3`. Users had to spell them `0.0005`, `0.001`, etc.

**Upstream fix:** extend the numeric-literal state machine in the
lexer to accept `(e|E)[+-]?digits` after an integer or decimal
literal (lookahead required — don't eat a trailing `e` unless
followed by an optional sign and at least one digit, or it breaks
identifiers like `42exit`).

Parser's static-initialiser path and `ir.mlr`'s FloatLit lowering
then apply a compile-time multiply/divide by `10^|exp|`, packed as
a uint64 when `|exp| ≤ 18` (stays within f64's normal range).

---

## Notes on MLRift-specific workarounds that *don't* apply upstream

These ones are MLRift-only (touch the MT or GPU tracks which KernRift
doesn't have) — no upstream port needed, but worth knowing about in
case KernRift later adds the same features:

- `thread_pool_wait` skip-idle fix in `std/thread.mlr` (only if KernRift adds a thread pool)
- `DYN_SYM_MAX = 64 → 256` in `src/dyn_sym_registry.mlr` (GPU track's @dynamic extern registry — MLRift-only)
- `hip_workaround.cuh` patch for ROCm 7.2's 64-bit warp-mask static-assert (cupy patch, not kr)

---

## Summary of severity

| # | issue | severity | MLRift status | upstream action |
|---|---|---|---|---|
| 1 | 16-arg call cap | critical (silent wrong) | committed | port `4b79dd9` |
| 2 | f64 = int bit-copy | critical (silent wrong) | committed | port `a79575c` |
| 3 | Statics in RO segment | critical (SIGSEGV) | committed | port `9242cda` |
| 4 | Static array alignment | correctness (futex EINVAL) | worked around | fix codegen aligner |
| 5 | `call_ptr` drops f64 return | correctness (silent 0) | worked around | fix IR_CALL_IND result routing |
| 6 | Local arrays → `IR_ALLOC` | 2000× perf | untouched | stack-allocate in codegen |
| 7 | asm `in()` move-order race | sharp edge | worked around | topo-sort parallel moves |
| 8 | Scientific notation | missing feature | committed + uncommitted | port lexer + parser + ir changes |
