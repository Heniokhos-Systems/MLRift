# MLRift SIMD codegen — kickoff spec

Status: **design-approved, implementation not started.** Third kickoff
doc alongside `GPU_BACKEND.md` and `MULTITHREADING.md`.

## Goal

Replace the 4-way scalar-unroll bodies of `std/vec_f64.mlr` helpers with
native AVX2 (x86_64) / NEON (ARM64) SIMD instructions. Callers don't
change.

Projected win on decay-heavy workloads: **2-3× on top of current 1.59×
scalar-unroll** (microbench), stacking to roughly 1.5-1.8× end-to-end
on stage 12, 1.1-1.3× on stage 13. Multiplies with multi-threading
when both ship.

## Priority vs the other tracks

- GPU (HIP source emission) — bigger win, bigger scope, independent
- Multi-threading — bigger end-to-end win than AVX2, similar scope
- **AVX2/NEON — smallest win, smallest scope, ships incrementally**

Reasonable ordering: MT first (3× wall-clock), AVX2 second (1.5× on
top). If GPU lands fast, AVX2 can deprioritise. If GPU stalls, AVX2 is
the cheap fallback.

## Approach

Two candidates:

### Approach A — hand-coded builtins in `src/codegen.mlr` (RECOMMENDED)

Add special-cased codegen for calls to `vec_f64_decay_inplace`,
`vec_f64_relax_inplace`, `vec_f64_fill`, `vec_f64_sum`. When the x86_64
backend sees a call to one of these names, emit AVX2 machine code
inline instead of a regular function call.

Pros:
- No new IR ops, no lexer/parser changes — the helpers stay plain
  kr functions; their bodies become unreachable (dead code)
- Drop-in: existing call sites in stages 12/13/15 _vec variants get
  the speedup automatically after rebuild
- Fallback: on non-AVX2 hardware (detect via CPUID at startup), the
  codegen emits a branch to the scalar body instead
- Per-helper incremental rollout — ship decay_inplace first, others
  follow

Cons:
- Encoding AVX2 VEX prefixes by hand is fiddly (~4-6 bytes per insn)
- AVX2 detection / cpuid at startup

This is the same pattern used for `exec_process_argv` (inline asm
emitted at specific builtin call sites).

### Approach B — full autovectoriser

Add vector IR ops (`IR_VEC_F64_LOAD`, `IR_VEC_F64_FMUL`, etc),
recognise vectorisable loops in the IR, emit SIMD automatically.

Pros:
- Every suitable scalar loop gets SIMD, not just the helpers

Cons:
- Proper autovectoriser is a 2-3 month project (alignment analysis,
  dependency analysis, scalar-remainder handling, cost model)
- Much larger blast radius; regressions likely on existing tests
- Overkill for the current Noesis-shaped workloads

**Go with Approach A.** Approach B is revisitable after MLRift has its
own neuron/synapse/area syntax and wants to autovectorise user code.

## AVX2 encoding reference (x86_64)

Target the hot path of `vec_f64_decay_inplace(buf, n, factor)`:

```asm
    vbroadcastsd ymm1, xmm0          ; ymm1 = [factor, factor, factor, factor]
    xor rcx, rcx
    mov rdx, rsi                     ; n
    and rdx, ~3                      ; n_aligned = n & ~3
.loop4:
    cmp rcx, rdx
    jge .tail
    vmovupd ymm0, [rdi + rcx*8]      ; ymm0 = buf[i..i+4]
    vfnmadd231pd ymm0, ymm1, ymm0    ; ymm0 = ymm0 - ymm0 * factor
    vmovupd [rdi + rcx*8], ymm0
    add rcx, 4
    jmp .loop4
.tail:
    cmp rcx, rsi
    jge .done
    movsd xmm2, [rdi + rcx*8]
    movapd xmm3, xmm2
    mulsd xmm3, xmm1                 ; xmm3 = x * factor (low 64 bits only)
    subsd xmm2, xmm3
    movsd [rdi + rcx*8], xmm2
    inc rcx
    jmp .tail
.done:
    ret
```

Encoded byte sequences (VEX prefixes):
- `vbroadcastsd ymm1, xmm0`: `C4 E2 7D 19 C8`
- `vmovupd ymm0, [rdi+rcx*8]`: `C5 FD 10 04 CF`
- `vfnmadd231pd ymm0, ymm1, ymm0`: `C4 E2 F5 BC C0`
- `vmovupd [rdi+rcx*8], ymm0`: `C5 FD 11 04 CF`

~70 bytes total for the whole function. vs ~200+ bytes for the
current scalar version.

## NEON encoding reference (ARM64)

Same workload as two `ld1 {v0.2d}, [x0]` (4 doubles via 2 loads) +
`fmls v0.2d, v1.2d, v0.2d` (multiply-subtract) + `st1 {v0.2d}, [x0]`.

~8 bytes per iteration (two 4-byte AArch64 insns). Simpler encoding
than AVX2's VEX.

## Hardware detection

At process start, `cpuid(1)` returns ECX with bit 28 = AVX support,
bit 12 = FMA. AVX2 is queried via `cpuid(7,0)` EBX bit 5. If both
present, install AVX2 versions of the helpers; else fall back to
scalar.

Simplest implementation: emit both variants, pick at startup via a
function-pointer table. Call-site overhead: one indirect call per
helper invocation. At helper-call frequency (thousands per sim run,
not per step), negligible.

Or: emit `cpuid` check inline at each call site, branch to AVX or
scalar. Lower call overhead but duplicated branches in every call
site. Prefer the function-pointer approach for simplicity.

## Milestones

**M1 — AVX2 vec_f64_decay_inplace (~1 day)**
  Emit the ~70-byte AVX2 function in codegen.mlr. Install as
  `vec_f64_decay_inplace_avx2`. CPUID at startup picks between it
  and the scalar version. Microbench and stage 12:
  - target microbench: 3-4× vs scalar (vs 1.59× today)
  - target stage 12: 1.5-1.8× end-to-end

**M2 — AVX2 relax / fill / sum (~0.5 day each)**
  Same pattern for the other three helpers. Relax is shaped like
  decay with an additional `target` broadcast. Fill is trivial
  (broadcast + vmovupd in a loop). Sum uses vertical reduction at
  the end (vhaddpd or manual).

**M3 — NEON variants (~2 days, depends on ARM64 test hardware)**
  Same functions, ARM64 codegen path. The Pi 400 is accessible but
  slow to test on. Defer until phase 1 AMD GPU integration gives us
  a reason to care about ARM64 perf.

**M4 — AVX-512 variants (optional, ~1 day)**
  AVX-512 doubles the vector width (8 f64 per op) on supported
  hardware. 5080 laptop probably has it; 7800XT host CPU probably
  doesn't. Marginal win. Ship when ready.

Total: phase 1 (AVX2 for all 4 helpers) ~3 days. phase 2 (NEON + AVX-
512) another ~3 days.

## Testing

Microbench already exists (`examples/vec_microbench.mlr`). Expand:
- correctness: run both variants, check sums are within tolerance
  (GPU-backend-style tolerance-diff, not byte-diff — reduction order
  changes with SIMD width)
- timing: report scalar vs AVX2 vs (later) NEON / AVX-512 ratios
- regression: stage 12 and stage 13 runtime in benchmark suite

## Related work

- `src/codegen.mlr` `exec_process_argv` — pattern for inline-asm
  emission at specific builtin call sites
- `std/vec_f64.mlr` — helpers whose bodies become the fallback
- `docs/MULTITHREADING.md` — stacks multiplicatively with SIMD once
  both ship

## Kickoff actions for the next session

1. Read this doc
2. Write a synthetic AVX2-encoding smoke test: emit the 5 instruction
   bytes for `vbroadcastsd ymm1, xmm0`, verify `objdump -d` disassembles
   to the expected mnemonic. Validates the encoding approach.
3. Add the `vec_f64_decay_inplace` builtin short-circuit in
   src/codegen.mlr (check is_extern_call == 0 && tok_matches to the
   helper name; emit the 70-byte function body directly).
4. CPUID dispatch at startup; wire into call lowering.
5. Microbench and measure.

End of spec.
