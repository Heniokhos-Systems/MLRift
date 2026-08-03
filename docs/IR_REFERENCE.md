# IR Reference

Everything below was checked against `src/` at commit `24796d5` (branch
`main`) and, where the check was runnable, against `build/mlrc`. Claims that
could not be verified are marked **UNVERIFIED** and say why. `file:line`
citations point at the code that establishes the claim — go read it before
you rely on it, because line numbers drift and this document does not.

MLRift is a fork of KernRift. Where the two IRs agree this document says so
and does not re-derive; where they diverge, the divergence is the finding and
is called out inline and collected in §15. KernRift's copy of this file is
the upstream counterpart.

> **Verification convention.** A claim is *verified* only if it was read out
> of source at this commit or observed by compiling and running a probe. This
> file previously carried several confident, wrong statements (the IR is not
> SSA; x86 does not use 5 colours; `IR_FFMA` is not a fused multiply-add;
> `-O0` is not the flag; the R1/R2 "known miscompiles" have been closed for
> over a year). A confident wrong IR reference is worse than a thin one,
> because it is trusted by whoever changes codegen next.

**Not covered here, deliberately:** the GPU pipeline
([`docs/GPU_BACKEND.md`](GPU_BACKEND.md),
[`docs/AMDGPU_NATIVE.md`](AMDGPU_NATIVE.md),
[`docs/KFD_GOTCHAS.md`](KFD_GOTCHAS.md)) and vector codegen
([`docs/SIMD_CODEGEN.md`](SIMD_CODEGEN.md)). The short version of why is in
§14 — neither currently goes through the IR at all.

---

## 1. What the IR is — and what it is not

The IR is a **flat, per-function, virtual-register instruction list with
explicit basic blocks**, produced by `ir_lower_function()`
(`src/ir.mlr:4834`) from the AST and consumed by four machine-code emitters.

**It is not SSA, despite the `--emit=ir` help text.** Verified two ways:

1. `IR_PHI` (opcode 60) is **completely dead**. Tree-wide, the only three
   references are the constant (`src/ir.mlr:73`), its name string in
   `ir_opcode_name` (`src/ir.mlr:4599`), and a stale comment
   (`src/ir_aarch64.mlr:2153`). No lowering emits it and no emitter handles
   it. There is no dominance or dominance-frontier computation anywhere:
   `grep -rniE 'dominan|frontier|idom' src/*.mlr` returns nothing.
2. A vreg is assigned in more than one block. Probe (`--emit=ir` on an
   if/else plus a `while`, run against `build/mlrc`):

   ```
   bb0:  v3 = copy v2        ← x = 1
         br_cond v5, bb1
   bb1:  v3 = copy v6        ← x = 2
   bb3:  v3 = copy v7        ← x = 3
   bb2:  v9 = copy v8        ← i = 0
   bb5:  v9 = copy v13       ← i = i + 1
   ```

   `v3` has three definitions and `v9` two. The identical probe against
   KernRift's `build/krc2` produces byte-for-byte the same dump.

What actually happens: the lowering keeps a *variable map* from name token to
"the vreg currently holding this variable". At a control-flow merge it
snapshots that map and emits reconciling `IR_COPY`s onto the incoming edges
via `ir_emit_copy_to_snapshot()` (`src/ir.mlr:1142`), using a two-phase
read-before-write ordering to avoid parallel-copy hazards. There is no phi
insertion and no SSA destruction pass. Passes that would need single
assignment carry their own single-def checks (§10).

> **Divergence.** KernRift's variable map is an FNV-1a open-addressed
> 4096-slot hash (`ir_var_hash`, `KernRift/src/ir.kr:940`). MLRift still uses
> a linear scan (`ir_var_find`, `src/ir.mlr:1013`). Same semantics, worse
> lowering time on large functions.

### Instruction record

32 bytes, fixed layout (accessors at `src/ir.mlr:448`–`510`):

| Offset | Width | Field | Notes |
|---|---|---|---|
| 0 | u32 | `opcode` | |
| 4 | u32 | `dest` | destination vreg; `0` = no destination |
| 8 | u32 | `src1` | `0` = unused |
| 12 | u32 | `src2` | `0` = unused |
| 16 | u64 | `imm` | interpretation is per-opcode |
| 24 | u32 | `bb` | owning basic block |
| 28 | u32 | `next` | **intrusive linked-list link**, `0xFFFFFFFF` = end |

The `next` field is why the arena is *not* walked as a contiguous index
range. LICM appends hoisted instructions at high indices and splices them
into a preheader's list; liveness and colouring walk the linked list.

One parallel, index-keyed side table: `ir_insn_src_tok` (`src/ir.mlr:222`),
u32 per instruction, holding the source token for the `-g` DWARF line
emitter.

> **Divergence.** KernRift has a **second** side table, `ir_insn_origin`
> (u64 per instruction), recording which surface builtin produced each
> instruction. MLRift has none — see §7 for why that matters and what it
> costs.

Basic blocks are 16-byte records — `first_insn`, `last_insn`, `succ0`,
`succ1`, each u32 with `0xFFFFFFFF` as the empty/none sentinel
(`ir_new_bb`, `src/ir.mlr:613`).

Initial capacities (`ir_init`, `src/ir.mlr:276`): 65536 instructions, 4096
basic blocks, 65536 vreg float-kind slots. The instruction arena grows; the
block arena does not — exceeding 4096 blocks is a hard
`error: IR basic block overflow` and `exit(1)`.

`ir_vreg_next` starts at 1 (`src/ir.mlr:293`), so **vreg 0 is reserved** and
means "no value".

**fkind** is a per-vreg byte tag: `0` = integer, `1` = f64, `2` = f32. It
selects the register class at emission. `ir_init` memsets the whole buffer,
because it is indexed without a length guard and a stale byte from the
previous function would mislabel a vreg as float.

---

## 2. Which invocations reach the IR, and which bypass it

Verified empirically at HEAD by compiling the same program with and without
`--legacy` and comparing bytes — identical output means `--legacy` changed
nothing, i.e. that mode was already using the legacy backend.

| Invocation | Backend | Evidence |
|---|---|---|
| default (fat `.mlbo`) | **IR** | differs from `--legacy` |
| `--emit=elfexe` | **IR** | differs from `--legacy` |
| `--emit=pe`, `--emit=macho`, `--emit=android` | **IR** | differs from `--legacy` |
| `--emit=asm` | **IR** | differs from `--legacy` |
| `--legacy` | legacy | by definition |
| **`--emit=obj` / `-c`** | **legacy** | byte-identical with and without `--legacy` |
| `--arch=riscv32`, `--arch=xtensa` (any emit mode) | **IR** | no legacy backend exists for these arches; `--legacy` is a silent no-op |
| `--emit=ir` | neither — lowering only | exits before codegen |
| `--target=hip-amd`, `--target=amdgpu-native`, `--emit-amdgpu-*` | **neither** | the GPU emitters walk the AST; see §14 |

The gate is one condition, in two places:

```
src/main.mlr:2746  if emit_ir_mode != 0 && arch == 0 && emit_mode != 3 {
src/main.mlr:2755  } else if emit_ir_mode != 0 && arch == 1 && emit_mode != 3 {
```

`emit_mode == 3` is `--emit=obj`. The in-source comment says why: *"skip for
`--emit=obj` which needs legacy for extern relocations"*.

> **Divergence.** KernRift's gate is `emit_mode != 3 && emit_mode != 7`,
> because it has an `--emit=lkm` (Linux kernel module) mode that MLRift does
> not. If you port a change from KernRift, do not port the `!= 7`.

> **The trap.** `--emit=obj` **is the legacy backend**. An `--emit=obj` test
> proves nothing whatsoever about IR lowering, and an IR-side guard leaves
> `--emit=obj` unguarded by construction. Upstream, this exact hole let a
> silently-wrong builtin ship after a full audit round.

Also verified, and frequently assumed backwards: **`--target=` selects the
ABI, not the container.** `--arch=x86_64 --target=windows` emits an ELF. PE
and Mach-O only come out of `--emit=pe` / `--emit=macho`. A byte-identity
gate built on `--target=` rows alone will never exercise the PE or Mach-O
header emitters.

### `--emit=ir` is pre-optimization

`ir_dump()` is called directly after `ir_lower_function()` with no
`ir_optimize()` in between. So `--emit=ir` shows what the lowering produced,
never what the optimizer did. Ops the optimizer creates (`IR_ADD_IMM`,
`IR_MUL_IMM`, `IR_LEA_BIS`, `IR_ROR`, `IR_SHL_IMM`, `IR_LOAD_BIS`,
`IR_STORE_BIS`) therefore never appear in a dump. To see post-optimization
code, read `--emit=asm` or disassemble.

`ir_opcode_name()` (`src/ir.mlr:4559`) is also **incomplete**: it has no
entry for 124, 125, or 135–148, so those print `???`. 124 (`IR_FMT_BOOL`) and
125 (`IR_FMT_F64`) are emitted by ordinary lowering, so this is observable on
any program that prints an `f64` or a `bool`. *(Identical gap in KernRift.)*

---

## 3. The per-OS dispatch shape, and where a new case goes

`target_os` values (assigned around `src/main.mlr:6650`–`6662`; defaults at
`3953`/`4180`/`4440`/`4694`/`4822`/`4951`):

| Value | OS |
|---|---|
| 0 | Linux |
| 1 | macOS / Darwin |
| 2 | Windows |
| 3 | Android |

**The house pattern is "special-case Windows/macOS, else fall through to
POSIX."** Measured at HEAD across the eight backend files (`src/ir.mlr
src/ir_aarch64.mlr src/ir_riscv.mlr src/ir_xtensa.mlr src/codegen.mlr
src/codegen_aarch64.mlr src/codegen_riscv.mlr src/codegen_xtensa.mlr`):

| Pattern | Count |
|---|---|
| `target_os == 2` (Windows) | 110 |
| `target_os == 1` (macOS) | 103 |
| `target_os != 2` | 11 |
| `target_os == 3` (Android) | 11 |
| **`target_os == 0` (Linux)** | **4** |
| `target_os != 0`, `target_os != 1` | 1 each |

Reproduce with:

```sh
grep -h "target_os == 0" src/ir.mlr src/ir_aarch64.mlr src/ir_riscv.mlr \
  src/ir_xtensa.mlr src/codegen.mlr src/codegen_aarch64.mlr \
  src/codegen_riscv.mlr src/codegen_xtensa.mlr \
  | grep -v '^[[:space:]]*//' | grep -o "target_os == 0" | wc -l
```

Four explicit Linux tests against 110 Windows tests is the whole story:
**Linux is not a case, it is the fall-through.** Consequences to design
around:

- **Adding a `target_os` value silently inherits Linux everywhere**, at
  roughly sixty branch sites. This is not hypothetical: upstream added
  `target_os == 4` (bare metal) and a Linux `syscall` instruction reached a
  kernel image because of it. MLRift has **no `target_os == 4`** — zero sites
  in the whole tree, verified — so the class does not exist here today. It
  will the moment a fifth OS is added.
- **The dangerous sub-class is a terminal `else` that produces a *value*.**
  `if os == A {…} else if os == B {…} else { mov rax, 0 }` compiles clean,
  exits 0, and ships an artifact containing a wrong constant. No trap-scanning
  guard can see it, because no trap instruction is emitted. Three such sites
  shipped upstream.
- **There is no arch × OS validation.** KernRift gained an enumerated
  allow-list (`arch_os_pair_supported`, per-row `if os == n { return 1 }` with
  a `return 0` fall-through, checked on the resolved pair before every compile
  entry). MLRift has none — `grep arch_os_pair_supported src/main.mlr`
  returns nothing. Before that check existed upstream,
  `--arch=riscv32 --target=windows` exited 0 and wrote a 296-byte RISC-V
  *ELF*: a PE was requested, an ELF was produced, and neither the requested
  OS nor the requested container appeared in the output. **That behaviour is
  still live here.** UNVERIFIED whether the exact byte count reproduces in
  MLRift; the absence of the check is verified.

**Where a new case goes.** A per-OS behaviour change or builtin refusal must
go in **both** the IR lowering (`src/ir.mlr`) **and** the legacy backend
(`src/codegen.mlr` / `src/codegen_aarch64.mlr`), because `--legacy` and
`--emit=obj` never call `ir_lower_expr`. Note also that `ir_lower_stmt`
(`src/ir.mlr:3562`–`4558`) contains **no `target_os` reference at all** —
statement-level constructs (local array decls, struct decls) are per-OS blind
by construction.

---

## 4. Reading an IR dump

```
$ mlrc --arch=x86_64 --emit=ir prog.mlr
function main:
  bb0:
    v1 = const 3
    arg v1 [0]
    v2 = call @32
    v0 = ret v2
```

- `call @32` — the `32` is a **token index**, not a symbol. See §7.
- `v0 = ret v2` — `v0` is the reserved "no destination" vreg.
- `arg v1 [0]` — argument position in `imm`.
- Blocks print in index order, which is also the emission order, but not the
  order the lowering created edges in: an `if`'s join block can have a lower
  index than its `else` arm.

---

## 5. Opcode reference

Numbers are the `static uint64 IR_* = N` constants at `src/ir.mlr:24`–`209`
plus `src/ir_hip.mlr:23`–`43`. Gaps (28–29, 33–39, 44–49, 53–59, 62–69, 89,
116–117) are unassigned.

**Opcodes 0–139 are numerically identical to KernRift's.** Everything from
140 up diverges; see §15 before copying any numeric literal between the
repos.

Legend for *Effect*: **SE** = in the side-effect set (survives DCE with a
dead `dest`, §9); **pure** = not in it.

### 5.1 Constants and integer arithmetic (1–13, 132–139, 148)

| # | Name | Semantics | Effect |
|---|---|---|---|
| 1 | `IR_CONST` | `dest = imm` | pure |
| 2–4 | `IR_ADD`/`SUB`/`MUL` | wrapping | pure |
| 5–6 | `IR_DIV`/`IR_MOD` | unsigned | pure |
| 7–9 | `IR_AND`/`OR`/`XOR` | | pure |
| 10–11 | `IR_SHL`/`IR_SHR` | logical | pure |
| 12–13 | `IR_NEG`/`IR_NOT` | | pure |
| 132–133 | `IR_SDIV`/`IR_SMOD` | signed | pure |
| 134 | `IR_SAR` | arithmetic shift right | pure |
| 135 | `IR_ADD_IMM` | `dest = src1 + imm` (signed i32); `src2` unused | pure |
| 136 | `IR_SUB_IMM` | `dest = src1 - imm` (signed i32); `src2` unused | pure |
| 137 | `IR_ROR` | rotate-right; `imm` = width | pure |
| 138 | `IR_MUL_IMM` | `dest = src1 * imm`; `src2` unused | pure |
| 139 | `IR_LEA_BIS` | `dest = src1 + src2 * imm`, `imm ∈ {1,2,4,8}` | pure |
| **148** | **`IR_SHL_IMM`** | `dest = src1 << imm` (0..63) — **KernRift calls this 143** | pure |

- Integer arithmetic wraps (two's complement). There is no
  signed-overflow-is-UB rule.
- Shift counts are masked to `& 63` by the hardware on both x86 and AArch64,
  and the compiler relies on it.
- **Divide by zero diverges by architecture, not by compiler choice.** x86
  `div` faults (SIGFPE); AArch64 `udiv` silently returns 0. `--debug` is the
  only thing that makes the two agree — it emits an explicit check that
  traps. Verified on KernRift, whose codegen for ops 5/6/132/133 is identical
  to MLRift's; **UNVERIFIED on MLRift directly** (not separately re-run).
- **135/136/138/139/148 are optimizer products only.** No lowering emits
  them.

> **Divergence, and it is a capability gap.** MLRift's const-folder **excludes
> arm64 from `ADD_IMM`/`SUB_IMM` fusion** (`src/ir.mlr:11683` gates on
> `0 || 2 || 3`), and consequently `src/ir_aarch64.mlr` has **no handler for
> ops 135/136** at all. KernRift includes arch 1 in the gate
> (`KernRift/src/ir.kr:12146`) and has the handler
> (`KernRift/src/ir_aarch64.kr:1257`). MLRift is internally consistent but
> emits worse arm64 code. Likewise cmp-with-immediate fusion is arch 0 only
> here (`src/ir.mlr:11590`) vs arch 0 and 1 upstream.

> **Correction to a standing project note.** "MLRift lacks pow2 MUL→SHL
> entirely" is **false**. It has it (`src/ir.mlr:11625`), gated to x86_64 and
> arm64. It lacks it only for riscv32/xtensa, where KernRift is ungated with
> a shamt guard.

### 5.2 Compares (14–19 unsigned, 120–123 signed)

All produce `1` or `0` in a full-width integer vreg; there is no `bool` at
the IR level.

| # | Name | | # | Name |
|---|---|---|---|---|
| 14 | `IR_CMP_EQ` | | 120 | `IR_SCMP_LT` |
| 15 | `IR_CMP_NE` | | 121 | `IR_SCMP_LE` |
| 16–19 | `IR_CMP_LT`/`LE`/`GT`/`GE` (unsigned) | | 122–123 | `IR_SCMP_GT`/`GE` |

Bare `<`, `<=`, `>`, `>=` at the surface are **type-directed**: unsigned
operands take 14–19, an operand carrying the signed flag takes 120–123. *(An
earlier revision of this file claimed bare comparisons are always unsigned
and that the signed opcodes appear only via explicit builtins. That is
wrong.)*

### 5.3 Floating point (20–27, 97–108, 118, 125)

| # | Name | Semantics |
|---|---|---|
| 20–23 | `IR_FADD`/`FSUB`/`FMUL`/`FDIV` | IEEE-754 binary64 |
| 24–25 | `IR_FCMP_EQ`/`IR_FCMP_LT` | ordered (false if either is NaN) |
| 26–27 | `IR_ITOF`/`IR_FTOI` | int64 ↔ f64; `FTOI` truncates |
| 97 | `IR_FSQRT` | `sqrtsd` / `fsqrt d` |
| 98 | `IR_FFMA` | `dest = src1 * src2 + imm_vreg` — **not fused**, see below |
| 99–102 | `IR_FCMP_NE`/`LE`/`GT`/`GE` | ordered compares |
| 103–104 | `IR_F32TOF64`/`IR_F64TOF32` | `cvtss2sd` / `cvtsd2ss` |
| 105–106 | `IR_F32TOF16`/`IR_F16TOF32` | f16 bit-pattern conversion |
| 107–108 | `IR_ITOF32`/`IR_FTOI32` | int64 ↔ f32 |
| 118 | `IR_FSQRT32` | f32 square root |

- **`IR_FFMA` is a misnomer.** It emits `mulsd` + `addsd` on x86_64 and
  `FMUL` + `FADD` on arm64 — two roundings, not one, so results differ from a
  true FMA. Its value is skipping the GPR round-trip between the multiply and
  the add, not accuracy. Verified by disassembly on KernRift (identical
  emitter, no `vfmadd` in the output); the op-98 handler here is at
  `src/ir.mlr:10658` and accumulates in xmm0 with the same `mulsd`/`addsd`
  pair. **UNVERIFIED by MLRift disassembly specifically.**
- **`IR_FTOI` saturates** on overflow and yields 0 for NaN, identically on
  both arches. Verified on KernRift; MLRift's op-27 handler is the same code.
- **105/106 are not x86-only** (an old claim): arm64 implements both as
  software bit manipulation.
- **Float literals are materialised at runtime.** `1.5` lowers to an
  `itof`/`itof`/`fdiv` chain, and const-fold does not fold float ops, so the
  `cvtsi2sd`/`divsd` survive to codegen.

`fkind` picks the register class: `1` → `xmm`/`d`, `2` → `xmm`/`s`, `0` → GPR.

### 5.4 Memory (30–32, 70–78, 84, 88, 94–95, 149–150)

| # | Name | Semantics | Effect |
|---|---|---|---|
| 30 | `IR_LOAD` | `dest = *(src1)` at width `imm` ∈ {1,2,4,8}, zero-extended | pure |
| 31 | `IR_STORE` | `*(src1) = src2` at width `imm`; truncates | **SE** |
| 32 | `IR_STACK_ADDR` | `dest = sp + imm` | pure |
| 70 | `IR_ALLOC` | `dest = alloc(size)`; per-OS, §8 | **SE** |
| 71 | `IR_DEALLOC` | free `src1` | **SE** |
| 72 | `IR_MEMCPY` | `memcpy(src1, src2, imm_vreg)` | **SE** |
| 73 | `IR_STRLEN` | `dest = strlen(src1)` | pure |
| 74 | `IR_FMT_UINT` | writes digits into `src1`; `dest` = length | **SE** |
| 75 | `IR_STR_EQ` | `dest = (strcmp(src1,src2) == 0)` | pure |
| 76 | `IR_MEMSET` | `memset(src1, src2, imm_vreg)` | **SE** |
| 77 | `IR_STATIC_LOAD` | `dest = static_data[imm]` | pure |
| 78 | `IR_STATIC_STORE` | `static_data[imm] = src1` | **SE** |
| 84 | `IR_STATIC_ADDR` | `dest = &static_data[imm]` | pure |
| 88 | `IR_MEMCMP` | `dest = (memcmp(src1,src2,imm) == 0)` | pure |
| 94 | `IR_VSTORE` | volatile store, fence after | **SE** |
| 95 | `IR_VLOAD` | volatile load, fence before | **SE** |
| **149** | **`IR_LOAD_BIS`** | `dest = *(src1 + src2 << log2(imm))`; `imm` = width 4\|8 — **KernRift calls this 147** | pure |
| **150** | **`IR_STORE_BIS`** | `*(src1 + imm << log2(dest)) = src2` — **KernRift calls this 148** | **SE** |

- `IR_LOAD` zero-extends: `load8` of `0xFF` yields `0x00000000000000FF`.
- **`IR_STORE_BIS` has an unusual operand shape** — `dest` holds a *width*,
  not a vreg, and `imm` holds a *vreg*, not a constant. It must be listed
  alongside `IR_STORE` in every "no def" list and alongside `IR_FFMA` in
  every "imm is a vreg" list for liveness, use-counting, DCE and interference
  (`src/ir.mlr:6172`, `12084`, `12259`, `4983`). Get this wrong and you get a
  use-after-free of a register.
- **149/150 are arm64-only**, produced solely by `ir_opt_fuse_lea_mem`
  (`src/ir.mlr:12683`), which runs under `target_arch == 1`. Handlers only at
  `src/ir_aarch64.mlr:1391` and `1429`.

### 5.5 Control flow (40–43, 50–52, 61, 85–87)

| # | Name | Semantics | Effect |
|---|---|---|---|
| 40 | `IR_BR` | jump to block `imm` | **SE** |
| 41 | `IR_BR_COND` | if `src1 != 0` → block `imm`; **else → the current block's `succ0`** | **SE** |
| 42–43 | `IR_RET` / `IR_RET_VOID` | | **SE** |
| 50 | `IR_CALL` | call the fn named by token `imm`; `src1` = arg count | **SE** |
| 51 | `IR_ARG` | stage `src1` as argument `imm` | **SE** |
| 52 | `IR_SYSCALL` | syscall; `imm` = the **Linux syscall number** | **SE** |
| 61 | `IR_COPY` | `dest = src1` | pure |
| 85 | `IR_SYSCALL_RAW` | `dest = syscall(src1 = nr vreg)` | **SE** |
| 86 | `IR_FN_ADDR` | `dest = &fn[imm_tok]` | pure |
| 87 | `IR_CALL_IND` | `dest = (*src1)()` | **SE** |

- **`IR_BR_COND` does not "fall through".** The false target is the block's
  `succ0` field, read separately. The emitter then picks one of four layout
  cases against the next block index. If you write a pass that reorders or
  renumbers blocks, you must maintain `succ0`/`succ1`, not just the
  terminator's `imm`.
- Terminators are **not** emitted from the per-instruction dispatch on any
  backend; they are handled in each backend's function-level block loop.
- **`IR_CALL` on x86 carries a GPU-specific hook.** `src/ir.mlr:9409` injects
  a `hipkfd_teardown()` call before a `@dynamic extern exit`, because the
  extern `exit` lowers to `IR_CALL` rather than `IR_SYSCALL` and would
  otherwise skip the syscall-site injection. Gated two ways (token text is
  exactly `exit`, **and** the name resolves to a `@dynamic` symbol) so a
  user-defined `exit` is not hijacked. Nothing like this exists upstream.

### 5.6 Multi-value returns (80–83)

| # | Name | Semantics | Effect |
|---|---|---|---|
| 80 | `IR_EXTRACT_RDX` | `dest = rdx` / `x1` after a call | **SE** |
| 81 | `IR_EXTRACT_R8` | `dest = r8` / `x2` after a call | **SE** |
| 82–83 | `IR_RET2` / `IR_RET3` | multi-value return | **SE** |

80/81 read a physical register that only holds the right value immediately
after the call. They are side-effectful specifically so DCE cannot delete
them and nothing can be scheduled between them and their call.

### 5.7 Strings, atomics, barriers, formatting, process

| # | Name | Semantics | Effect |
|---|---|---|---|
| 79 | `IR_STR_CONST` | `dest = &str_buf[imm]` | pure |
| 90–91 | `IR_ATOMIC_STORE` / `IR_ATOMIC_LOAD` | | **SE** |
| 92 | `IR_ATOMIC_ADD` | returns **old** value | **SE** |
| 93 | `IR_ATOMIC_CAS` | expect `src2`, store `imm_vreg`; returns 1/0 | **SE** |
| 96 | `IR_ASM_BLOCK` | inline-asm passthrough; `imm` = AST node index | **SE** |
| 109–112 | `IR_ATOMIC_SUB`/`AND`/`OR`/`XOR` | return **old** | **SE** |
| 113–114 | `IR_EXEC` / `IR_EXEC_ARGV` | | **SE** |
| 115 | `IR_SET_EXEC` | chmod +x | **SE** |
| 119 | `IR_TIME_NS` | monotonic nanosecond counter | **SE** |
| 124 | `IR_FMT_BOOL` | write `"true"`/`"false"` into `src1`; `dest` = length | **SE** |
| 125 | `IR_FMT_F64` | write a decimal into `src1` from `src2` | **SE** |
| 126 | `IR_ISB` | instruction-sync barrier | **SE** |
| 127 | `IR_DCACHE_FLUSH` | D-cache clean+invalidate by VA | **SE** |
| 128 | `IR_ICACHE_INV` | I-cache invalidate by VA | **SE** |
| 129–130 | `IR_DSB` / `IR_DMB` | barriers | **SE** |
| 131 | `IR_ARR_CHECK` | under `--debug`, trap if `src1 >= imm` | **SE** |

Atomics are sequentially consistent: `lock`-prefixed RMW on x86_64,
LDAXR/STLXR retry loops on arm64.

`IR_VSTORE`/`IR_VLOAD` are intended for MMIO. `DSB SY` on arm64 is a
data-sync barrier only — it does **not** flush the instruction cache. If you
write code to RAM and then call it, issue `ISB` explicitly.

**KernRift has one opcode here that MLRift does not:** `IR_MODULE_PATH`
(KernRift 146), the Windows `GetModuleFileNameA` path lookup. Do not port
its number — 146 is `IR_GPU_SYNC` here.

### 5.8 GPU host opcodes (140–147) — declared, never used

`src/ir_hip.mlr` (43 lines) declares:

| # | Name | Intended semantics |
|---|---|---|
| 140 | `IR_GPU_ALLOC` | `hipMalloc` |
| 141 | `IR_GPU_FREE` | `hipFree` |
| 142 | `IR_GPU_H2D` | host → device memcpy |
| 143 | `IR_GPU_D2H` | device → host memcpy |
| 144 | `IR_GPU_D2D` | device → device memcpy |
| 145 | `IR_KERNEL_LAUNCH` | variadic launch, reusing the `IR_CALL`/`IR_ARG` convention |
| 146 | `IR_GPU_SYNC` | `hipDeviceSynchronize` |
| 147 | `IR_GPU_BARRIER` | `__syncthreads()` |

**Nothing produces or consumes any of them.** Verified: the eight
identifiers appear nowhere outside `ir_hip.mlr` except in comments; the file
contains no `fn` at all; no emitter dispatches on 140–147; and the three real
GPU emitters (`format_hip.mlr`, `format_amdgpu.mlr`,
`format_amdgpu_megakernel.mlr`) walk the **AST**, not the IR — zero
`ir_insn_*` / `ir_fn_*` references between them, and `hip_emit_expr(node)`
(`src/format_hip.mlr:171`) takes AST nodes.

The file **is** in the build (`Makefile:14`, concatenated by the
`cat $(SRCS)` step), so it compiles to eight dead constants.

**Its only live effect is squatting on 140–147**, which pushed `IR_SHL_IMM`,
`IR_LOAD_BIS` and `IR_STORE_BIS` to 148/149/150 and created the entire
opcode-numbering divergence with KernRift — for opcodes nothing emits.

Two stale statements in that file, worth not believing:

- `src/ir_hip.mlr:8`–`11` claims "the 132-139 gap above is intentional" and
  that CPU ops end at 131. **132–139 are fully occupied** (`IR_SDIV` through
  `IR_LEA_BIS`, `src/ir.mlr:140`–`168`). There is no gap.
- `src/ir_hip.mlr:3` calls `format_hip` "forthcoming". It shipped — 844
  lines — and shipped without the IR.

See [`docs/GPU_BACKEND.md`](GPU_BACKEND.md) for the design these opcodes were
meant to serve; note that document self-labels as a **historical kickoff
document** and its proposed IR-op table describes a pipeline that was never
built. The live GPU path is documented in
[`docs/AMDGPU_NATIVE.md`](AMDGPU_NATIVE.md) and
[`docs/KFD_GOTCHAS.md`](KFD_GOTCHAS.md).

---

## 6. Backend coverage matrix

Four emitters, one dispatch function each:

| Arch | Dispatch fn | File:line | NYI helper |
|---|---|---|---|
| x86_64 | `ir_emit_x86_insn` | `src/ir.mlr:7605` | `x86_nyi_op`, `src/ir.mlr:7594` |
| AArch64 | `ir_emit_arm64_insn` | `src/ir_aarch64.mlr:881` | `a64_nyi_op`, `src/ir_aarch64.mlr:870` |
| RV32IMC | `ir_emit_riscv_insn` | `src/ir_riscv.mlr:1128` | `rv_nyi_op`, `src/ir_riscv.mlr:174` |
| Xtensa LX6 | `ir_emit_xtensa_insn` | `src/ir_xtensa.mlr:1828` | `xt_nyi_op`, `src/ir_xtensa.mlr:214` |

All four NYI helpers write `error: <arch>: IR op <N> not yet implemented\n`
to fd 2 and `exit(1)`. **There is never a silent fallback.** RISC-V has a
second, earlier rejection: `rv_op_is_float` (`src/ir_riscv.mlr:192`) covers
20–27, 97–108, 118 and 125, and `rv_float_trap` (`src/ir_riscv.mlr:185`)
prints `error: float not supported on riscv32 (no hardware FPU)`.

**Opcodes 1–123 are handled identically by every backend in both repos.** No
emitter implements the whole set; the old claim that "x86_64 and AArch64
implement the whole set" is false in both directions.

`✓` = handled, `—` = falls to NYI, `F` = rejected by the riscv float
backstop, `H` = handled only when `freestanding == 0`, `fn` = handled in the
function-level loop.

| Op | Name | x86 | a64 | rv32 | xtensa |
|---|---|:--:|:--:|:--:|:--:|
| 1–11 | const, add…shr | ✓ | ✓ | ✓ | ✓ |
| 12–13 | neg, not | ✓ | ✓ | — | — |
| 14–19 | unsigned compares | ✓ | ✓ | ✓ | ✓ |
| 20–27 | f64 arith / cmp / conv | ✓ | ✓ | F | — |
| 30–32 | load, store, stack_addr | ✓ | ✓ | ✓ | ✓ |
| 40–41 | br, br_cond | fn | fn | fn | fn |
| 42–43 | ret, ret_void | ✓ | ✓ | ✓ | ✓ |
| 50–52 | call, arg, syscall | ✓ | ✓ | ✓ | ✓ |
| 60 | **phi** | — | — | — | — |
| 61 | copy | ✓ | ✓ | ✓ | ✓ |
| 70–71 | alloc, dealloc | ✓ | ✓ | H | — |
| 72–79 | memcpy…str_const | ✓ | ✓ | ✓ | ✓ |
| 80–83 | extract/ret2/ret3 | ✓ | ✓ | — | — |
| 84 | static_addr | ✓ | ✓ | ✓ | ✓ |
| 85 | syscall_raw | ✓ | ✓ | H | — |
| 86–88 | fn_addr, call_ind, memcmp | ✓ | ✓ | ✓ | ✓ |
| 90–92 | atomic store/load/add | ✓ | ✓ | — | — |
| 93 | atomic_cas | ✓ | ✓ | — | **✓** |
| 94–96 | vstore, vload, asm_block | ✓ | ✓ | ✓ | ✓ |
| 97–108, 118 | float builtins | ✓ | ✓ | F | — |
| 109–112 | atomic sub/and/or/xor | ✓ | ✓ | — | — |
| 113–115 | exec, exec_argv, set_exec | ✓ | ✓ | — | — |
| 119 | time_ns | ✓ | ✓ | — | — |
| 120–123 | signed compares | ✓ | ✓ | ✓ | ✓ |
| 124 | fmt_bool | ✓ | ✓ | — | — |
| 125 | fmt_f64 | ✓ | ✓ | F | — |
| 126–131 | barriers, cache, arr_check | ✓ | ✓ | — | — |
| 132–134 | sdiv, smod, sar | ✓ | ✓ | ✓ | ✓ |
| 135–136 | add_imm, sub_imm | ✓ | **—** | ✓ | ✓ |
| 137 | ror | ✓ | ✓ | ✓ | — |
| 138 | mul_imm | ✓ | — | — | — |
| 139 | lea_bis | ✓ | ✓ | — | — |
| 140–147 | GPU host ops | — | — | — | — |
| 148 | shl_imm | ✓ | ✓ | — | — |
| 149–150 | load_bis, store_bis | — | ✓ | — | — |

Consequences worth stating plainly:

- **`IR_NEG` (12) and `IR_NOT` (13) are unimplemented on riscv32 and
  xtensa**, and both *are* emitted by ordinary lowering, so `-x` and `~x` on
  a non-constant operand are hard build failures on those targets.
- **`IR_ALLOC` (70) is hosted-only on riscv32 and absent on xtensa.** Since
  structs and local arrays over the stack threshold lower to `IR_ALLOC`, they
  are unavailable on freestanding riscv32 and on xtensa. Same shape for 71
  and 85.
- **Xtensa has `IR_ATOMIC_CAS` but none of the other atomics** — it has
  `S32C1I` and nothing else.
- **`IR_ROR` is unimplemented on xtensa in both repos** — `x ROR y` on xtensa
  aborts.
- **`IR_SHL_IMM` is x86/arm64 only here.** KernRift's 143 has handlers on all
  four backends; MLRift's 148 has two, because riscv32/xtensa have no
  producer and their NYI guards must stay loud.

---

## 7. Opcodes that are many-to-one with the surface language

### `IR_CALL` (50) carries a token index, resolved by text

`imm` on an `IR_CALL` is the index of a token in the *source text*. At fixup
time, `fn_lookup()` walks the function table comparing with `tok_text_eq()`,
which compares byte by byte through the compiler-wide `cg_source`.

**Consequence: a compiler-synthesised call to a name that has no token in the
program's source is impossible.** There is no "make me a symbol" path. Any
lowering that wants to call something must find a real token for it.

### Extern relocations, and the one place the IR does them

`--emit=obj` is routed to the legacy backend precisely because the x86_64 and
arm64 IR paths record only internal `fixup_table` entries and cannot emit an
extern relocation. riscv32 has no legacy backend, so its `--emit=obj`
necessarily goes through the IR; **UNVERIFIED for MLRift** whether its
`src/ir_riscv.mlr` carries the `R_RISCV_CALL_PLT` path that KernRift added at
`KernRift/src/ir_riscv.kr:1584` — it was not checked, and the four
`*_fixups_ensure` helpers MLRift adds to that file suggest the two have
drifted.

> **Divergence, MLRift-only, and it is significant.** MLRift's IR **can**
> emit dynamic-linking calls, which KernRift cannot at all.
> `dyn_sym_registry.mlr` + `format_elf_dyn.mlr` are wired into the x86 IR
> emitter: `dyn_sym_lookup(imm) != 0xFFFFFFFF` decides whether an `IR_CALL`
> becomes a PLT call (`src/ir.mlr:9423`), `dyn_call_record(call_patch, idx)`
> registers the relocation (`src/ir.mlr:9425`), `dyn_sym_abi_at()` selects
> the libm calling convention for a call (`src/ir.mlr:3024`), and
> `dyn_sym_count_get() > 0` switches the whole output to the dynamic-ELF
> path (`src/ir.mlr:13115`). The registry is token-keyed and populated by the
> parser on `@dynamic extern fn`. `grep -rn 'dyn_sym_\|edyn_' ../KernRift/src`
> returns nothing.

### `IR_SYSCALL` (52) carries a syscall number, not a builtin

`imm` is the **canonical Linux syscall number**, not an internal kind. Each
emitter feeds it through an arch/OS remapper, and branches on specific Linux
values (`imm == 2` triggers the `open`→`openat` argument shift on arm64 and
riscv32).

The mapping from builtins to numbers is **many-to-one**. Verified by reading
the `ir_emit(IR_SYSCALL, …)` call sites in `src/ir.mlr`: nr 1 (`write`) is
produced at 1875, 2301, 2318, 2331, 2485 and more; nr 0 (`read`) at 2349,
2499; nr 231 (`exit_group`) at 1860. So `write()`, `print()`, `println()`,
`print_str()`, `println_str()` and `file_write()` all become an
indistinguishable `IR_SYSCALL imm=1`.

**The emitter cannot recover which builtin it is serving.** Today that costs
only diagnostic precision. It became a correctness problem upstream, where a
`--target=none` refusal had to name the construct the programmer wrote and
could only say "syscall" six different ways.

> **Divergence.** KernRift solved this with a parallel `ir_insn_origin`
> table and a **single constructor**, `ir_emit_syscall()`, which is the only
> function in the tree allowed to create an `IR_SYSCALL` — so a future
> lowering cannot add an anonymous one. **MLRift has neither**: there is no
> `ir_insn_origin`, no `ir_emit_syscall`, and 12+ direct
> `ir_emit(IR_SYSCALL, …)` sites. If MLRift ever needs to attribute a syscall
> back to its builtin, that is the change to port, and the single-constructor
> discipline is the part that makes it hold.

`IR_SYSCALL_RAW` (85) is a *different* opcode and carries no origin in either
repo.

---

## 8. Per-target divergence, opcode by opcode

### 8.1 Per-OS

The opcodes whose behaviour depends on `target_os`:

| Op | What differs |
|---|---|
| 52 `IR_SYSCALL` | Windows → IAT thunks; macOS numbers get `0x2000000` OR'd; arm64 `open`→`openat` shift on Linux/Android but not macOS |
| 70 `IR_ALLOC` | Windows → `VirtualAlloc(NULL, size+8, 0x3000, 4)` through the IAT; elsewhere inline `mmap` with flags `0x22` (Linux/Android) vs `0x1002` (macOS) |
| 71 `IR_DEALLOC` | `VirtualFree(ptr-8, 0, MEM_RELEASE)` vs `munmap` |
| 85 `IR_SYSCALL_RAW` | arm64 syscall-number register is **x16 on macOS, x8 elsewhere** |
| 113/114 `IR_EXEC*` | `CreateProcessA` + `WaitForSingleObject` + `ExitProcess` vs `execve` |
| 115 `IR_SET_EXEC` | **emits nothing on Windows**; `fchmodat` on Linux/Android, `chmod` nr 15 on macOS |
| 119 `IR_TIME_NS` | QPC/QPF on Windows, `gettimeofday` on macOS, `clock_gettime(CLOCK_MONOTONIC)` on Linux/Android |
| 131 `IR_ARR_CHECK` | trap sequence differs per OS; **on arm64 + Windows the check is skipped entirely** |

The `IR_ALLOC` header convention is worth knowing because it affects
`IR_DEALLOC` and any hand-written allocator interop: the emitted `mmap`
requests `size + 8`, stores the size at the base, and returns `base + 8`.
`IR_DEALLOC` reads the size back from `ptr - 8`.

**Per-OS lowering** (as opposed to emission) is rare. `ir_lower_stmt`
(`src/ir.mlr:3562`–`4558`) has **zero** `target_os` references. In
`ir_lower_expr` the notable cases are `file_open`'s flags constant (Linux/
Android `0x241` vs macOS `0x601`, baked into an `IR_CONST`), and
`get_target_os` / `get_arch_id` folding a crosstable into an `IR_CONST`.

### 8.2 Per-arch, beyond instruction encoding

| Op | Divergence |
|---|---|
| 126 `IR_ISB` | real `ISB` on arm64; **complete no-op on x86_64**, zero bytes emitted |
| 128 `IR_ICACHE_INV` | `IC IVAU; DSB ISH; ISB` on arm64; **no-op on x86_64** |
| 129 / 130 `IR_DSB` / `IR_DMB` | distinct instructions on arm64; **both collapse to the same `MFENCE`** on x86 — two IR ops, one instruction |
| 127 `IR_DCACHE_FLUSH` | `CLFLUSH; MFENCE` on x86 vs `DC CIVAC; DSB ISH; ISB` to Point of Coherency on arm64 — different reach; the arm64 form needs EL1+ on real hardware |
| 5/6 `IR_DIV`/`IR_MOD` | x86 faults on divide-by-zero, arm64 returns 0 (§5.1) |
| 135/136 | handled on x86 and produced for x86/riscv32/xtensa; **arm64 has no handler and no producer** (§5.1) |

---

## 9. Side effects, purity, and what survives DCE

There are **two different predicates** and conflating them is a trap.

**`ir_opt_is_side_effect(op)`** — `src/ir.mlr:11868`. Used by DCE: an
instruction whose `dest` is dead is **NOP'd out** (its opcode set to 0)
*unless* this returns 1. Nothing is ever removed from the arena or the block
lists, and no block is ever deleted. The complete set at HEAD:

```
31 STORE            150 STORE_BIS       40 BR            41 BR_COND
42 RET              43 RET_VOID         50 CALL          51 ARG
52 SYSCALL          70 ALLOC            71 DEALLOC       72 MEMCPY
74 FMT_UINT         76 MEMSET          124 FMT_BOOL     125 FMT_F64
78 STATIC_STORE     80 EXTRACT_RDX      81 EXTRACT_R8    82 RET2
83 RET3             85 SYSCALL_RAW      87 CALL_IND
90..96              (atomics, vstore, vload, asm_block)
109..115            (atomic RMW, exec, exec_argv, set_exec)
119 TIME_NS        126 ISB             127 DCACHE_FLUSH 128 ICACHE_INV
129 DSB            130 DMB             131 ARR_CHECK
```

The old text's version of this list omitted `FMT_UINT`, `FMT_BOOL`,
`FMT_F64`, `STORE_BIS`, and the whole barrier/cache/`ARR_CHECK` group, and
mislabelled "exec/set_exec (109–115)" — 109–112 are the atomic RMW ops.

The function is byte-identical to KernRift's **except** for the `STORE_BIS`
line: `op == 150` here, `op == 148` there. That single number is the clearest
illustration of why an opcode literal must never be copied between the repos.

**`ir_opt_cse_is_pure(op)`** — `src/ir.mlr:12077`. Used by CSE. This is a
*whitelist of arithmetic*, not the complement of the side-effect set; see §10
for the full three-way split. `IR_LOAD` (30) is in neither set: DCE-able when
dead, but not CSE-pure, because there is no alias analysis.

### What needs a DCE seed

AST-level DCE seeds only `main` and `@export`. Anything reachable **only**
through a compiler-internal rewrite — not through a `Call` node in the AST —
is invisible to it and gets pruned. Upstream hit this twice (a freestanding
`_start`, and `@builtin_override` providers reached only via override
resolution). If you add a lowering that synthesises a call, seed its target
explicitly or it will be pruned before codegen sees it.

---

## 10. Optimizer

The driver is `ir_optimize()` (`src/ir.mlr:12440`). **There is no
`ir_opt_run()`** — that name appears only in older revisions of this
document. It runs once per function, after lowering and before liveness.

`--O0` sets `ir_opt_level = 0` (`src/ir.mlr:11319`, written only at
`src/main.mlr:6788`) and each backend skips the whole call
(`src/ir.mlr:13032`, `src/ir_aarch64.mlr:3218`, `src/ir_riscv.mlr:2330`,
`src/ir_xtensa.mlr:2891`). Note the spelling: **`--O0`**, two dashes.
*(Earlier revisions of this file said `-O0`. There is no such flag, and no
`-O1`/`-O2` either — no other write to `ir_opt_level` exists.)*

**What `--O0` does *not* disable**, because these live outside
`ir_optimize()`: liveness, use counts, wide-colour-file selection, colour
ceilings, **copy coalescing**, the XMM/FPR allocators, and the emit-time
spill peepholes. `--O0` is not "straight from lowering to regalloc".

Exact order at HEAD — **identical to KernRift's**:

1. `ir_opt_const_fold()` — `src/ir.mlr:11351`
2. `ir_opt_dce()` — `src/ir.mlr:11922`
3. `ir_opt_cse()` — `src/ir.mlr:12095`
4. `ir_opt_dce()`
5. `ir_opt_licm()` — `src/ir.mlr:12424`, which **loops up to 8×** over
   `ir_opt_licm_one_pass()` (`src/ir.mlr:12329`), stopping at fixpoint
6. `ir_opt_dce()`
7. `ir_opt_recognize_rotate()` — `src/ir.mlr:12913`
8. `ir_opt_const_fold()` + `ir_opt_dce()`
9. `ir_opt_recognize_lea_bis()` — `src/ir.mlr:12558`
10. `ir_opt_dce()`
11. **arm64 only** (`target_arch == 1`): `ir_compute_use_count()` →
    `ir_opt_fuse_lea_mem()` (`src/ir.mlr:12683`) → `ir_opt_dce()`
12. `ir_compute_use_count()` → `ir_opt_recognize_ffma()`
    (`src/ir.mlr:12509`) → `ir_opt_dce()`

There is no outer fixpoint loop over the whole pipeline, and no `-O` level
selects a subset. The only arch gate is step 11.

> **Divergence, compile-time only.** KernRift makes every cleanup DCE
> conditional on the preceding pass having rewritten something, using an
> `ir_opt_rewrites` counter (19 uses in `ir.kr`, **0** in `ir.mlr`). MLRift
> runs all six unconditionally. KernRift also bounds the per-function lattice
> memset to `min(ir_vreg_next, 65536)`; MLRift memsets the full 65536 twice
> (`src/ir.mlr:12444`). No codegen difference either way.

`ir_licm_enabled` (`src/ir.mlr:12422`) gates step 5 but is **never assigned
anywhere** — a dead knob with no CLI flag. Same upstream.

### Per-pass contracts

**`ir_opt_const_fold` (11351)** — linear walk, but **flow-insensitive**: the
constant map is wiped at every block boundary, so it is effectively
intra-block. It does far more than fold two `IR_CONST`s: binary fold for
2–11 and 14–19, unary `NEG`/`NOT`, identity and zero rules, `AND(x,
0xFFFFFFFF)` → `COPY` when `x` is w32-clean, pow2 `MUL` → `SHL_IMM`, and the
immediate-fusion families. It **propagates through `IR_COPY`**, so operands
need not be literal `IR_CONST`s.
*Breaking invariant:* the w32-clean lattice is sound only for single-def
vregs, and a single-def gate guards every set — because merges redefine
snapshot vregs via `COPY` (§1). Remove that gate and live `& 0xFFFFFFFF`
masks get silently elided.

**Branch folding is inline here**: an `IR_BR_COND` with a constant condition
becomes an `IR_BR`. **CFG successors are deliberately left as lowered** —
liveness over-approximates, and **nothing ever removes the untaken block.**

**`ir_opt_dce` (11922)** — a **backward** sweep over the flat array iterated
to fixpoint, then a forward NOP-out sweep. It crosses blocks. Terminators
survive via `ir_opt_is_side_effect`.
*Breaking invariant:* the operand-shape exclusion list (which opcodes really
read `src2`, and the imm-is-a-vreg set `{72, 76, 83, 93, 98, 150}`) is
**duplicated in several other places** — `ir_graph_color` (`src/ir.mlr:6161`,
`6172`), `ir_seed_wide_ceilings_generic` (`5728`), `ir_compute_use_count`
(`5357`), `ir_lea_mem_reads`. Any divergence between the copies miscompiles.
Note KernRift's copies carry an extra `&& (op < 140 || op > 145)` clause to
exclude its riscv immediate ops; MLRift's do not, correctly, because 140–145
are not CPU ops here.

**`ir_opt_cse` (12095)** — strictly **intra-block**. Hashes
`{op, canon(src1), canon(src2), canon(imm)}` into a direct-mapped table; a
hit rewrites the instruction to `IR_COPY <earlier>`. Any op in
`ir_opt_cse_invalidates` (`src/ir.mlr:12063`) bumps a generation stamp,
wiping the table in O(1).

**`ir_opt_licm` (12424)** — crosses blocks. Loop metadata comes **only from
the `while` lowering**. A hoist requires the dest to be **single-def**. The
hoistability predicate is *stricter* than CSE purity: it excludes `IR_CONST`
(rematerialising is cheaper) and `IR_STATIC_LOAD`.
*Breaking invariant:* hoisted instructions get **high array indices but low
block positions**, so anything assuming index order matches execution order
breaks.

**`ir_opt_recognize_ffma` (12509)** — matches `FADD` whose `src1` or `src2`
is defined by an `FMUL` with **exactly one use**. *Breaking invariant:* gated
to f64 on both. `IR_FFMA`'s emitter is f64-only and has no float-kind slot;
firing on f32 would run `mulsd`/`addsd` over f32 bit patterns and silently
produce garbage.

**`ir_opt_recognize_lea_bis` (12558)** — `ADD(base, MUL_IMM(idx, K))` or
`ADD(base, SHL_IMM(idx, k))` → `IR_LEA_BIS`. **Does not check use counts.**

**`ir_opt_fuse_lea_mem` (12683)** — arm64 only, **intra-block**. Commits only
if every use of the LEA's dest was matched, with a non-saturated use count,
which is why step 11 recomputes counts first.

**`ir_opt_recognize_rotate` (12913)** — matches the canonical 32-bit rotation
idiom (seeing past CSE's copy chains) and rewrites the `OR` in place to
`IR_ROR`. The dead chain is left for the DCE at step 8.

### `ir_opt_cse_is_pure` vs `ir_opt_is_side_effect` — the trap, precisely

They are **not complements**, and there is a large gap between them.

- **CSE-pure** (`src/ir.mlr:12077`): 1–19, 20–27, 77, 79, 84, 86, 97–108,
  120–123, **and 148 only** of the immediate family.
- **Side-effecting**: the list in §9.
- **Neither** — DCE-able when dead, but *never* CSE'd or hoisted:
  `IR_LOAD` 30, `IR_STACK_ADDR` 32, `IR_COPY` 61, `IR_STRLEN` 73,
  `IR_STR_EQ` 75, `IR_MEMCMP` 88, 105, 106, 118, `IR_SDIV` 132,
  `IR_SMOD` 133, `IR_SAR` 134, `IR_ADD_IMM` 135, `IR_SUB_IMM` 136,
  `IR_ROR` 137, `IR_MUL_IMM` 138, `IR_LEA_BIS` 139, `IR_LOAD_BIS` 149.

Two consequences worth writing down:

1. **`IR_STATIC_LOAD` (77) is CSE-pure but is a memory read.** Sound only
   because a `STORE`/`CALL` bumps the CSE generation. LICM has no such
   protection and excludes 77 explicitly.
2. **Strength reduction destroys CSE and LICM opportunities.** const-fold
   runs *before* CSE and rewrites `ADD`→`ADD_IMM`, `MUL`→`MUL_IMM`. Those
   products are mathematically pure but not on the CSE whitelist, so
   expressions that would have been collapsed or hoisted as `ADD`/`MUL` no
   longer are. It was only fixed for the `SHL_IMM` case.

Also latent: `IR_STORE_BIS` (150) is in `ir_opt_is_side_effect` but **not** in
`ir_opt_cse_invalidates`. Not exploitable today — 150 is created at step 11,
strictly after the only `ir_opt_cse()` call — but it becomes a live bug the
moment CSE is moved or re-run. Identical latent bug upstream.

---

## 11. Register allocator

`ir_graph_color()` (`src/ir.mlr:6022`) is shared by all four backends through
a single colour→physical-register table.

**It is not Chaitin-style.** There is no simplify/select stack and no
spill-and-rebuild loop. One pass: pre-set every vreg to `0xFFFFFFFF`
(spilled); order representatives by **saturating use count, descending**;
for each, take the lowest colour not used by an interfering neighbour and
below the rep's colour ceiling; if none is free the vreg simply **stays
spilled** — there is no retry. Rep colours then propagate to coalescing
followers.

### Colour files, verified from the init functions

The old text's "x86_64: 5 colors → rbx, r12, r13, r14, r15" is wrong on the
count, the set, **and** on which file is the default.

| Arch | Narrow | Registers | Wide (**the default**) | Extra registers |
|---|---|---|---|---|
| x86_64 | 6 (`src/ir.mlr:5453`, init 5542) | rbx, r12, r13, r14, r15, **rbp** | **12** (`src/ir.mlr:5580`, init 5582) | + rsi, rdi, r8–r11 (all caller-saved) |
| arm64 | 10 (`src/ir_aarch64.mlr:150`) | x19–x28 | **23** (`src/ir_aarch64.mlr:194`, init 202) | colours 10–18 → x0–x8; 19–22 → x12–x15 |
| riscv32 | 12 (`src/ir_riscv.mlr:148`) | — | (none) | |
| xtensa | `ir_regalloc_init_xtensa` (`src/ir_xtensa.mlr:527`) | — | `ir_regalloc_init_xtensa_wide` (`src/ir_xtensa.mlr:584`) | |

**Wide is what runs.** Narrow is the fallback. The narrow file is
callee-saved only, so a colour survives a call for free. The wide file
appends caller-saved registers **after** the callee-saved prefix, so the
prologue/epilogue (which look only at colours below the prefix length) need
no change and "must survive a call" becomes simply "ceiling = prefix length".

Deliberate exclusions: on x86, **rax/rcx/rdx** — rax/rcx are the universal
spill-reload scratch pair, rcx is the variable-shift count (CL), rdx is
`div`/`mod`'s implicit high half and `IR_FFMA`'s third-operand scratch. On
arm64, **x9/x10** (scratch pair), **x11** (`IR_MOD` quotient temp and
large-frame address scratch), **x16/x17** (linker veneer scratch; x16 is also
the ADRP target of the static/str-const fixup sequences and macOS's
syscall-number register) and **x18** (platform-reserved).

**Wide-mode gate — exactly two disqualifiers.** The function contains an
`IR_ASM_BLOCK` (op 96) anywhere in its body, or it is `@naked`.
`ir_x86_fn_wide_ok` (`src/ir.mlr:5671`) / `ir_a64_fn_wide_ok`
(`src/ir_aarch64.mlr:246`) test only the former; the call sites add
`&& is_naked == 0`. The asm check must stay a **whole-body scan, not a
constraint-list scan** — the comment at `src/ir.mlr:5646` records the real
case: `std/vec_f64_dispatch`'s CPUID block writes r13 with no constraint
naming it. `@naked` is excluded because it gets `frame_size = 0`, so a
ceiling-forced spill would store outside any frame.

### Colour ceilings

This is the mechanism that makes the wide file correct without auditing every
op handler, and it is the part of the allocator most likely to be
misunderstood.

The wide file hands out **caller-saved** registers. A value that must survive
a `CALL`, a syscall, a helper loop or a Windows IAT thunk cannot live in one.
Rather than teach the colourer about physical registers, each vreg gets a
per-vreg cap: *you may only take colours below N*. Setting N to the
callee-saved prefix length (6 on x86, 10 on arm64) confines the value to
registers the ABI preserves.

`ir_seed_wide_ceilings_generic` (`src/ir.mlr:5728`) seeds these at
**instruction** granularity, not block granularity: sha256's whole 64-round
compression body is one block containing two small calls, so block-level
capping surrendered every register in exactly the code the feature exists
for. It walks each block backward and, at every "dirty" (non-whitelisted)
instruction, caps everything live after it, its own dest, and everything live
before it. Temporaries living entirely between two dirty points keep the full
file.

Whitelists: `ir_x86_op_widesafe` (`src/ir.mlr:5622`) and `ir_a64_op_widesafe`
(`src/ir_aarch64.mlr:226`) — "safe" means the handler touches no physical
register outside the scratch set beyond its allocator-assigned operands. The
arm64 list additionally excludes 77/78/79/84/86, because arm64's
static-data / str-const / fn-addr sequences materialise through **X16 *and*
X0**. A missing whitelist entry is a lost optimisation, never a miscompile.
*(These two functions are identical to KernRift's except for the trailing
opcode numbers — `148` here vs `143` there, and `148/149/150` vs
`143/147/148` on arm64.)*

**Parameter ceilings** solve a different, sharper problem.
`ir_a64_seed_param_ceilings` (`src/ir_aarch64.mlr:275`) and
`ir_x86_seed_param_ceilings` (`src/ir.mlr:5856`) cap every register-passed
parameter to the callee-saved prefix. The leading param copies read x0–x7 /
rdi,rsi,rdx,rcx,r8,r9 **one at a time in ascending param order**; if param
0's home were x3, the copy for param 3 would read a register param 0 already
clobbered.

Ceilings must also be pushed onto coalescing **representatives** before
colouring: without that, a capped vreg coalesced with an uncapped one would
inherit a colour its own ceiling forbids — a value a call is about to destroy
quietly landing in a caller-saved register.

### Copy coalescing

Lives **inside** `ir_graph_color` (`src/ir.mlr:6190` onward), after the
interference graph is built and before the greedy loop. On by default;
`ir_coalesce_enabled` (`src/ir.mlr:5481`, setter `ir_set_coalesce` at 5483)
is cleared by `--no-coalesce`. **`--O0` does not disable it.**

The **lower vreg is always kept** as the representative — a correctness
requirement, not a preference: otherwise an intermediate vreg colours against
a rep that is still uncoloured and can collide.

- **Briggs** (`src/ir.mlr:6263`): count neighbours of the merged class whose
  rep's cached degree is `>= IR_NUM_REGS`; accept if that count is below the
  effective K.
- **K is per-merge**: the tightest of the two vregs' colour ceilings. Note the
  asymmetry — the *significant-degree* test uses the global `IR_NUM_REGS`, the
  *accept threshold* uses the ceiling.
- **George** (`src/ir.mlr:6310`) is tried only when Briggs fails, and is
  skipped when K is small. KernRift's comment carries the measurement that
  set the threshold: arm64 (K=10) gains ~800 B, x86 (K=6) loses ~900 B.

Interference-freedom is necessary but **not sufficient** — Briggs/George gate
it as well.

### Spilling

**Slots are vreg-indexed, one per vreg**, not a compacted set of spilled
values (`ir_spill_offset`, `src/ir.mlr:7244`; `ir_a64_spill_offset`,
`src/ir_aarch64.mlr:312`). The older `vreg - IR_NUM_REGS - 1` mapping was
removed because it assumed a low-numbered vreg could never spill, but a
colour ceiling can spill **any** vreg, and the subtraction underflowed.

- **x86:** `[rsp + (vreg-1)*8]`; the load/store helpers pick disp8 below 128
  else disp32. No offset limit, no fallback needed.
- **arm64:** `IR_A64_OVERFLOW_RESERVE + (vreg-1)*8`
  (`src/ir_aarch64.mlr:307`, default 128). Offsets up to 32760 (= 4095 × 8)
  use the imm12-scaled form; beyond that the helpers fall back to
  `MOV x11, imm; ADD x11, sp, x11; LDR/STR [x11]`. The scratch is
  specifically **x11**.
- **Emit-time peepholes**, *not* part of `ir_optimize()` and *not* disabled by
  `--O0`: a `store_spill r,v` immediately followed by `load_spill r,v` with no
  bytes emitted between is elided entirely; the same vreg into a *different*
  register becomes a reg-reg `mov`.

### Floating point — two allocators, two different designs

**Both** arches have a dedicated float allocator; the old claim that x86
handles floats inside the integer allocation is wrong.

**arm64** — `ir_a64_fpr_alloc` (`src/ir_aarch64.mlr:388`), run after
`ir_graph_color`. It **reuses the same interference graph and union-find** but
colours an independent 8-entry file `d8..d15` over representatives whose fkind
is 1 (f64). Order is **highest vreg first**, so loop-carried values born deep
in the function beat early constants to the homes. Because `d8..d15` are
AAPCS **callee-saved**, no call-liveness restriction is needed at all.

**x86** — `ir_x86_xmm_alloc` (`src/ir.mlr:7109`), homes `xmm2..xmm15`
(`xmm0`/`xmm1` are the float scratch pair and xmm0 is the SysV float return
register). It is called **unconditionally** with an `enable` argument, so a
stale map from the previous function can never leak.

The decisive difference: **SysV x86-64 has no callee-saved XMM registers at
all** — a call may destroy the entire XMM file. So x86 needs a bar arm64 does
not: `ir_x86_xmm_mark_unsafe()` (`src/ir.mlr:6980`) disqualifies, at
instruction granularity, every vreg live across a non-widesafe op. A second
bar is coalition purity: a rep whose coalition contains any non-f64 member is
barred, because a mixed coalition shares one storage location.

Enforcement on x86 is **funnel-based, not per-handler**: a homed vreg's GPR
colour is forcibly cleared to "spilled", so all ~70 op handlers read and write
it through `ir_resolve_src` / `ir_emit_load_spill` / `ir_emit_store_spill`,
and those funnels redirect to a `movq` against the xmm home instead of a stack
slot. The one emitter that reads slots directly — the >6-argument overflow
path in `IR_CALL` — carries an explicit home check at the site. If you add an
emitter that touches a spill slot without going through the funnels, you must
add the same check.

### Divergence

`ir_graph_color` is 545 lines here against 568 upstream; the algorithm,
colour files, ceilings and coalescing are the same. Three implementation
differences:

- **Instruction walk.** MLRift copies each block's linked list into a
  65536-entry `ir_walk_stack` then unwinds it. KernRift indexes flat per-block
  lists built by `ir_build_bb_lists` (`ir_bb_list_buf`: 8 uses in `ir.kr`, 0
  in `ir.mlr`). KernRift is ahead here.
- **Count-trailing-zeros.** MLRift has `ir_ctz64` (`src/ir.mlr:5495`, used at
  6097/6275/6313); KernRift open-codes `ir_popcount64(iso - 1)` at each site.
  MLRift is ahead here.
- **Builtin-name dispatch in the lowering.** MLRift has a
  (first char × name length) 128×32 prefilter (`ir_bi_filter_init`,
  `src/ir.mlr:916`, 81 uses); KernRift is linear. MLRift is ahead here.

---

## 12. Bare metal — not supported

KernRift has a `--target=none` / `target_os == 4` bare-metal target: a
lowering layer (`ir_bm_*`), 39 `target_os == 4` sites, `@builtin_override`
providers for `write` and `alloc`, and four per-architecture trap-instruction
choke points that refuse to emit `SYSCALL` / `SVC` / `ECALL` / `SIMCALL`.

**MLRift has none of it.** Verified: zero `target_os == 4` occurrences in the
entire tree, and `--target=none` is rejected by argument parsing
(`mlrc: unknown --target= value (expected: linux, macos/darwin, windows/win,
android, esp32, hip-amd, amdgpu-native)`).

If that capability is ever wanted here, the two structural lessons from
upstream are worth taking before the code:

1. **Guard the choke point, not N sites — and verify N.** A design brief
   there asserted "`emit_a64_svc` is a single function behind 49 call sites";
   seven sites emitted the raw `SVC` word directly. Guarding the named
   function would have shipped seven live traps.
2. **The refusal must live in `codegen.mlr`, not `ir.mlr`.** The legacy
   backend has its own builtin dispatch and never calls `ir_lower_expr`, so an
   `ir.mlr`-only guard leaves `--legacy` and `--emit=obj` unguarded *by
   construction*. That is exactly how a silently-wrong `time_ns` (a constant
   zero, exit 0, artifact written) survived a full audit round upstream.

---

## 13. Known traps and defects

Current at HEAD unless stated. The old "Known miscompiles R1/R2" section has
been removed: R1 (the arm64 `compile_fat` miscompile forcing `--legacy` on
shipped arm64 binaries) and R2 (IR output 10–34 % larger than legacy) were
both closed over a year ago by the partial callee-save prologue, the AST
inliner, and Briggs/George coalescing. Do not reinstate them without
re-measuring.

**Documented behaviour that surprises people**

1. **`--emit=obj` is the legacy backend.** §2. An `--emit=obj` test proves
   nothing about IR lowering.
2. **`--emit=ir` is pre-optimization**, and `ir_opcode_name` prints `???`
   for opcodes 124, 125 and 135–148.
3. **`--target=` picks the ABI, not the container.** §2.
4. **The IR is not SSA.** §1.
5. **`IR_FFMA` is not fused.** §5.3.
6. **arm64 division by zero silently returns 0** without `--debug`. §5.1.
7. **The GPU opcodes are dead** and cost the entire numbering divergence.
   §5.8.

**Live defects and gaps**

8. **arm64 loses `ADD_IMM`/`SUB_IMM`.** The const-folder excludes arch 1, so
   `n - 1` and `i + 4` stay as `mov const; add` pairs on arm64 instead of a
   single fused instruction, and `src/ir_aarch64.mlr` has no handler for
   135/136. KernRift fuses both. This is a straightforward port: add arch 1 to
   the gate at `src/ir.mlr:11683` **and** the handler
   (`KernRift/src/ir_aarch64.kr:1257` is the reference), never one without the
   other.
9. **cmp-with-immediate fusion is x86-only** (`src/ir.mlr:11590`). KernRift
   covers arm64 too, with an imm12 (4095) cap.
10. **No arch × OS validation.** §3. `--arch=riscv32 --target=windows` is
    accepted.
11. **`IR_ARR_CHECK` is skipped entirely on arm64 + Windows** — deliberate
    (the `ExitProcess`-via-IAT sequence does not fit the short-jump pattern)
    but it means `--debug` bounds checking does not exist on that pair.
12. **`IR_STORE_BIS` (150) is in `ir_opt_is_side_effect` but not in
    `ir_opt_cse_invalidates`.** Not exploitable today; becomes a live bug the
    moment CSE moves or is re-run. §10.
13. **Duplicated operand-shape tables.** The "which opcodes really read
    `src2`" exclusion list and the imm-is-a-vreg set are written out in five
    separate places (§10). Any divergence between them miscompiles. Adding an
    opcode with an unusual operand shape means editing all five.
14. **`ir_compute_liveness_OLD` (`src/ir.mlr:5196`) is dead** — its only
    reference is its own definition.

**Stale artifacts found while writing this document**

15. `src/ir_hip.mlr:8`–`11` — "the 132-139 gap above is intentional". 132–139
    are dense. §5.8.
16. `src/ir_hip.mlr:3` — calls `format_hip` "forthcoming". It shipped, and
    shipped without the IR.
17. `src/format_hip.mlr:9`–`16` — claims the kernel body is a placeholder
    ("TODO") pending a follow-up. `hip_emit_kernel` (`:612`) emits full
    bodies: it walks the body Block's statements through `hip_emit_stmt` and
    injects an 8 KiB `__shared__` LDS pool. The same header (`:17`) points at
    `ir_hip.kr` — wrong extension, and the wrong mechanism (§5.8).
18. [`docs/GPU_BACKEND.md`](GPU_BACKEND.md) `:105`–`114` — presents the IR
    opcode table as the mechanism. Nothing produces or consumes those
    opcodes. The header disclaims the document as historical, but the table
    is a trap for anyone landing mid-page.
19. `.gitignore:3` — `!/build/krc2` should read `!/build/mlrc`. The path it
    un-ignores does not exist in this repo; `build/mlrc` survives only because
    it was already in the index, and gitignore does not apply to tracked
    files.

**Process lessons, inherited from upstream and worth keeping**

20. **A terminal `else` that returns a value is invisible to a trap scan.**
    Three shipped upstream: an arm64 `set_executable` that emitted *zero
    instructions* (chmod never happened, returned 0 = success), and two
    `time_ns` paths returning constant 0 while exiting 0 and writing an
    artifact. Only a byte-identity or behaviour gate catches this class.
21. **A dropped assertion is a finding you decided not to have.** The
    `time_ns` zero was seen as "a naming quirk" and the assertion was removed
    instead of chased. The symptom was the bug.
22. **Print the whole output before concluding the compiler did not diagnose
    something.** A scratch harness piping through `head -4` cut off a
    diagnostic that lands on line 6 and produced a false claim that was then
    propagated into a permanent test comment.
23. **Reverting an injected byte-identity regression poisons the next
    bootstrap generation** — the injected compiler emits the bad lowering into
    its successor's own code. Recovery is to restore the tracked seed
    (`build/mlrc`) and delete the generated concatenation.

---

## 14. GPU and SIMD — where they actually live

**Neither is in the IR.**

- **GPU.** The eight opcodes in `src/ir_hip.mlr` are declarations with no
  producer or consumer (§5.8). The working emitters — `format_hip.mlr`,
  `format_amdgpu.mlr`, `format_amdgpu_megakernel.mlr` — walk the **AST**
  directly. See [`docs/AMDGPU_NATIVE.md`](AMDGPU_NATIVE.md),
  [`docs/KFD_GOTCHAS.md`](KFD_GOTCHAS.md),
  [`docs/MEGAKERNEL_DESIGN.md`](MEGAKERNEL_DESIGN.md), and
  [`docs/GPU_BACKEND.md`](GPU_BACKEND.md) (historical).
- **SIMD.** There is **no IR-level SIMD in either repo** — no `IR_VEC_*` or
  `IR_SIMD_*` opcode, no VEX/AVX2 encoder in `codegen.mlr` or `ir.mlr`.
  [`docs/SIMD_CODEGEN.md`](SIMD_CODEGEN.md) is self-labelled
  *design-approved, implementation not started*, and its Approach A would add
  hand-coded builtins in `codegen.mlr` rather than IR opcodes.

  CPU-side vector work today goes through **inline asm** (`IR_ASM_BLOCK`,
  op 96). That is exactly why `ir_x86_fn_wide_ok` / `ir_a64_fn_wide_ok`
  disqualify any function containing op 96 from the wide colour file (§11) —
  an asm block can name any physical register, and `std/vec_f64_dispatch`'s
  CPUID block really does write r13 without a constraint saying so.

The one place the GPU work reaches into the IR is the auto-teardown hook in
the x86 `IR_CALL` handler (§5.5), which injects `hipkfd_teardown()` before a
`@dynamic extern exit`.

---

## 15. Divergence from KernRift — summary

`../KernRift` (`src/*.kr`) is upstream. **113 opcode names are shared and 110
of them have identical numbers**; ops 0–139 are identical everywhere, and all
eight backend dispatches agree on opcodes 1–123.

### Opcode numbers 140–150 mean different things

MLRift reserves 140–147 for the GPU host ops, so its later opcodes are
shifted:

| Number | MLRift | KernRift |
|---|---|---|
| 140 | `IR_GPU_ALLOC` | `IR_AND_IMM` |
| 141 | `IR_GPU_FREE` | `IR_OR_IMM` |
| 142 | `IR_GPU_H2D` | `IR_XOR_IMM` |
| **143** | **`IR_GPU_D2H`** | **`IR_SHL_IMM`** |
| 144 | `IR_GPU_D2D` | `IR_SHR_IMM` |
| 145 | `IR_KERNEL_LAUNCH` | `IR_SAR_IMM` |
| 146 | `IR_GPU_SYNC` | `IR_MODULE_PATH` |
| **147** | **`IR_GPU_BARRIER`** | **`IR_LOAD_BIS`** |
| **148** | **`IR_SHL_IMM`** | **`IR_STORE_BIS`** |
| 149 | `IR_LOAD_BIS` | *(unassigned)* |
| 150 | `IR_STORE_BIS` | *(unassigned)* |

**A numeric opcode literal copied between the two repos is a miscompile.**
`ir_opt_is_side_effect` is the clearest case: MLRift lists `op == 150`
(`STORE_BIS`), KernRift lists `op == 148` — same intent, different number.
Swapping them would mark `IR_SHL_IMM` side-effectful in one repo and leave a
store DCE-eligible in the other.

### Opcodes present in only one dialect

- **MLRift only**: the eight GPU host ops (`src/ir_hip.mlr`) — declared,
  never emitted.
- **KernRift only**: `IR_AND_IMM`/`IR_OR_IMM`/`IR_XOR_IMM`/`IR_SHR_IMM`/
  `IR_SAR_IMM` (a riscv32 immediate-fusion family MLRift's const-folder never
  produces) and `IR_MODULE_PATH` (Windows `GetModuleFileNameA`).

### Real capability differences, in both directions

| | MLRift | KernRift |
|---|---|---|
| Dynamic linking from the IR | **yes** — `dyn_sym_registry.mlr` + `format_elf_dyn.mlr`, PLT calls decided inside the x86 IR emitter (§7) | **no** — no PLT/GOT/`DT_NEEDED` path at all |
| GPU targets | `--target=hip-amd`, `--target=amdgpu-native`, `--emit-amdgpu-*` (AST-based) | none |
| Bare metal (`--target=none`) | **no** — zero `target_os == 4` sites | yes — full `ir_bm_*` layer, four trap choke points, provider routing |
| Syscall→builtin attribution | **no** `ir_insn_origin`, no single constructor, 12+ direct `ir_emit(IR_SYSCALL, …)` sites | `ir_insn_origin` table + `ir_emit_syscall()` as the sole constructor |
| `--emit=lkm` (Linux kernel module) | no; IR gate is `emit_mode != 3` | yes; gate is `emit_mode != 3 && emit_mode != 7` |
| arch × OS allow-list | **absent** | `arch_os_pair_supported()` |
| arm64 `ADD_IMM`/`SUB_IMM` | **neither producer nor handler** | both |
| cmp-with-immediate fusion | arch 0 only | arch 0 and 1 |
| pow2 `MUL` → `SHL_IMM` | present, arch 0/1 | present, ungated with a shamt guard |
| riscv32 immediate-fused logicals | absent | present (opcodes 140–145) |
| var map | linear scan | FNV-1a open-addressed hash |
| CTZ | `ir_ctz64` | open-coded popcount |
| builtin-name dispatch | 128×32 prefilter | linear |
| per-block instruction lists for colouring | 65536-entry walk stack | flat lists (`ir_build_bb_lists`) |
| conditional cleanup DCE | all six run unconditionally | gated on an `ir_opt_rewrites` delta |

The optimizer **pass list and order are identical**. The register allocator is
identical, including the wide colour files, the colour ceilings, and Briggs +
George. Both use **`--O0`** (two dashes).

The living compiler (`src/living.mlr`) **does not touch the IR** — zero
`ir_*` calls; it works on tokens and the AST. It is not an MLRift
development: it was inherited wholesale at the `.kr → .mlr` rename and has
not been touched since, while KernRift's copy has had four further commits.
Nothing there to cross-reference from an IR document.

### Build seed

**`build/mlrc` is a tracked git seed** — `git ls-files build` returns it.
KernRift's `build/krc2` is gitignored and returns nothing. Do not "fix"
either to match the other; and see defect 19 about the stale `.gitignore`
negation.

---

## 16. Adding a new opcode

1. **Pick a number that is free in *both* MLRift and KernRift**, or accept
   that the two dialects will disagree about what it means. Adding a CPU
   opcode below 140 is safe; anything at or above 140 is not. Add
   `static uint64 IR_FOO = N` to `src/ir.mlr`.
2. Add the name to `ir_opcode_name()` (`src/ir.mlr:4559`). It is currently
   incomplete; do not add to the gap.
3. Add the lowering in `ir_lower_expr()` or `ir_lower_stmt()`, **or** the
   optimizer pass that synthesises it. If the op is created by the optimizer
   only, it will never appear in `--emit=ir`.
4. Add the emission branch to every backend that must support it, and check
   the others fail *loudly*. **A producer without a handler is the shape of
   the arm64 `ADD_IMM` gap** — add both or neither, in the same change.
5. If it has a side effect, add it to `ir_opt_is_side_effect()`
   (`src/ir.mlr:11868`). If it is safe to CSE, add it to
   `ir_opt_cse_is_pure()` (`src/ir.mlr:12077`). If it invalidates memory
   state, add it to `ir_opt_cse_invalidates()` (`src/ir.mlr:12063`). Those are
   three separate decisions.
6. If any operand slot is used unconventionally — `imm` holding a vreg,
   `dest` holding a width — audit **all five** liveness / use-count / DCE /
   interference / ceiling lists for it (§10, defect 13). `IR_STORE_BIS` and
   `IR_FFMA` are the existing examples to copy.
7. If the op needs per-OS behaviour, remember that the legacy backend has its
   own dispatch and never sees your lowering (§2, §3).
8. Add a test that exercises a mode which actually uses the IR backend —
   **not `--emit=obj`**.
9. Update this reference, and re-verify the counts in §3 and §6 rather than
   editing them by hand.
