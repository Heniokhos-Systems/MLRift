# MLRift Language Support

Editor support for the [MLRift](https://github.com/Heniokhos-Systems/MLRift)
systems language — syntax highlighting plus a language server that drives
live diagnostics and IntelliSense from the real compiler.

## Features

- **Syntax highlighting** for `.mlr` files, with a file icon in the explorer
  and tabs.
- **Live diagnostics** — errors and warnings from `mlrc check`, mapped to the
  exact line and column as you type.
- **IntelliSense** — completions, hover docs, and go-to-definition for
  keywords, built-ins, types, functions, structs, and enums.

Highlighting covers:

- Integer types — short aliases `u8`/`u16`/`u32`/`u64`, `i8`/`i16`/`i32`/`i64`
  and long forms `uint8`..`int64`.
- Floating point — `f16`, `f32`, `f64`.
- Built-in functions — pointer ops (`load8..64`, `store8..64`), volatile ops
  (`vload/vstore8..64`), atomics (`atomic_load/store/cas/add/...`), bitfield ops
  (`bit_get/set/clear/range/insert`), signed compares (`signed_lt/gt/le/ge`),
  string output (`print_str`, `println_str`), and platform hooks
  (`get_target_os`, `get_arch_id`, `syscall_raw`, `exec_process`).
- Keywords including `match`, `loop`, `defer`, `for`, `while`, and the
  declaration set (`fn`, `struct`, `enum`, `static`, `const`, `device`, ...).
- `device` blocks for MMIO: `device NAME at ADDR { FIELD at OFF : TYPE rw }`.
- Static and struct arrays (`static u8[N] name`, `Point[10] pts`), slice
  parameters (`fn foo([u8] data)` with `data.len`), and method syntax
  (`fn Point.sum(Point self) -> u64`).
- Annotations (`@export`, `@noreturn`, `@naked`, `@packed`, `@section("name")`)
  and the `#lang` directive (`#lang stable` / `#lang experimental`).
- String/char literals with escapes, line (`//`) and block (`/* */`) comments,
  auto-closing brackets, indentation, and folding.

## Requirements

Syntax highlighting works on its own. **Diagnostics and hover docs need the
`mlrc` compiler installed** — the language server shells out to `mlrc check`.
If `mlrc` is not found, highlighting and completions still work, but no
errors are reported.

Build the compiler from the
[repository](https://github.com/Heniokhos-Systems/MLRift), and make sure
`mlrc` is on your `PATH` or set `mlrift.compilerPath` below.

## Extension Settings

| Setting | Default | Description |
| --- | --- | --- |
| `mlrift.compilerPath` | `mlrc` | Path to the `mlrc` compiler binary used for diagnostics. Set this if `mlrc` is not on your `PATH`. |

## About MLRift

MLRift is a self-hosted systems language for machine-learning and GPU
workloads. Forked from KernRift, it compiles itself ahead-of-time to native
machine code — no VM, no interpreter, no libc — and extends the base
language with ML-specific primitives (tensors, event streams, sparse ops)
and a native backend targeting AMD GPUs (HIP/AMDGPU) alongside embedded
MCUs (ESP32/Xtensa, RISC-V). One `.mlrbo` fat binary bundles multiple
platform slices; a small runner extracts and executes the matching slice
at startup. Self-host bootstrap fixed point is verified by CI on every push.

The compiler is written entirely in MLRift and includes an SSA IR backend,
graph-coloring register allocator, constant folding / DCE / CSE, and
per-target native code emitters.

- [GitHub](https://github.com/Heniokhos-Systems/MLRift)
