# Contributing to MLRift

MLRift is a systems language for machine learning and artificial biology, built
on top of MLRift. The compiler's own source is written in MLRift (`.mlr`
files in `src/`), reuses MLRift's backend, and extends its IR with
ML-specific primitives. MLRift user programs (when the frontend lands) will
use the `.mlr` extension.

## Prerequisites

The repo ships a committed `build/mlrc` as the bootstrap binary, so there is
no external toolchain required — a clean clone self-hosts on Linux x86_64.

## Build

```sh
make build       # build/mlrc compiles build/mlrc.mlr → build/mlrc (in place)
```

## Test

```sh
make test        # full suite
make bootstrap   # verify stage3 == stage4 (fixed point)
```

## Install

```sh
make install     # installs to ~/.local/bin/mlrc
```

## Source Structure

All compiler source is in `src/` (written in MLRift):

| File | Purpose |
|------|---------|
| `lexer.mlr` | Tokenizer |
| `parser.mlr` | Parser (recursive descent + Pratt) |
| `ast.mlr` | AST node definitions |
| `analysis.mlr` | Safety passes |
| `ir.mlr` | SSA IR + x86_64 codegen |
| `ir_aarch64.mlr` | AArch64 IR codegen |
| `codegen.mlr` | x86_64 legacy code generation |
| `codegen_aarch64.mlr` | AArch64 legacy code generation |
| `format_*.mlr` | Output formats (ELF, Mach-O, PE, AR, MLRBO) |
| `runtime.mlr`, `living.mlr`, `formatter.mlr` | Supporting infrastructure |
| `main.mlr` | CLI and compilation driver |

Standard library modules in `std/` are inherited from MLRift.

## Guidelines

- The compiler must always self-compile to a fixed point (`make bootstrap`)
- Run `make test` before submitting changes
- No external dependencies — the compiler is fully self-contained
- When porting bug fixes from upstream MLRift, reference the MLRift
  commit hash in the commit message so the lineage stays traceable
