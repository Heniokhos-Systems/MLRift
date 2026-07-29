# Changelog

## 1.1.0

- First release of the MLRift VS Code extension.
- Syntax highlighting for `.mlr` files: keywords, types (`u8`..`u64`,
  `i8`..`i64`, `f16`/`f32`/`f64`), built-in functions, annotations
  (`@export`, `@noreturn`, `@naked`, `@packed`, `@section`, and the kernel/PCI
  annotation set), `device` MMIO blocks, struct/enum/method syntax, and the
  `#lang` directive.
- Language server providing live diagnostics (via `mlrc check`), completions,
  hover docs, and go-to-definition for keywords, built-ins, types, functions,
  structs, and enums.
- Adapted from the KernRift VS Code extension; MLRift's grammar omits the
  `let` keyword, which KernRift's lexer has but MLRift's does not.
