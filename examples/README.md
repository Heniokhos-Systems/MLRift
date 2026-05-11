# MLRift Examples

Each file in this directory is a standalone runnable MLRift program that
compiles with the current `mlrc` and exercises a specific language feature.

## Running

```sh
mlrc hello.mlr --arch=x86_64 -o hello
./hello
```

Or build a fat binary and run with the `mlr` runner:

```sh
mlrc hello.mlr -o hello.mlrbo
mlr hello.mlrbo
```

## What's in here

| File | Demonstrates |
|------|--------------|
| `hello.mlr` | The minimum program: `println` of a string literal. |
| `fib.mlr` | Recursive Fibonacci — recursion, `for..in`, integer arithmetic. |
| `fizzbuzz.mlr` | `for` loops, `if/else` chains, `println` with both literals and variables. |
| `pointers.mlr` | `load8/16/32/64` and `store8/16/32/64` builtins — the clean way to access memory. |
| `count_chars.mlr` | Byte-level string iteration via `load8`. |
| `slices.mlr` | `[T] name` slice parameters + `.len` — fat pointer pattern. |
| `struct_arrays.mlr` | `Point[10] pts` — fixed arrays of structs. |
| `mmio_driver.mlr` | `device` blocks — named typed MMIO registers with volatile semantics. |
| `echo.mlr` | `scan_str` / `print_str` for stdin / stdout with variable strings. |
| `extern_libc.mlr` | `extern fn` — call libc (`strlen`, `write`) via ELF/Mach-O/COFF relocations. |
| `linked_list.mlr` | Canonical heap-struct pattern — `Node n = alloc(16)`, append, traverse. |

## Notes

- All examples use the short type aliases (`u8`, `u16`, `u32`, `u64`) rather
  than the long forms (`uint8` etc.).
- `println(variable)` formats the variable as a decimal integer. For
  variables that hold string pointers, use `print_str` / `println_str`.
- Range loops use the exclusive form `0..n`. The inclusive form `0..=n`
  is not currently supported — use `0..n+1`.
