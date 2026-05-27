# MLRift benchmark — Local / Ryzen 9 7900X

**Date:** 2026-05-11 04:26:40 UTC
**Host:** AMD Ryzen 9 7900X 12-Core Processor
**Kernel:** Linux 6.17.0-23-generic x86_64
**Toolchains:** mlrc=yes, gcc=yes, rustc=yes
**mlrc flags:** `--arch=x86_64`

## fib

| Compiler | Compile (ms) | Binary (B) | Runtime median-of-3 (ms) |
|---|---|---|---|
| mlrc (self-hosted) | 2 | 292 | 416 |
| gcc -O0 | 57 | 15800 | 385 |
| gcc -O2 | 43 | 15800 | 80 |
| rustc (debug) | 377 | 3889248 | 393 |
| rustc -O opt2 | 85 | 3887792 | 164 |

## sort

| Compiler | Compile (ms) | Binary (B) | Runtime median-of-3 (ms) |
|---|---|---|---|
| mlrc (self-hosted) | 2 | 467 | 79 |
| gcc -O0 | 32 | 15960 | 154 |
| gcc -O2 | 29 | 15960 | 275 |
| rustc (debug) | 123 | 3905344 | 2649 |
| rustc -O opt2 | 93 | 3888048 | 44 |

## sieve

| Compiler | Compile (ms) | Binary (B) | Runtime median-of-3 (ms) |
|---|---|---|---|
| mlrc (self-hosted) | 2 | 462 | 3 |
| gcc -O0 | 31 | 16008 | 4 |
| gcc -O2 | 28 | 16008 | 2 |
| rustc (debug) | 76 | 3901200 | 22 |
| rustc -O opt2 | 86 | 3888144 | 2 |

## matmul

| Compiler | Compile (ms) | Binary (B) | Runtime median-of-3 (ms) |
|---|---|---|---|
| mlrc (self-hosted) | 2 | 1106 | 25 |
| gcc -O0 | 30 | 15960 | 16 |
| gcc -O2 | 30 | 15960 | 4 |
| rustc (debug) | 77 | 3900272 | 128 |
| rustc -O opt2 | 85 | 3888488 | 3 |

