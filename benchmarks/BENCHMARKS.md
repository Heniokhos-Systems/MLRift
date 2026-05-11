# MLRift Benchmarks — v1.0.0

**Release:** v1.0.0
**Run date:** 2026-05-11
**Compilers compared:** mlrc 1.0.0 (self-hosted), gcc 13.3.0, rustc 1.93.1

Four targets benchmarked side-by-side:

| Tag | Machine | CPU | OS / kernel | mlrc | gcc | rustc |
|---|---|---|---|:---:|:---:|:---:|
| **Linux x86_64** | Desktop | AMD Ryzen 9 7900X | Linux 6.17 | ✓ | ✓ | ✓ |
| **Linux aarch64** | Pi 400 | Broadcom BCM2711 (Cortex-A72 ×4 @ 1.8 GHz) | Linux 6.17.0-1014-raspi | ✓ | ✓ | ✓ |
| **Windows x86_64** | Laptop | Intel Core Ultra 9 275HX | Windows 11 Pro 10.0.26200 | ✓ | — | — |
| **Android aarch64** | Redmi Note 8 Pro | MediaTek Helio G90T (MT6785V/CC) | Android 11 / Linux 4.14 | ✓ | — | — |

Reproduce locally with `MLRC=build/mlrc bash benchmarks/run_benchmarks.sh`. The script auto-detects host arch and skips toolchains that aren't on PATH (Windows / Android machines run mlrc-only by design for this release).

---

## 1. Micro-benchmarks — single-file programs

Compile-then-run pipeline. Runtime is the median of 3 consecutive runs after a warmup. Source under `benchmarks/{fib,sort,sieve,matmul}.{mlr,c,rs}`.

### fib(40)

**Linux x86_64 (Ryzen 9 7900X)**

| Compiler | Compile (ms) | Binary (B) | Runtime (ms) |
|---|---:|---:|---:|
| mlrc (self-hosted) | 2 | 292 | 416 |
| gcc -O0 | 57 | 15 800 | 385 |
| gcc -O2 | 43 | 15 800 | 80 |
| rustc (debug) | 377 | 3 889 248 | 393 |
| rustc -O opt2 | 85 | 3 887 792 | 164 |

**Linux aarch64 (Pi 400)**

| Compiler | Compile (ms) | Binary (B) | Runtime (ms) |
|---|---:|---:|---:|
| mlrc (self-hosted) | 44 | 392 | 3 263 |
| gcc -O0 | 1 314 | 70 368 | 1 746 |
| gcc -O2 | 299 | 70 392 | 726 |
| rustc (debug) | 5 565 | 3 975 040 | 3 316 |
| rustc -O opt2 | 879 | 3 974 144 | 1 136 |

**Windows x86_64 (Core Ultra 9 275HX)**

| Compiler | Compile (ms) | Binary (B) | Runtime (ms) |
|---|---:|---:|---:|
| mlrc (self-hosted) | 2 573 | 2 048 | 641 |

**Android aarch64 (Redmi Note 8 Pro)**

| Compiler | Compile (ms) | Binary (B) | Runtime (ms) |
|---|---:|---:|---:|
| mlrc (self-hosted) | 57 | 131 072 | 1 787 |

### sort (quicksort, 200 k ints)

**Linux x86_64**

| Compiler | Compile (ms) | Binary (B) | Runtime (ms) |
|---|---:|---:|---:|
| mlrc | 2 | 467 | 79 |
| gcc -O0 | 32 | 15 960 | 154 |
| gcc -O2 | 29 | 15 960 | 275 |
| rustc debug | 123 | 3 905 344 | 2 649 |
| rustc -O opt2 | 93 | 3 888 048 | 44 |

**Linux aarch64 (Pi 400)**

| Compiler | Compile (ms) | Binary (B) | Runtime (ms) |
|---|---:|---:|---:|
| mlrc | 45 | 544 | 476 |
| gcc -O0 | 375 | 70 384 | 1 066 |
| gcc -O2 | 192 | 70 408 | 341 |
| rustc debug | 1 236 | 3 983 024 | 12 260 |
| rustc -O opt2 | 745 | 3 974 128 | 196 |

**Windows x86_64** — mlrc: 2 872 ms compile, 2 048 B binary, 128 ms runtime
**Android aarch64** — mlrc: 72 ms compile, 131 072 B binary, 566 ms runtime

### sieve (primes up to 10⁶)

**Linux x86_64**

| Compiler | Compile (ms) | Binary (B) | Runtime (ms) |
|---|---:|---:|---:|
| mlrc | 2 | 462 | 3 |
| gcc -O0 | 31 | 16 008 | 4 |
| gcc -O2 | 28 | 16 008 | 2 |
| rustc debug | 76 | 3 901 200 | 22 |
| rustc -O opt2 | 86 | 3 888 144 | 2 |

**Linux aarch64 (Pi 400)**

| Compiler | Compile (ms) | Binary (B) | Runtime (ms) |
|---|---:|---:|---:|
| mlrc | 38 | 576 | 37 |
| gcc -O0 | 208 | 70 432 | 46 |
| gcc -O2 | 233 | 70 432 | 33 |
| rustc debug | 612 | 3 980 512 | 100 |
| rustc -O opt2 | 678 | 3 974 128 | 31 |

**Windows x86_64** — mlrc: 564 ms compile, 2 048 B binary, 43 ms runtime
**Android aarch64** — mlrc: 78 ms compile, 131 072 B binary, 60 ms runtime

### matmul (256×256 int)

**Linux x86_64**

| Compiler | Compile (ms) | Binary (B) | Runtime (ms) |
|---|---:|---:|---:|
| mlrc | 2 | 1 106 | 25 |
| gcc -O0 | 30 | 15 960 | 16 |
| gcc -O2 | 30 | 15 960 | 4 |
| rustc debug | 77 | 3 900 272 | 128 |
| rustc -O opt2 | 85 | 3 888 488 | 3 |

**Linux aarch64 (Pi 400)**

| Compiler | Compile (ms) | Binary (B) | Runtime (ms) |
|---|---:|---:|---:|
| mlrc | 45 | 808 | 83 |
| gcc -O0 | 165 | 70 384 | 127 |
| gcc -O2 | 218 | 70 408 | 30 |
| rustc debug | 517 | 3 978 824 | 544 |
| rustc -O opt2 | 629 | 3 974 136 | 31 |

**Windows x86_64** — mlrc: 4 387 ms compile, 2 560 B binary, 66 ms runtime
**Android aarch64** — mlrc: 76 ms compile, 131 072 B binary, 176 ms runtime

### Takeaways

- **Compile speed.** On Linux, `mlrc` compiles 15–40× faster than `gcc` and 40–200× faster than `rustc`. On Pi the gap widens (slow disk + ARM `gcc` startup): 7–30× over `gcc`, 14–125× over `rustc`. Windows ssh-driven compile is much slower (process-spawn / antivirus overhead) but still produces 2 KB binaries.
- **Binary size.** `mlrc` outputs are 5–80× smaller than `gcc -O2` and ~3 000–10 000× smaller than `rustc` because no C / Rust runtime is linked. Android `.so`-style PIE binaries page-align to 128 KB, which dominates the absolute number there.
- **Runtime.** Competitive with `gcc -O0` across the board and beats `rustc debug` everywhere. `gcc -O2` / `rustc -O2` still win the optimizable cases — `mlrc`'s IR optimizer is currently constant-folding + CSE + DCE + basic regalloc, with no inliner / vectorizer / loop transforms. The `sort` case is interesting: `mlrc` beats `gcc -O2` on x86_64 (the libc qsort comparison-function call overhead pessimizes the C version).

---

## 2. Self-host — mlrc compiling itself

Source concatenated to a single ~1.7 MB file (≈215 k tokens, ≈134 k AST nodes), then fed to each configuration.

### Single-architecture compile, per-target (Linux host, Ryzen 9 7900X)

| Target | IR compile | IR binary | Legacy compile | Legacy binary |
|--------|-----------:|----------:|---------------:|--------------:|
| linux x86_64 ELF | 1 543 ms | 1 189 473 B | 246 ms | 1 184 375 B |
| linux arm64 ELF | 1 543 ms | 818 510 B | *(not run)* | *(not run)* |
| windows x86_64 PE | 1 547 ms | 1 247 732 B | *(not run)* | *(not run)* |
| windows arm64 PE | *(via fat slice)* | 880 640 B | — | — |
| macOS x86_64 Mach-O | *(via fat slice)* | 1 196 032 B | — | — |
| macOS arm64 Mach-O | *(via fat slice)* | 868 352 B | — | — |
| android x86_64 ELF | *(via fat slice)* | 1 310 720 B | — | — |
| android arm64 ELF | 1 546 ms | 917 504 B | — | — |

### Fat-binary self-compile (all 8 targets at once)

| Configuration | Time | Output size |
|---------------|-----:|------------:|
| Default (IR for all 8 slices) | 12 202 ms | 3 818 000 B (≈ 3.82 MB) |
| `--legacy` (all 8 slices legacy) | 1 935 ms | 4 086 000 B (≈ 4.09 MB) |

IR is the default for every slice. Legacy codegen is retained as an explicit opt-out behind `--legacy` and is ~6× faster but emits ~7% larger output on ARM64 / PE / Mach-O slices.

### Native-hardware self-compile (not cross-compiled on x86_64 host)

| Host | CPU | Single-arch IR | Fat binary (default IR) |
|------|-----|---------------:|------------------------:|
| Linux x86_64 | AMD Ryzen 9 7900X | 1 543 ms | 12 202 ms |
| Linux aarch64 (Pi 400) | Broadcom BCM2711 (Cortex-A72 ×4 @ 1.8 GHz) | ~9 800 ms | ~78 000 ms |
| Windows 11 x86_64 | Intel Core Ultra 9 275HX | 1 794 ms | 22 709 ms |
| Linux ARM64 (qemu) | Ryzen 9 7900X + qemu-aarch64-static | 20 100 ms | *(not benched)* |
| Android aarch64 | Redmi Note 8 Pro / MT6785V (Helio G90T, 6 GB) | 19 782 ms | 161 274 ms |

The Pi 400 native ARM64 numbers fall between Cortex-A76 mobile (Android) and Zen 4 desktop, as expected for a 4× A72 @ 1.8 GHz. Windows x86_64 is 1.16× Linux x86_64 for single-arch (essentially parity) but 1.86× for fat — the widening is the per-`alloc`/`print` cross-DLL IAT tax hitting 8× re-parses.

### Bootstrap fixed-point (stage 1 → stage 2 reproducibility)

| Stage | Time | md5 |
|-------|-----:|-----|
| Stage 1: `mlrc → stage1` | 1 545 ms | `2881d820…` |
| Stage 2: `stage1 → stage2` | 1 544 ms | `2881d820…` |

Binaries match byte-for-byte — the compiler reaches its own fixed point in two passes.

---

## 3. Compiler feature coverage (439 test suite)

```
=== Results: 439/439 passed, 0 failed ===
```

Under IR ARM64 via qemu: **432/439** pass. The 7 skips/fails are:
- `asm_hex` / `naked_fn` / `asm_rdtsc_out` / `asm_shl_in_out` — x86-only inline-assembly tests, correctly gated by `$ARCH != aarch64` on native ARM64 CI.
- `device_block_read_write` — uses an absolute mmap VA that qemu-user can't honor.
- `custom_fat_smaller` — exercises compile_fat with IR for all slices.
- `arm64 f16 conversions` — not implemented on ARM64.

---

## Reproducing

```bash
# Micro-benchmarks (auto-detects host arch + available toolchains)
MLRC=build/mlrc bash benchmarks/run_benchmarks.sh

# Override the platform label / output path:
PLATFORM="My machine" RESULTS=/tmp/my-bench.md bash benchmarks/run_benchmarks.sh

# Self-host timings / binary sizes (section 2)
make build                                                # produces build/mlrc.mlr + build/mlrc
./build/mlrc --arch=x86_64 build/mlrc.mlr -o /tmp/out     # IR single-arch
./build/mlrc --legacy --arch=x86_64 build/mlrc.mlr -o /tmp/out  # legacy single-arch
./build/mlrc build/mlrc.mlr -o /tmp/fat.mlrbo             # fat (all 8 slices)

# Fixed-point
./build/mlrc --arch=x86_64 build/mlrc.mlr -o /tmp/s1
chmod +x /tmp/s1
/tmp/s1 --arch=x86_64 build/mlrc.mlr -o /tmp/s2
md5sum /tmp/s1 /tmp/s2   # must match
```

`mlrc` self-reports wall time in `(X.XX ms)` for every `-o` invocation on every host including Windows (via `QueryPerformanceCounter` through the IAT). No external `time` / `Measure-Command` wrapper is needed.

### Remote-host harnesses used for v1.0.0

- **Pi 400 (SSH)** — full `run_benchmarks.sh` on the Pi after `scp`-ing `mlrc-arm64` + sources, `PATH` extended with `~/.cargo/bin` so `rustc` is picked up.
- **Windows 11 (SSH)** — `benchmarks/win_bench.ps1` (PowerShell, mlrc-only) run via `sshpass` + `ssh pante@<ip> powershell -ExecutionPolicy Bypass -File ...`.
- **Android (ADB)** — `benchmarks/android_bench.sh` (POSIX sh / mksh, mlrc-only) pushed to `/data/local/tmp/mlrift-bench/` and invoked via `adb shell`. Sources push as `.mlr`; mlrc emits with `--emit=android` PIE.

`win_bench.ps1` and `android_bench.sh` are committed alongside `run_benchmarks.sh` so the Linux-flavoured 3-way and the platform-only mlrc runs are both reproducible.
