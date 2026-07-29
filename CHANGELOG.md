# Changelog

All notable changes to `mlrc` are documented in this file.

## v1.1.0 — 2026-07-29

The first release since v1.0.1 (May 2026), covering 342 commits. It brings MCU
targets, an int8 neural-network layer, a large codegen and register-allocation
overhaul, and several correctness fixes — two of which produce wrong results
rather than crashes.

> **Anyone who compiled for ARM64 with v1.0.x should upgrade and recompile.**
> Those releases silently miscompile 32-bit rotations on ARM64. Rotations are the
> core primitive of SHA-2, MD5, ChaCha, most PRNGs and many checksums, so the
> visible symptom is *wrong output*, not a crash. x86_64 was unaffected.

### Correctness

- **ARM64 `IR_ROR` miscompile fixed.** Unhandled opcodes now loud-fail instead of
  emitting nothing, which is how this shipped broken in the first place.
- **Saturating `f64` → integer conversion on x86_64**, matching ARM64 and
  WebAssembly. Out-of-range and NaN conversions previously diverged between
  architectures: x86 returned the `INT64_MIN` sentinel where ARM64 saturated, and
  NaN became `INT64_MIN` on x86 but `0` on ARM64. Both now saturate, NaN → 0.
- **A variable's signedness comes from its declaration, not its inferred value**,
  and signed return types are no longer lost at call sites (which selected
  `SHR`/`DIV`/`MOD` where `SAR`/`SDIV`/`SMOD` were required).
- **Method `self` is by-reference** — writes through `self` now persist. Struct
  arguments are passed by value for all lvalue forms, not only plain identifiers.
- **All 26 remaining raw ARM64 branch-displacement sites are range-checked.** An
  out-of-range branch now fails loudly instead of silently truncating into a
  wrong target.
- Register allocation: one spill slot per vreg (the previous mapping could
  underflow), and a coalesce is judged against the merged node's colour ceiling
  rather than the whole register file.
- Growable DCE, function, import, srcmap and fixup tables — several fixed-size
  caps could be overrun by large inputs. ELF image buffers are now sized from the
  computed layout instead of a fixed cap.
- Loud failure on struct field/size overflow and on unhandled x86_64 opcodes.
- **The standard library now resolves for installed users.** `mlrc` searched
  `/usr/local/share/kernrift/`, `/usr/share/kernrift/` and
  `$HOME/.local/share/kernrift/` — never renamed after the fork — while
  `install.sh` writes the stdlib to `$HOME/.local/share/mlrift/std`. No
  `import "std/..."` could resolve from an OS-level install on Linux or macOS.
- **An import that cannot be opened now aborts the build.** It previously
  printed an error but continued, so a file whose missing module happened not
  to break the parse produced a binary and exit 0 — a green build and a
  silently wrong artifact. This is what kept the stdlib path bug hidden.

### Performance

Register allocation and codegen were overhauled across every backend:

- **x86_64**: XMM register class (f64 values live in `xmm2`–`xmm15`) and a wide
  12-colour integer file (`rsi`/`rdi`/`r8`–`r11` join as colours 6–11).
- **ARM64**: f64 register class (`d8`–`d15`), a wide 23-colour integer file,
  a partial callee-save frame that saves only the colours actually used, CMP+B.cond
  fusion, and shifted-index addressing on loads and stores
  (`ldr Xd, [Xbase, Xidx, LSL #3]`).
- **Shared IR**: pow2 multiply strength reduction (`IR_SHL_IMM`), and loop
  back-edge copies are staged through temporaries only when they actually conflict.
- **Xtensa/RISC-V**: 9 register colours instead of 4 (leaf functions first, then
  functions that make calls), and ADD/SUB immediate fusion extended to both
  32-bit backends.

Measured on a Cortex-A72 (Raspberry Pi 400), best-of-9: `matmul` 83 → 56 ms,
`sha256` 959 → 947 ms.

Compiler throughput also improved substantially: **self-compile 2131 → 385 ms**
(5.5x), and fat-binary self-compile peak memory **1.16 GB → 86 MB** after fixing
leaked per-call and per-slice arenas.

### New targets

- **ESP32 / Xtensa LX6** via `--target=esp32`, emitting a bootable esp-image.
  `--model <file>` appends a blob as a DRAM segment at `0x3FFB0000`.
- **RISC-V 32** freestanding images.
- Taking the ESP32 off its 40 MHz crystal onto the PLL is worth **6.0x** on its own.

### Neural networks

- `std/nn_int8.mlr` — an int8 quantized kernel library.
- `std/nn_model.mlr` — a flat KRNN model container and layer walker.
- On real silicon (YuNet 96px, ESP32 @240 MHz), the accumulated register
  allocation, accessor and immediate-fusion work is **1.898x** end to end
  (4.583 s → 2.472 s), with all 3025 output values byte-identical to the
  original baseline at every stage.
- Host, Xtensa and RISC-V32 outputs are byte-identical to the Python reference.

### GPU

- AMDGPU/HIP host-side ops, kernel launch, and a range-checked SOPP branch encoder.
- Grouped-GEMM register-blocked (2x2) variant.

### Tooling and packaging

- A **VS Code extension** (`editors/vscode`): syntax highlighting, live
  diagnostics, and IntelliSense for `.mlr`.
- Packaging manifests and a release workflow — v1.0.1 shipped with no binary
  assets attached, so the documented `install.sh` one-liner could not work.
- Repository references corrected to `Heniokhos-Systems/MLRift`.
- `mlr --help` no longer claims "BCJ+LZ4-compressed, 7 platform slices"; the fat
  binary is BCJ+LZ-Rift-compressed with 8 slices.

---

## v1.0.1 — 2026-05-17

## v1.0.0 — 2026-05-11

Initial public releases. See the git history for details.
