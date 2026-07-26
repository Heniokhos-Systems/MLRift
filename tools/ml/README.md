# ONNX → MLRift int8 pipeline

Offline tooling that turns a float ONNX CNN into int8 weights + quantisation
parameters for `std/nn_int8.mlr`. Host-side Python; nothing here runs on the
target.

## Stages

| script | does |
|---|---|
| `onnx_to_mlrift.py` | parse graph, quantise weights per-channel symmetric (data-free), emit `.layers.json` / `.weights.bin` / `.qparams.json` |
| `float_ref.py` | NumPy walk of exactly the layer set `nn_int8.mlr` provides (+Resize/Sigmoid). Ground truth. **Validated against onnxruntime to 7.6e-07 on YuNet.** |
| `calibrate.py` | expose every intermediate as a graph output, accumulate per-tensor activation ranges over real images. `--pct P` switches from min/max to percentile-clipped (two-pass histogram) |
| `int8_sim.py` | int8 pipeline that is bit-faithful to `nn_int8.mlr` (srdhm + rounding shift), scored against `float_ref.py` |

## Setup

Needs `onnx`, `onnxruntime`, `numpy`, `pillow` — install into a venv; they are
NOT dependencies of MLRift itself.

## Notes that cost time to learn

- opencv_zoo ONNX files on `raw.githubusercontent.com` are **Git LFS pointers**
  (a 4 KB text stub). Fetch from `media.githubusercontent.com/media/...`.
- onnxruntime enforces the model's declared static input shape. To run YuNet at
  128×128, patch `graph.input[0]` dims and clear `value_info` + output dims.
  It is fully convolutional, so this is safe.
- `srdhm` DOUBLES (`a*b*2 >> 31`), so a multiplier of `m_norm * 2^31` needs a
  right shift of `s+1`, not `s`. Getting this wrong is exactly a factor of 2
  and it is not obvious in end-to-end output — test the primitive directly.

## Calibration: mild clipping only

Measured on YuNet @128px, 75 WIDER FACE images, worst per-tensor error vs the
float reference:

| calibration | worst error |
|---|---|
| min/max | 59.7% |
| **`--pct 99.99`** | **34.3%** |
| `--pct 99.9` | 73.8% |
| `--pct 99.5` | 97.3% |

Aggressive clipping makes things WORSE here — these activations do not have
long wasteful tails, so clipping past ~99.99% saturates real signal. Do not
assume tighter is better; measure.

## Judge a detector by detections, not per-tensor error

Per-tensor error is a misleading yardstick. What matters is whether the same
anchors win. On the same image, int8 (`--pct 99.99`) vs float:

  top-5 anchors   5/5 agree
  top-20 anchors  19/20 agree
  score correlation 0.976

So a "34% worst per-tensor error" still reproduces the float model's
detections almost exactly, because detection depends on the *ranking* of
anchor scores rather than their absolute values.

## Stage 4: on-target execution

| script | does |
|---|---|
| `krnn.py` | the **KRNN** flat container format — spec, packers, arena planner. The spec block is duplicated verbatim in `std/nn_model.mlr`; the runner reads by fixed offset and cannot notice drift |
| `onnx_to_mlrift.py --emit-binary` | NHWC shape inference, quantisation-parameter derivation, offline arena planning, packing |
| `krnn_to_mlr.py` | bake a `.krnn` into MLRift source (see the caveat below) |
| `gen_frame.py` | the deterministic validation frame, mirroring `nn_mt_rand()` in `tests/nn/nn_model_core.mlr` |
| `frame_from_image.py` | a real image → the quantised int8 NHWC frame the runner is handed |

`std/nn_model.mlr` walks the container and dispatches to `std/nn_int8.mlr`.
There is no JSON on target, no allocator, no strings and no shape inference:
every tensor's offset inside a single activation arena is decided here, by a
greedy-by-size planner over tensor lifetimes, and baked into the layer record.

### End to end on YuNet

```sh
# convert (needs the calibration ranges from stage 2)
python3 tools/ml/onnx_to_mlrift.py yunet_128.onnx -o /tmp/y \
    --emit-binary /tmp/yunet128.krnn --ranges ranges_p99.99.json

# the frame both sides will start from
python3 tools/ml/gen_frame.py 128 128 3 /tmp/frame.i8

# the oracle
python3 tools/ml/int8_sim.py yunet_128.onnx ranges_p99.99.json \
    --input-i8 /tmp/frame.i8 --dump /tmp/sim --no-float-ref
python3 - <<'EOF' > /tmp/expected.txt
import numpy as np
for n in ['cls_8','cls_16','cls_32','obj_8','obj_16','obj_32',
          'bbox_8','bbox_16','bbox_32','kps_8','kps_16','kps_32']:
    for v in np.fromfile(f'/tmp/sim/out_{n}.i8', dtype=np.int8):
        print(int(v))
EOF

# host, xtensa and riscv32, all compared against it
NN_KRNN_MODEL=/tmp/yunet128.krnn NN_KRNN_EXPECTED=/tmp/expected.txt \
    MLRC=./build/mlrc bash tests/nn/run_nn_tests.sh
```

Measured (YuNet @128, 106 nodes, 5376 output values): host x86_64, xtensa
under `qemu-system-xtensa -M lx60` and riscv32 under `qemu-system-riscv32
-M virt` are **byte-identical to `int8_sim.py` and to each other** — and so
are all 106 intermediate tensors, not just the outputs. Peak arena 131072 B.

### int8_sim.py is now bit-faithful, and was not before

Two bugs had to be fixed before "bit-exact" was even a testable claim:

- `srdhm` used an arithmetic right shift where gemmlowp (and `nn_ref.py`, and
  `nn_int8.mlr`) divide with **truncation toward zero**. Off by one on every
  negative product with a remainder.
- it carried a **doubled** product compensated by an extra right shift, i.e.
  the opposite shift-sign convention to TFLite and `nn_mul_by_qm`. That is a
  clean factor of 2 in every requantised value, and it is invisible in the
  float-error summary because a uniform halving of a per-channel multiplier
  looks like a slightly worse quantiser, not like a bug.
- `Add` was done in floating point. It is now TFLite's integer AddGeneral,
  which is what `nn_add_int8` implements.

Both files now use TFLite's convention throughout, so `--emit-binary` passes
`(multiplier, shift)` straight through.

### Baking a model into an image: don't, at this size

`krnn_to_mlr.py` exists and works, but MLRift has no `include_bytes`, array
statics take no initialiser and the compiler's whole-program string pool caps
at 65536 bytes — so a model can only be emitted as *code*, about 19 bytes of
`store32` per 4 model bytes. YuNet's 89 KB becomes a ~590 KB image. On
freestanding **xtensa that hangs**: `XT_STACK_TOP` is pinned at `0xd0040000`,
256 KiB into the load window, so the initial stack grows straight down into
the text of any larger image, mid-fill, with no diagnostic. Keep freestanding
xtensa images under ~200 KiB.

Load the model instead. `nn_model_run()` never writes to it and never assumes
where it lives, so a flash partition address and `qemu -device
loader,file=...,addr=...,force-raw=on` are the same thing. Note that qemu's
loader address is PHYSICAL: on the dc232b the cached window maps `0xd0000000`
→ `0x00000000`, so the model goes to `0x00800000` and the runner reads
`0xd0800000`. On real ESP32 silicon the same idea is spelled `--model` — see
below.

### On real ESP32 silicon: `--model` makes the blob a flash segment

**Measured, not projected:** YuNet @64 px on an ESP32-D0WD-V3 (40 MHz XTAL,
4 MB flash) prints all 1344 output values **byte-identical to
`int8_sim.py`**, and did so on 11 consecutive runs. So host x86_64,
xtensa-under-qemu, riscv32-under-qemu and real Xtensa LX6 hardware all agree
to the byte. One inference plus its UART dump takes ~20 s on the chip.


The qemu trick has an exact hardware equivalent, and it is not a flash driver.
`mlrc --target=esp32 --model <file>` appends the file's raw bytes to the
esp-image as an extra RAM segment loaded at `0x3FFB0000`; the mask ROM copies
it from flash into DRAM *before* the entry point runs, which you can watch in
the boot log:

```
rst:0x1 (POWERON_RESET),boot:0x13 (SPI_FAST_FLASH_BOOT)
mode:DIO, clock div:2
load:0x3ffb0000,len:89304      <- the model
load:0x3ffc5ce0,len:60         <- .data, moved up above it
load:0x40080400,len:13788      <- code
entry 0x40083894
BOOT 1313755723                <- 0x4E4E524B, 'KRNN' read back from 0x3FFB0000
BEGIN
...1344 values...
END 0
```

No MMU work, no SPI reads, and no ~590 KB image. The blob goes at the BOTTOM
of the DRAM window and `.data`/`.bss` move up above it, so `0x3FFB0000` is a
compile-time constant the program hardcodes (`tests/nn/nn_model_esp32.mlr`)
and it does not move when an unrelated static is added.

**The 192 KiB DRAM window is the binding constraint.** `[0x3FFB0000,
0x3FFE0000)` holds the model AND `.data`/`.bss` AND the stack. YuNet's `.krnn`
is 89304 B at any input size, and the runner's arena is exactly `8 * side^2`:

| side | arena | model + arena | fits 192 KiB? |
|---|---|---|---|
| 64 | 32768 | 122072 | yes, ~72 KiB spare |
| 96 | 73728 | 163040 | yes, ~32 KiB spare |
| 128 | 131072 | 220376 | **no** — over before code or stack |

YuNet is fully convolutional, so a smaller input is a valid model; it only
reduces detection range. `xt_esp32_check_layout` counts the blob against the
budget and loud-fails the compile rather than emitting an image that
overruns — see `src/ir_xtensa.mlr`.

Two things that cost real time on this board and are worth designing around:
it has no DTR/RTS, so **the host cannot reset the chip** and a capture that
attaches a second late misses a single-shot run entirely; and there is no
JTAG, so a hang is indistinguishable from a dead board. Hence
`nn_model_esp32.mlr` runs in a `loop` and prints a `BOOT <magic>` line before
each run: the loop makes the capture window irrelevant, and the magic word
read back from `0x3FFB0000` proves the ROM actually delivered the blob before
any number it prints can be trusted.

### Static declaration order decides what lands in .bss

Every static up to and including the last one with an initialiser is written
into the image file. Declaring an initialised scalar *after* a 192 KiB array
put the whole array in the ELF: a 210 KB freestanding image instead of 14 KB.
