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
