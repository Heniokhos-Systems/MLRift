# ONNX → MLRift int8 pipeline

Offline tooling that turns a float ONNX CNN into int8 weights + quantisation
parameters for `std/nn_int8.mlr`. Host-side Python; nothing here runs on the
target.

## Stages

| script | does |
|---|---|
| `onnx_to_mlrift.py` | parse graph, quantise weights per-channel symmetric (data-free), emit `.layers.json` / `.weights.bin` / `.qparams.json` |
| `float_ref.py` | NumPy walk of exactly the layer set `nn_int8.mlr` provides (+Resize/Sigmoid). Ground truth. **Validated against onnxruntime to 7.6e-07 on YuNet.** |
| `calibrate.py` | expose every intermediate as a graph output, accumulate per-tensor min/max over real images → activation ranges |
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

## Status

Weights + activation ranges + a validated float reference all exist. Current
int8 accuracy on YuNet @128px with naive min/max calibration: most outputs
4–18% of range, `obj_*` worse (values sit near zero). Percentile-clipped
calibration is the standard next improvement.
