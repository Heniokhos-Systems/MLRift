#!/usr/bin/env python3
"""Stage 2: collect per-tensor activation ranges over real images.

Exposes every intermediate tensor as a graph output and runs onnxruntime over
the calibration set, accumulating min/max. These ranges are what the int8
requantise (multiplier, shift) pairs are derived from — weight quantisation is
data-free, activation quantisation is not.
"""
import json, sys, glob
import numpy as np, onnx, onnxruntime as ort
from PIL import Image

MODEL, SIDE, OUT = sys.argv[1], int(sys.argv[2]), sys.argv[3]
imgs = sorted(glob.glob(sys.argv[4] + "/*.jpg"))

m = onnx.load(MODEL)
d = m.graph.input[0].type.tensor_type.shape.dim
d[2].dim_value = SIDE; d[3].dim_value = SIDE
del m.graph.value_info[:]
for o in m.graph.output: del o.type.tensor_type.shape.dim[:]

produced = {o for n in m.graph.node for o in n.output}
existing = {o.name for o in m.graph.output}
for t in sorted(produced - existing):
    m.graph.output.extend([onnx.helper.make_tensor_value_info(t, onnx.TensorProto.FLOAT, None)])

so = ort.SessionOptions(); so.log_severity_level = 3
sess = ort.InferenceSession(m.SerializeToString(), so, providers=['CPUExecutionProvider'])
names = [o.name for o in sess.get_outputs()]
iname = sess.get_inputs()[0].name

lo = {n: np.inf for n in names}; hi = {n: -np.inf for n in names}
for k, p in enumerate(imgs):
    im = Image.open(p).convert("RGB").resize((SIDE, SIDE))
    x = np.asarray(im, dtype=np.float32).transpose(2, 0, 1)[None]  # NCHW, 0..255
    for n, v in zip(names, sess.run(None, {iname: x})):
        v = np.asarray(v)
        lo[n] = min(lo[n], float(v.min())); hi[n] = max(hi[n], float(v.max()))
    if (k + 1) % 25 == 0: print(f"    {k+1}/{len(imgs)}")

json.dump({n: [lo[n], hi[n]] for n in names}, open(OUT, "w"))
print(f"  {len(imgs)} images, {len(names)} tensors -> {OUT}")
wide = sorted(((hi[n]-lo[n], n) for n in names), reverse=True)[:5]
print("  widest activation ranges:")
for r, n in wide: print(f"    {n:<24} [{lo[n]:+.2f}, {hi[n]:+.2f}]  span {r:.2f}")
