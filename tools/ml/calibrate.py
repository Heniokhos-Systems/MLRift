#!/usr/bin/env python3
"""Stage 2: collect per-tensor activation ranges over real images.

Exposes every intermediate tensor as a graph output and runs onnxruntime over
the calibration set. These ranges are what the int8 requantise
(multiplier, shift) pairs are derived from -- weight quantisation is data-free,
activation quantisation is not.

Two modes:

  min/max      (default)  the raw observed extremes. Simple, but one outlier
                          activation inflates the range for every image and
                          wastes most of the 256 levels.
  percentile   (--pct P)  two-pass: min/max bounds a histogram, then each tail
                          is clipped to (100-P)/2 %. Standard practice; keeps
                          the bulk of the distribution at full resolution and
                          lets rare outliers saturate instead.
"""
import argparse, glob, json
import numpy as np
import onnx
import onnxruntime as ort
from PIL import Image

BINS = 2048


def build_session(model_path, side):
    m = onnx.load(model_path)
    d = m.graph.input[0].type.tensor_type.shape.dim
    d[2].dim_value = side
    d[3].dim_value = side
    del m.graph.value_info[:]
    for o in m.graph.output:
        del o.type.tensor_type.shape.dim[:]
    produced = {o for n in m.graph.node for o in n.output}
    existing = {o.name for o in m.graph.output}
    for t in sorted(produced - existing):
        m.graph.output.extend([
            onnx.helper.make_tensor_value_info(t, onnx.TensorProto.FLOAT, None)])
    so = ort.SessionOptions()
    so.log_severity_level = 3
    return ort.InferenceSession(m.SerializeToString(), so,
                                providers=["CPUExecutionProvider"])


def load(path, side):
    im = Image.open(path).convert("RGB").resize((side, side))
    return np.asarray(im, dtype=np.float32).transpose(2, 0, 1)[None]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("size", type=int)
    ap.add_argument("out"); ap.add_argument("images")
    ap.add_argument("--pct", type=float, default=None,
                    help="keep this %% of the distribution (e.g. 99.9)")
    args = ap.parse_args()

    imgs = sorted(glob.glob(args.images + "/*.jpg"))
    sess = build_session(args.model, args.size)
    names = [o.name for o in sess.get_outputs()]
    iname = sess.get_inputs()[0].name

    # pass 1 -- observed extremes
    lo = {n: np.inf for n in names}
    hi = {n: -np.inf for n in names}
    for k, p in enumerate(imgs):
        for n, v in zip(names, sess.run(None, {iname: load(p, args.size)})):
            v = np.asarray(v)
            lo[n] = min(lo[n], float(v.min()))
            hi[n] = max(hi[n], float(v.max()))
        if (k + 1) % 25 == 0:
            print(f"    pass1 {k+1}/{len(imgs)}")

    if args.pct is None:
        out = {n: [lo[n], hi[n]] for n in names}
    else:
        # pass 2 -- histogram within the observed bounds, then clip the tails
        hist = {n: np.zeros(BINS, dtype=np.int64) for n in names}
        for k, p in enumerate(imgs):
            for n, v in zip(names, sess.run(None, {iname: load(p, args.size)})):
                span = hi[n] - lo[n]
                if span <= 0:
                    continue
                idx = np.clip(((np.asarray(v).ravel() - lo[n]) / span
                               * (BINS - 1)).astype(np.int32), 0, BINS - 1)
                hist[n] += np.bincount(idx, minlength=BINS)
            if (k + 1) % 25 == 0:
                print(f"    pass2 {k+1}/{len(imgs)}")

        tail = (100.0 - args.pct) / 200.0
        out = {}
        for n in names:
            span = hi[n] - lo[n]
            h = hist[n]
            tot = h.sum()
            if span <= 0 or tot == 0:
                out[n] = [lo[n], hi[n]]
                continue
            c = np.cumsum(h) / tot
            i_lo = int(np.searchsorted(c, tail))
            i_hi = int(np.searchsorted(c, 1.0 - tail))
            l = lo[n] + span * i_lo / (BINS - 1)
            r = lo[n] + span * i_hi / (BINS - 1)
            if r <= l:
                l, r = lo[n], hi[n]
            out[n] = [float(l), float(r)]

    json.dump(out, open(args.out, "w"))
    mode = "min/max" if args.pct is None else f"{args.pct}% percentile"
    print(f"  {len(imgs)} images, {len(names)} tensors, {mode} -> {args.out}")
    if args.pct is not None:
        shrink = [(hi[n] - lo[n]) / max(out[n][1] - out[n][0], 1e-12)
                  for n in names if hi[n] > lo[n]]
        print(f"  range shrink vs min/max: median {np.median(shrink):.2f}x, "
              f"max {max(shrink):.2f}x")


if __name__ == "__main__":
    main()
