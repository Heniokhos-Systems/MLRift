#!/usr/bin/env python3
"""Float reference runner for the MLRift NN port.

Executes an ONNX CNN with a hand-written NumPy implementation of exactly the
layer set std/nn_int8.mlr provides (plus Resize/Sigmoid). Two jobs:

 1. Prove our understanding of the graph is right — compare against onnxruntime
    if available, else against a second independent path.
 2. Give the int8 pipeline a ground truth to be scored against later, and
    record per-tensor activation ranges (the calibration stage-2 needs).

NCHW throughout here (ONNX native). The MLRift kernels are NHWC; the converter
transposes weights, so this reference deliberately stays in ONNX's layout to
avoid baking the same transpose bug into both sides.
"""
import argparse, json, sys
import numpy as np
import onnx
from onnx import numpy_helper


def conv(x, w, b, strides, pads, groups, dilations):
    N, C, H, W = x.shape
    O, IC, KH, KW = w.shape
    sh, sw = strides
    dh, dw = dilations
    pt, pl, pb, pr = pads
    xp = np.pad(x, ((0, 0), (0, 0), (pt, pb), (pl, pr)))
    OH = (xp.shape[2] - (KH - 1) * dh - 1) // sh + 1
    OW = (xp.shape[3] - (KW - 1) * dw - 1) // sw + 1
    out = np.zeros((N, O, OH, OW), dtype=np.float64)
    ocpg = O // groups
    for g in range(groups):
        xs = xp[:, g * IC:(g + 1) * IC]
        ws = w[g * ocpg:(g + 1) * ocpg]
        for oy in range(OH):
            for ox in range(OW):
                patch = xs[:, :, oy*sh:oy*sh + (KH-1)*dh + 1:dh,
                              ox*sw:ox*sw + (KW-1)*dw + 1:dw]
                out[:, g*ocpg:(g+1)*ocpg, oy, ox] = np.tensordot(
                    patch, ws, axes=([1, 2, 3], [1, 2, 3]))
    if b is not None:
        out += b.reshape(1, -1, 1, 1)
    return out


def maxpool(x, kernel, strides, pads):
    N, C, H, W = x.shape
    kh, kw = kernel; sh, sw = strides
    pt, pl, pb, pr = pads
    xp = np.pad(x, ((0, 0), (0, 0), (pt, pb), (pl, pr)), constant_values=-np.inf)
    OH = (xp.shape[2] - kh) // sh + 1
    OW = (xp.shape[3] - kw) // sw + 1
    out = np.empty((N, C, OH, OW))
    for oy in range(OH):
        for ox in range(OW):
            out[:, :, oy, ox] = xp[:, :, oy*sh:oy*sh+kh, ox*sw:ox*sw+kw].max(axis=(2, 3))
    return out


def resize_nearest(x, scales):
    _, _, H, W = x.shape
    OH, OW = int(H * scales[2]), int(W * scales[3])
    iy = np.floor(np.arange(OH) / scales[2]).astype(int).clip(0, H - 1)
    ix = np.floor(np.arange(OW) / scales[3]).astype(int).clip(0, W - 1)
    return x[:, :, iy][:, :, :, ix]


def run(model_path, x, collect_ranges=True):
    m = onnx.load(model_path)
    g = m.graph
    env = {t.name: numpy_helper.to_array(t).astype(np.float64)
           for t in g.initializer}
    env[g.input[0].name] = x.astype(np.float64)
    ranges = {}

    def A(n):
        d = {}
        for a in n.attribute:
            if a.type == onnx.AttributeProto.INT:      d[a.name] = a.i
            elif a.type == onnx.AttributeProto.INTS:   d[a.name] = list(a.ints)
            elif a.type == onnx.AttributeProto.FLOAT:  d[a.name] = a.f
            elif a.type == onnx.AttributeProto.STRING: d[a.name] = a.s.decode()
        return d

    for n in g.node:
        a = A(n)
        op = n.op_type
        if op == "Conv":
            w = env[n.input[1]]
            b = env[n.input[2]] if len(n.input) > 2 else None
            y = conv(env[n.input[0]], w, b,
                     a.get("strides", [1, 1]), a.get("pads", [0, 0, 0, 0]),
                     a.get("group", 1), a.get("dilations", [1, 1]))
        elif op == "Relu":
            y = np.maximum(env[n.input[0]], 0)
        elif op == "Sigmoid":
            y = 1.0 / (1.0 + np.exp(-env[n.input[0]]))
        elif op == "Add":
            y = env[n.input[0]] + env[n.input[1]]
        elif op == "MaxPool":
            y = maxpool(env[n.input[0]], a.get("kernel_shape", [2, 2]),
                        a.get("strides", [2, 2]), a.get("pads", [0, 0, 0, 0]))
        elif op == "Resize":
            scales = env[n.input[2]] if len(n.input) > 2 and n.input[2] in env else None
            if scales is None or len(scales) == 0:
                sizes = env[n.input[3]]
                src = env[n.input[0]].shape
                scales = np.array([sizes[i] / src[i] for i in range(4)])
            y = resize_nearest(env[n.input[0]], scales)
        elif op == "Transpose":
            y = np.transpose(env[n.input[0]], a.get("perm"))
        elif op == "Reshape":
            shape = env[n.input[1]].astype(int).tolist()
            y = env[n.input[0]].reshape([int(s) for s in shape])
        else:
            raise NotImplementedError(f"{op} (node {n.name})")
        env[n.output[0]] = y
        if collect_ranges:
            ranges[n.output[0]] = (float(np.min(y)), float(np.max(y)))

    return {o.name: env[o.name] for o in g.output}, ranges


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--size", type=int, default=128, help="square input side")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ranges-out")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    x = rng.random((1, 3, args.size, args.size), dtype=np.float64) * 255.0

    outs, ranges = run(args.model, x)
    print(f"  input {tuple(x.shape)}")
    for k, v in outs.items():
        print(f"    {k:<10} {str(v.shape):<22} "
              f"min={v.min():+.4f} max={v.max():+.4f}")
    if args.ranges_out:
        json.dump(ranges, open(args.ranges_out, "w"))
        print(f"  activation ranges -> {args.ranges_out} ({len(ranges)} tensors)")


if __name__ == "__main__":
    sys.exit(main())
