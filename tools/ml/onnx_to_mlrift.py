#!/usr/bin/env python3
"""ONNX -> MLRift int8 model converter (stage 1: structure + weights).

Reads an fp32 ONNX CNN and emits:
  <out>.layers.json  — layer descriptors the MLRift runner walks
  <out>.weights.bin  — int8 weights, per-channel symmetric, weight zp = 0
  <out>.qparams.json — per-layer per-channel weight scales + int32 biases

Weight quantisation is data-free (symmetric per output channel), which is
exactly what std/nn_int8.mlr assumes. ACTIVATION scales still need a
calibration pass over real images — that is stage 2; without it the
(multiplier, shift) requantise pairs cannot be computed, so this stage
records weight scales only and leaves activation ranges unset.
"""
import argparse, json, struct, sys
import numpy as np
import onnx
from onnx import numpy_helper


def attrs(node):
    out = {}
    for a in node.attribute:
        if a.type == onnx.AttributeProto.INT:      out[a.name] = a.i
        elif a.type == onnx.AttributeProto.INTS:   out[a.name] = list(a.ints)
        elif a.type == onnx.AttributeProto.FLOAT:  out[a.name] = a.f
        elif a.type == onnx.AttributeProto.STRING: out[a.name] = a.s.decode()
    return out


def quantize_per_channel(w):
    """Symmetric int8, one scale per output channel (axis 0). zp = 0."""
    flat = w.reshape(w.shape[0], -1)
    amax = np.abs(flat).max(axis=1)
    amax[amax == 0] = 1e-12                     # dead channel -> avoid /0
    scale = amax / 127.0
    q = np.rint(flat / scale[:, None]).clip(-127, 127).astype(np.int8)
    return q.reshape(w.shape), scale


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("-o", "--out", required=True, help="output prefix")
    args = ap.parse_args()

    m = onnx.load(args.model)
    g = m.graph
    init = {t.name: numpy_helper.to_array(t) for t in g.initializer}

    layers, blob, qparams = [], bytearray(), []
    unsupported = set()
    SUPPORTED = {"Conv", "Relu", "MaxPool", "Add", "Resize", "Sigmoid",
                 "Transpose", "Reshape"}

    for idx, n in enumerate(g.node):
        if n.op_type not in SUPPORTED:
            unsupported.add(n.op_type)
        a = attrs(n)
        rec = {"idx": idx, "op": n.op_type, "name": n.name,
               "inputs": list(n.input), "outputs": list(n.output)}

        if n.op_type == "Conv":
            w = init[n.input[1]]                       # OIHW
            groups = a.get("group", 1)
            depthwise = groups > 1 and groups == w.shape[0]
            q, scale = quantize_per_channel(w)
            # OIHW -> OHWI (what nn_conv2d_int8 expects); depthwise -> 1HWC
            q_nhwc = np.transpose(q, (0, 2, 3, 1))
            rec.update({
                "kind": "depthwise" if depthwise else
                        ("pointwise" if w.shape[2:] == (1, 1) else "conv"),
                "groups": groups,
                "weight_shape_ohwi": list(q_nhwc.shape),
                "strides": a.get("strides", [1, 1]),
                "pads": a.get("pads", [0, 0, 0, 0]),
                "dilations": a.get("dilations", [1, 1]),
                "weight_off": len(blob),
                "weight_len": q_nhwc.size,
            })
            blob += q_nhwc.tobytes()
            entry = {"idx": idx, "weight_scale": scale.tolist()}
            if len(n.input) > 2:                        # fp32 bias, requantised in stage 2
                entry["bias_fp32"] = init[n.input[2]].tolist()
            qparams.append(entry)

        elif n.op_type in ("MaxPool",):
            rec.update({"kernel": a.get("kernel_shape", [2, 2]),
                        "strides": a.get("strides", [2, 2]),
                        "pads": a.get("pads", [0, 0, 0, 0])})
        elif n.op_type == "Resize":
            rec.update({"mode": a.get("mode", "nearest"),
                        "nearest_mode": a.get("nearest_mode", "floor")})
        elif n.op_type == "Transpose":
            rec.update({"perm": a.get("perm", [])})

        layers.append(rec)

    meta = {
        "source": args.model,
        "opset": [o.version for o in m.opset_import],
        "input": {i.name: [d.dim_value or d.dim_param
                           for d in i.type.tensor_type.shape.dim] for i in g.input},
        "outputs": [o.name for o in g.output],
        "layers": layers,
        "weight_bytes": len(blob),
        "note": "activation scales NOT set — requires stage-2 calibration",
    }
    open(f"{args.out}.layers.json", "w").write(json.dumps(meta, indent=1))
    open(f"{args.out}.weights.bin", "wb").write(bytes(blob))
    open(f"{args.out}.qparams.json", "w").write(json.dumps(qparams))

    print(f"  layers      : {len(layers)}")
    print(f"  int8 weights: {len(blob):,} bytes")
    if unsupported:
        print(f"  UNSUPPORTED ops present: {sorted(unsupported)}")
    else:
        print("  all ops within the supported set")


if __name__ == "__main__":
    sys.exit(main())
