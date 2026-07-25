#!/usr/bin/env python3
"""Reference for the std/nn_model.mlr runner: build a .krnn, then say exactly
what running it must print.

    nn_model_ref.py <out.krnn>      # writes the model, prints expected output

This is the self-contained half of the runner's validation. It needs nothing
but the standard library — no ONNX, no numpy, no model files — so the test
suite always exercises the container format, the layer walk, the arena offsets
and every opcode, on host and under qemu, on any machine.

The other half is the real thing: tools/ml/int8_sim.py against a converted
YuNet. That one proves the numbers are right on a 106-node graph; this one
proves the mechanism is right and keeps proving it in CI.

The synthetic graph deliberately covers what YuNet does NOT exercise
(avgpool, a materialised COPY, a depth multiplier > 1 is left out only
because the kernel test already covers it) as well as what it does:

    L0  CONV2D      8x8x3  -> 8x8x4    3x3 s1 p1, per-channel, bias
    L1  RELU        in place
    L2  MAXPOOL     8x8x4  -> 4x4x4    2x2 s2
    L3  DEPTHWISE   4x4x4  -> 4x4x4    3x3 s1 p1
    L4  POINTWISE   4x4x4  -> 4x4x6    1x1
    L5  RESIZE_NN   4x4x6  -> 8x8x6    scale 2x2
    L6  CONV2D      8x8x6  -> 4x4x6    3x3 s2 p1
    L7  ADD         L4 + L6            (L4 stays live across three layers)
    L8  SIGMOID_LUT 4x4x6              256-entry table
    L9  NOP         aliases L8
    L10 COPY        4x4x6              a materialised layout change
    L11 AVGPOOL     4x4x6  -> 2x2x6    2x2 s2

Outputs: L8's tensor and L11's tensor.

The kernels themselves are tests/nn/nn_ref.py's — already validated against
std/nn_int8.mlr to exact equality. What is new here, and what this file is
actually testing, is that the runner reads the right field from the right
offset, hands the right pointer to the right kernel, and keeps tensors that
are still live from being overwritten.
"""
import os, struct, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "tools", "ml"))

import nn_ref as R          # noqa: E402  kernels + Rng + fixed point
import krnn as K            # noqa: E402  the container format


# --------------------------------------------------------------------------
# Model construction
# --------------------------------------------------------------------------

IN_H, IN_W, IN_C = 8, 8, 3
IN_ZP = -5


def build():
    rng = R.Rng(777)
    blob = K.Blob()
    layers = []
    params = []                 # per layer: whatever the reference walk needs

    # Arena offsets are hand-assigned here rather than planned: this model is
    # small, and a fixed layout is one less thing between a failure and its
    # cause. The real planner is exercised by the YuNet path.
    off = {}
    cur = [0]

    def place(name, nbytes):
        off[name] = cur[0]
        cur[0] += (nbytes + 3) // 4 * 4
        return off[name]

    place("in", IN_H * IN_W * IN_C)
    place("t0", 8 * 8 * 4)
    place("t2", 4 * 4 * 4)
    place("t3", 4 * 4 * 4)
    place("t4", 4 * 4 * 6)
    place("t5", 8 * 8 * 6)
    place("t6", 4 * 4 * 6)
    place("t7", 4 * 4 * 6)
    place("t8", 4 * 4 * 6)
    place("t10", 4 * 4 * 6)
    place("t11", 2 * 2 * 6)

    def qparams(n):
        return ([rng.bias() for _ in range(n)],
                [rng.mult() for _ in range(n)],
                [rng.shift() for _ in range(n)])

    def conv_layer(op, src, dst, in_h, in_w, in_c, in_zp, out_c, kh, kw,
                   sh, sw, ph, pw, out_zp, depth_mult=0):
        n_w = (out_c * kh * kw * in_c) if op != K.OP_DEPTHWISE \
            else (kh * kw * out_c)
        w = R.fill_i8(rng, n_w)
        bias, mult, shift = qparams(out_c)
        rec = K.blank_layer()
        rec.update(op=op, in0_off=off[src], out_off=off[dst],
                   in_h=in_h, in_w=in_w, in_c=in_c,
                   out_h=R.out_dim(in_h, kh, sh, ph),
                   out_w=R.out_dim(in_w, kw, sw, pw), out_c=out_c,
                   kh=kh, kw=kw, stride_h=sh, stride_w=sw, pad_h=ph, pad_w=pw,
                   depth_mult=depth_mult, q_stride=1,
                   in_zp=in_zp, out_zp=out_zp)
        rec["w_off"] = blob.add_i8_list(w)
        rec["bias_off"] = blob.add_i32(bias)
        rec["mult_off"] = blob.add_i32(mult)
        rec["shift_off"] = blob.add_i32(shift)
        rec["n_elem"] = rec["out_h"] * rec["out_w"] * out_c
        layers.append(rec)
        params.append(dict(w=w, bias=bias, mult=mult, shift=shift))

    # L0 conv2d
    conv_layer(K.OP_CONV2D, "in", "t0", 8, 8, 3, IN_ZP, 4, 3, 3, 1, 1, 1, 1, 7)
    # L1 relu, in place on t0
    r = K.blank_layer()
    r.update(op=K.OP_RELU, in0_off=off["t0"], out_off=off["t0"],
             in_h=8, in_w=8, in_c=4, out_h=8, out_w=8, out_c=4,
             in_zp=7, out_zp=7, n_elem=8 * 8 * 4)
    layers.append(r); params.append({})
    # L2 maxpool
    r = K.blank_layer()
    r.update(op=K.OP_MAXPOOL, in0_off=off["t0"], out_off=off["t2"],
             in_h=8, in_w=8, in_c=4, out_h=4, out_w=4, out_c=4,
             kh=2, kw=2, stride_h=2, stride_w=2,
             in_zp=7, out_zp=7, n_elem=4 * 4 * 4)
    layers.append(r); params.append({})
    # L3 depthwise
    conv_layer(K.OP_DEPTHWISE, "t2", "t3", 4, 4, 4, 7, 4, 3, 3, 1, 1, 1, 1,
               -3, depth_mult=1)
    # L4 pointwise
    conv_layer(K.OP_POINTWISE, "t3", "t4", 4, 4, 4, -3, 6, 1, 1, 1, 1, 0, 0, 2)
    # L5 resize x2
    r = K.blank_layer()
    r.update(op=K.OP_RESIZE_NN, in0_off=off["t4"], out_off=off["t5"],
             in_h=4, in_w=4, in_c=6, out_h=8, out_w=8, out_c=6,
             stride_h=2, stride_w=2, in_zp=2, out_zp=2, n_elem=8 * 8 * 6)
    layers.append(r); params.append({})
    # L6 conv2d stride 2 back down
    conv_layer(K.OP_CONV2D, "t5", "t6", 8, 8, 6, 2, 6, 3, 3, 2, 2, 1, 1, -9)
    # L7 add
    am, ash = rng.mult(), rng.shift()
    bm, bsh = rng.mult(), rng.shift()
    om, osh = rng.mult(), rng.shift()
    r = K.blank_layer()
    r.update(op=K.OP_ADD, in0_off=off["t4"], in1_off=off["t6"],
             out_off=off["t7"], in_h=4, in_w=4, in_c=6,
             out_h=4, out_w=4, out_c=6, in_zp=2, in1_zp=-9, out_zp=4,
             n_elem=4 * 4 * 6, left_shift=20,
             a_mult=am, a_shift=ash, b_mult=bm, b_shift=bsh,
             o_mult=om, o_shift=osh)
    layers.append(r)
    params.append(dict(am=am, ash=ash, bm=bm, bsh=bsh, om=om, osh=osh))
    # L8 sigmoid via a 256-entry table
    lut = [rng.i8() for _ in range(256)]
    r = K.blank_layer()
    r.update(op=K.OP_SIGMOID_LUT, in0_off=off["t7"], out_off=off["t8"],
             in_h=4, in_w=4, in_c=6, out_h=4, out_w=4, out_c=6,
             in_zp=4, out_zp=0, n_elem=4 * 4 * 6, flags=1)
    r["lut_off"] = blob.add_i8_list(lut)
    layers.append(r); params.append(dict(lut=lut))
    # L9 nop, aliases t8
    r = K.blank_layer()
    r.update(op=K.OP_NOP, in0_off=off["t8"], out_off=off["t8"],
             in_h=4, in_w=4, in_c=6, out_h=4, out_w=4, out_c=6,
             n_elem=4 * 4 * 6)
    layers.append(r); params.append({})
    # L10 copy
    r = K.blank_layer()
    r.update(op=K.OP_COPY, in0_off=off["t8"], out_off=off["t10"],
             in_h=4, in_w=4, in_c=6, out_h=4, out_w=4, out_c=6,
             n_elem=4 * 4 * 6)
    layers.append(r); params.append({})
    # L11 avgpool
    r = K.blank_layer()
    r.update(op=K.OP_AVGPOOL, in0_off=off["t10"], out_off=off["t11"],
             in_h=4, in_w=4, in_c=6, out_h=2, out_w=2, out_c=6,
             kh=2, kw=2, stride_h=2, stride_w=2,
             n_elem=2 * 2 * 6, flags=1)
    layers.append(r); params.append({})

    outputs = [(off["t8"], 4 * 4 * 6, 0, 3277),
               (off["t11"], 2 * 2 * 6, 0, 6554)]
    return layers, params, blob, off, cur[0], outputs


# --------------------------------------------------------------------------
# The reference walk — same graph, nn_ref.py's kernels, no .krnn involved
# --------------------------------------------------------------------------

def reference(layers, params):
    rng = R.Rng(12345)
    x = [((rng.r32() >> 16) & 0xFF) for _ in range(IN_H * IN_W * IN_C)]
    x = [v - 256 if v > 127 else v for v in x]

    p = params
    t0 = R.conv2d(x, 8, 8, 3, IN_ZP, p[0]["w"], 3, 3, p[0]["bias"],
                  p[0]["mult"], p[0]["shift"], 1, 4, 7, 1, 1, 1, 1, -128, 127)
    t0 = [v if v >= 7 else 7 for v in t0]                       # L1 relu
    t2 = R.maxpool(t0, 8, 8, 4, 2, 2, 2, 2, 0, 0, -128, 127)    # L2
    t3 = R.depthwise(t2, 4, 4, 4, 7, p[3]["w"], 3, 3, p[3]["bias"],
                     p[3]["mult"], p[3]["shift"], 1, -3, 1, 1, 1, 1, 1,
                     -128, 127)                                  # L3
    t4 = R.conv2d(t3, 4, 4, 4, -3, p[4]["w"], 1, 1, p[4]["bias"],
                  p[4]["mult"], p[4]["shift"], 1, 6, 2, 1, 1, 0, 0,
                  -128, 127)                                     # L4
    t5 = []                                                      # L5 resize
    for oy in range(8):
        iy = oy // 2
        for ox in range(8):
            ix = ox // 2
            base = (iy * 4 + ix) * 6
            t5 += t4[base:base + 6]
    t6 = R.conv2d(t5, 8, 8, 6, 2, p[6]["w"], 3, 3, p[6]["bias"],
                  p[6]["mult"], p[6]["shift"], 1, 6, -9, 2, 2, 1, 1,
                  -128, 127)                                     # L6
    a = p[7]
    t7 = R.add_int8(t4, 2, a["am"], a["ash"], t6, -9, a["bm"], a["bsh"],
                    4, a["om"], a["osh"], 4 * 4 * 6, 20, -128, 127)  # L7
    lut = p[8]["lut"]
    t8 = [lut[v + 128] for v in t7]                              # L8
    t10 = list(t8)                                               # L9 nop, L10
    t11 = R.avgpool(t10, 4, 4, 6, 2, 2, 2, 2, 0, 0, -128, 127)   # L11
    return t8 + t11


# --------------------------------------------------------------------------

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    layers, params, blob, off, arena_used, outputs = build()

    layers_off = K.HEADER_BYTES
    outputs_off = layers_off + len(layers) * K.LAYER_STRIDE
    blob_off = outputs_off + len(outputs) * K.OUTPUT_STRIDE
    hdr = K.pack_header(len(layers), (arena_used + 3) // 4 * 4, layers_off,
                        outputs_off, len(outputs), blob_off, len(blob),
                        off["in"], IN_H, IN_W, IN_C, IN_ZP, 65536)
    data = hdr + b"".join(K.pack_layer(r) for r in layers)
    data += b"".join(K.pack_output(*o) for o in outputs)
    data += bytes(blob.buf)
    open(sys.argv[1], "wb").write(data)

    for v in reference(layers, params):
        print(v)
    return 0


if __name__ == "__main__":
    sys.exit(main())
