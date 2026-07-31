#!/usr/bin/env python3
"""tools/ml/gen_yunet_shape.py — a SYNTHETIC .krnn with YuNet-class shapes.

    gen_yunet_shape.py <out.krnn>

Purpose: Stage 4 sizing question "is Task 6 (splitting the five spatial
kernels across both ESP32 cores) worth building on a REALISTIC model", not
on tests/nn/nn_model_ref.py's 12-layer smoke fixture (a container-format
test fixture, never meant to be shape-representative).

Cycle share is a property of LAYER SHAPES, not weight values (an int8 kernel's
loop trip counts and memory access pattern depend on H/W/C/kernel size, not
on what's in the tensors) -- so this model's weights, biases and quantization
params are arbitrary/deterministic (same Rng as nn_ref.py), and only the
SHAPES matter.

ARCHITECTURE, and what it is/isn't based on
--------------------------------------------
YuNet (Wu et al., "YuNet: A Tiny Millisecond-level Face Detector") is a
depthwise-separable-conv backbone (~75k params) with 5 backbone stages --
stages 0-1 downsample to 1/4 resolution and 3 -> 64 channels, stages 2-4
"use exactly the same network structure" and each feeds one scale of a
tiny FPN neck, followed by 1x1-conv detection heads at 3 scales. That much
is public (paper abstract + opencv_zoo README); the exact per-layer channel
schedule is behind a paywalled PDF and was not available to this script, so
the layer-by-layer plan below is an ENGINEERING RECREATION of that shape
class, not a transcription of the real weights file:

  stem      : conv2d 3x3 s2 p1        3 ->16   (stage 0, downsample 1/2)
  stage1    : dw 3x3 s1 p1  16->16  + pw 1x1 16->32 + maxpool 2x2 s2
              (downsample 1/2 again -> 1/4 total, channels 16 -> 32, as
              "stages 0-1 ... increase channels 3 -> 64" -- 32 here, one
              stage short of the paper's 64, is the acknowledged simplification)
  stage2    : dw 3x3 s2 p1  32->32  + pw 1x1 32->64      -> feat2 (1/8)
  stage3    : dw 3x3 s2 p1  64->64  + pw 1x1 64->64      -> feat3 (1/16)
  stage4    : dw 3x3 s2 p1  64->64  + pw 1x1 64->64      -> feat4 (1/32)
              (stage2-4 share one structure, per the paper -- this is the
              one part transcribed directly)
  neck      : resize_nn feat4 x2, add with feat3 -> fused3
              resize_nn fused3 x2, add with feat2 -> fused2
              (a tiny top-down FPN: the real TFPN may fuse differently, but
              upsample+add across adjacent scales is the standard shape)
  heads     : one pointwise (C -> 15 = 1 cls + 4 box + 10 kpt) + one
              sigmoid_lut per scale (feat4, fused3, fused2)

Every stage after stage1 is a SINGLE dw+pw block, where the real backbone
likely repeats more than one block per stage at the same channel count. That
under-counts total layers but does NOT bias the op-mix conclusion much: an
extra identical-shape block at an existing stage keeps the SAME per-element
kernel costs, so it dilutes elementwise (relu) and spatial (dw/pw) roughly
in the same proportion already present. See the report for why this matters.

Input 96x128x3 (H x W x C): NOT the 320x240 the task text floats, chosen
instead to divide cleanly by 32 (five halvings to a 3x4 head) so every
resize/add/pool in the neck lines up exactly with no ceil-rounding
special-casing. Channel counts (16/32/64) match the task's target exactly.
"""
import os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "tests", "nn"))
sys.path.insert(0, HERE)

import nn_ref as R      # noqa: E402
import krnn as K        # noqa: E402

IN_H, IN_W, IN_C = 96, 128, 3
IN_ZP = -5


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2

    rng = R.Rng(20260731)
    blob = K.Blob()
    layers = []

    off = {}
    cur = [0]

    def place(name, nbytes):
        off[name] = cur[0]
        cur[0] += (nbytes + 3) // 4 * 4
        return off[name]

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
        return rec["out_h"], rec["out_w"]

    def relu_layer(name, h, w, c, zp):
        r = K.blank_layer()
        r.update(op=K.OP_RELU, in0_off=off[name], out_off=off[name],
                  in_h=h, in_w=w, in_c=c, out_h=h, out_w=w, out_c=c,
                  in_zp=zp, out_zp=zp, n_elem=h * w * c)
        layers.append(r)

    def maxpool_layer(src, dst, h, w, c, kh, kw, sh, sw):
        oh, ow = R.out_dim(h, kh, sh, 0), R.out_dim(w, kw, sw, 0)
        r = K.blank_layer()
        r.update(op=K.OP_MAXPOOL, in0_off=off[src], out_off=off[dst],
                  in_h=h, in_w=w, in_c=c, out_h=oh, out_w=ow, out_c=c,
                  kh=kh, kw=kw, stride_h=sh, stride_w=sw,
                  n_elem=oh * ow * c)
        layers.append(r)
        return oh, ow

    def resize_layer(src, dst, h, w, c, sh, sw):
        oh, ow = h * sh, w * sw
        r = K.blank_layer()
        r.update(op=K.OP_RESIZE_NN, in0_off=off[src], out_off=off[dst],
                  in_h=h, in_w=w, in_c=c, out_h=oh, out_w=ow, out_c=c,
                  stride_h=sh, stride_w=sw, n_elem=oh * ow * c)
        layers.append(r)
        return oh, ow

    def add_layer(a, b, dst, h, w, c):
        am, ash = rng.mult(), rng.shift()
        bm, bsh = rng.mult(), rng.shift()
        om, osh = rng.mult(), rng.shift()
        r = K.blank_layer()
        r.update(op=K.OP_ADD, in0_off=off[a], in1_off=off[b], out_off=off[dst],
                  in_h=h, in_w=w, in_c=c, out_h=h, out_w=w, out_c=c,
                  in_zp=0, in1_zp=0, out_zp=0, n_elem=h * w * c,
                  left_shift=20, a_mult=am, a_shift=ash, b_mult=bm,
                  b_shift=bsh, o_mult=om, o_shift=osh)
        layers.append(r)

    def sigmoid_layer(name, h, w, c):
        lut = [rng.i8() for _ in range(256)]
        r = K.blank_layer()
        r.update(op=K.OP_SIGMOID_LUT, in0_off=off[name], out_off=off[name],
                  in_h=h, in_w=w, in_c=c, out_h=h, out_w=w, out_c=c,
                  n_elem=h * w * c, flags=1)
        r["lut_off"] = blob.add_i8_list(lut)
        layers.append(r)

    # ---- arena tensor placement -------------------------------------------
    place("in", IN_H * IN_W * IN_C)
    place("stem", 48 * 64 * 16)
    place("s1dw", 48 * 64 * 16)
    place("s1pw", 48 * 64 * 32)
    place("s1pool", 24 * 32 * 32)
    place("s2dw", 12 * 16 * 32)
    place("feat2", 12 * 16 * 64)
    place("s3dw", 6 * 8 * 64)
    place("feat3", 6 * 8 * 64)
    place("s4dw", 3 * 4 * 64)
    place("feat4", 3 * 4 * 64)
    place("up4", 6 * 8 * 64)
    place("fused3", 6 * 8 * 64)
    place("up3", 12 * 16 * 64)
    place("fused2", 12 * 16 * 64)
    place("head4", 3 * 4 * 15)
    place("head3", 6 * 8 * 15)
    place("head2", 12 * 16 * 15)

    # ---- stem ---------------------------------------------------------
    conv_layer(K.OP_CONV2D, "in", "stem", IN_H, IN_W, IN_C, IN_ZP, 16,
               3, 3, 2, 2, 1, 1, 6)
    relu_layer("stem", 48, 64, 16, 6)

    # ---- stage 1 (dw + pw + pool) --------------------------------------
    conv_layer(K.OP_DEPTHWISE, "stem", "s1dw", 48, 64, 16, 6, 16,
               3, 3, 1, 1, 1, 1, 4, depth_mult=1)
    relu_layer("s1dw", 48, 64, 16, 4)
    conv_layer(K.OP_POINTWISE, "s1dw", "s1pw", 48, 64, 16, 4, 32,
               1, 1, 1, 1, 0, 0, 3)
    relu_layer("s1pw", 48, 64, 32, 3)
    maxpool_layer("s1pw", "s1pool", 48, 64, 32, 2, 2, 2, 2)

    # ---- stage 2 (dw s2 + pw) -> feat2, 1/8 -----------------------------
    conv_layer(K.OP_DEPTHWISE, "s1pool", "s2dw", 24, 32, 32, 3, 32,
               3, 3, 2, 2, 1, 1, 2, depth_mult=1)
    relu_layer("s2dw", 12, 16, 32, 2)
    conv_layer(K.OP_POINTWISE, "s2dw", "feat2", 12, 16, 32, 2, 64,
               1, 1, 1, 1, 0, 0, 1)
    relu_layer("feat2", 12, 16, 64, 1)

    # ---- stage 3 (dw s2 + pw) -> feat3, 1/16 ----------------------------
    conv_layer(K.OP_DEPTHWISE, "feat2", "s3dw", 12, 16, 64, 1, 64,
               3, 3, 2, 2, 1, 1, 2, depth_mult=1)
    relu_layer("s3dw", 6, 8, 64, 2)
    conv_layer(K.OP_POINTWISE, "s3dw", "feat3", 6, 8, 64, 2, 64,
               1, 1, 1, 1, 0, 0, 1)
    relu_layer("feat3", 6, 8, 64, 1)

    # ---- stage 4 (dw s2 + pw) -> feat4, 1/32 ----------------------------
    conv_layer(K.OP_DEPTHWISE, "feat3", "s4dw", 6, 8, 64, 1, 64,
               3, 3, 2, 2, 1, 1, 2, depth_mult=1)
    relu_layer("s4dw", 3, 4, 64, 2)
    conv_layer(K.OP_POINTWISE, "s4dw", "feat4", 3, 4, 64, 2, 64,
               1, 1, 1, 1, 0, 0, 1)
    relu_layer("feat4", 3, 4, 64, 1)

    # ---- neck: top-down FPN, upsample + add -----------------------------
    resize_layer("feat4", "up4", 3, 4, 64, 2, 2)
    add_layer("up4", "feat3", "fused3", 6, 8, 64)
    resize_layer("fused3", "up3", 6, 8, 64, 2, 2)
    add_layer("up3", "feat2", "fused2", 12, 16, 64)

    # ---- heads: pointwise (C -> 15) + sigmoid, one per scale ------------
    conv_layer(K.OP_POINTWISE, "feat4", "head4", 3, 4, 64, 1, 15,
               1, 1, 1, 1, 0, 0, 0)
    sigmoid_layer("head4", 3, 4, 15)
    conv_layer(K.OP_POINTWISE, "fused3", "head3", 6, 8, 64, 0, 15,
               1, 1, 1, 1, 0, 0, 0)
    sigmoid_layer("head3", 6, 8, 15)
    conv_layer(K.OP_POINTWISE, "fused2", "head2", 12, 16, 64, 0, 15,
               1, 1, 1, 1, 0, 0, 0)
    sigmoid_layer("head2", 12, 16, 15)

    outputs = [(off["head4"], 3 * 4 * 15, 0, 65536),
               (off["head3"], 6 * 8 * 15, 0, 65536),
               (off["head2"], 12 * 16 * 15, 0, 65536)]

    layers_off = K.HEADER_BYTES
    outputs_off = layers_off + len(layers) * K.LAYER_STRIDE
    blob_off = outputs_off + len(outputs) * K.OUTPUT_STRIDE
    arena_bytes = (cur[0] + 3) // 4 * 4
    hdr = K.pack_header(len(layers), arena_bytes, layers_off, outputs_off,
                        len(outputs), blob_off, len(blob), off["in"],
                        IN_H, IN_W, IN_C, IN_ZP, 65536)
    data = hdr + b"".join(K.pack_layer(r) for r in layers)
    data += b"".join(K.pack_output(*o) for o in outputs)
    data += bytes(blob.buf)
    open(sys.argv[1], "wb").write(data)

    print("layers=%d arena_bytes=%d blob_bytes=%d" %
          (len(layers), arena_bytes, len(blob)), file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
