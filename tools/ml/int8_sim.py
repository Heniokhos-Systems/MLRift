#!/usr/bin/env python3
"""Stage 3: int8 pipeline simulator, bit-faithful to std/nn_int8.mlr.

Every arithmetic step here mirrors the MLRift kernels (which in turn follow
gemmlowp/TFLite-Micro), so the numbers this prints are what the MCU will
produce -- not an approximation of it:

  * per-output-channel symmetric int8 weights, zero point 0
  * asymmetric int8 activations (scale + zero point) from calibration ranges
  * accumulate in int32
  * requantise with saturating-rounding-doubling-high-mul + rounding shift

Scored against float_ref.py, which is itself validated against onnxruntime.
"""
import json, sys, argparse
import numpy as np
import onnx
from onnx import numpy_helper

INT8_MIN, INT8_MAX = -128, 127


# ---- fixed-point primitives (mirror nn_srdhm / nn_rdbpot in nn_int8.mlr) ----

def quantize_multiplier(m):
    """float multiplier -> (int32 multiplier in [2^30, 2^31), signed shift).

    This is TFLite's QuantizeMultiplier and its SIGN CONVENTION: the pair
    means `m == (q / 2^31) * 2^shift`, so a POSITIVE shift shifts the input
    LEFT before the multiply and a NEGATIVE shift divides the product right
    afterwards. That is exactly what nn_mul_by_qm() in std/nn_int8.mlr
    consumes, so the .krnn emitter passes these straight through.
    """
    if m == 0.0:
        return 0, 0
    s = 0
    while m < 0.5:
        m *= 2.0; s -= 1
    while m >= 1.0:
        m /= 2.0; s += 1
    q = int(round(m * (1 << 31)))
    if q == (1 << 31):
        q //= 2; s += 1
    return q, s


def srdhm(a, b):
    """gemmlowp SaturatingRoundingDoublingHighMul, int32 x int32 -> int32.

    Mirrors nn_srdhm in std/nn_int8.mlr and srdhm in tests/nn/nn_ref.py:
    (a*b + nudge) / 2^31 over a TRUE 64-bit product, divided with C's
    truncation TOWARD ZERO -- not an arithmetic right shift, which floors and
    is off by one for a negative product with a remainder. "Doubling" is
    inherent in dividing by 2^31 rather than 2^32; the product is NOT scaled.
    """
    a = np.asarray(a, dtype=np.int64); b = np.int64(b)
    overflow = (a == -(1 << 31)) & (b == -(1 << 31))
    ab = a * b
    nudge = np.where(ab >= 0, 1 << 30, 1 - (1 << 30))
    s = ab + nudge
    q = np.abs(s) // (1 << 31)
    q = np.where(s < 0, -q, q)
    return np.where(overflow, (1 << 31) - 1, q).astype(np.int64)


def rdbpot(x, exp):
    """Rounding divide by power of two (round half away from zero)."""
    if exp <= 0:
        return x
    x = np.asarray(x, dtype=np.int64)
    mask = (1 << exp) - 1
    rem = x & mask
    threshold = (mask >> 1) + np.where(x < 0, 1, 0)
    return (x >> exp) + np.where(rem > threshold, 1, 0)


def requantize(acc, mult, shift):
    """TFLite MultiplyByQuantizedMultiplier, == nn_mul_by_qm in nn_int8.mlr.
    `shift` is TFLite-signed: positive shifts the input left first."""
    left = max(0, shift)
    right = max(0, -shift)
    if left:
        acc = np.asarray(acc, dtype=np.int64) * (1 << left)
    return rdbpot(srdhm(acc, mult), right)


ADD_LEFT_SHIFT = 20


def add_general(xa, za, sa, xb, zb, sb, s_out, z_out):
    """TFLite AddGeneral, mirroring nn_add_int8 in std/nn_int8.mlr.

    Both inputs are shifted left by 20 so the two rescalings onto a common
    intermediate scale lose no precision, summed, then rescaled once onto the
    output quantisation. This is INTEGER end to end -- an earlier version of
    this file did the add in floating point, which made it a good model of
    what quantized addition means and a bad model of what the kernel does.
    """
    twice = 2.0 * max(sa, sb)
    ma, sha = quantize_multiplier(sa / twice)
    mb, shb = quantize_multiplier(sb / twice)
    mo, sho = quantize_multiplier(twice / ((1 << ADD_LEFT_SHIFT) * s_out))
    va = (np.asarray(xa, dtype=np.int64) - za) * (1 << ADD_LEFT_SHIFT)
    vb = (np.asarray(xb, dtype=np.int64) - zb) * (1 << ADD_LEFT_SHIFT)
    s = requantize(va, ma, sha) + requantize(vb, mb, shb)
    return np.clip(requantize(s, mo, sho) + z_out, INT8_MIN, INT8_MAX)


# ---- quantisation parameter derivation ----

def act_qparams(lo, hi):
    """Asymmetric int8 scale/zero-point from an observed range."""
    lo = min(lo, 0.0); hi = max(hi, 0.0)
    if hi == lo:
        return 1.0, 0
    scale = (hi - lo) / 255.0
    zp = int(round(INT8_MIN - lo / scale))
    return scale, max(INT8_MIN, min(INT8_MAX, zp))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("ranges")
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--image")
    ap.add_argument("--input-i8", metavar="FILE",
                    help="already-quantised int8 input in NHWC flat order "
                         "(exactly the bytes the MLRift runner is handed). "
                         "Bypasses the float->int8 quantisation step so the "
                         "two pipelines provably start from the same tensor.")
    ap.add_argument("--dump", metavar="DIR",
                    help="write every tensor as raw int8 in NHWC flat order: "
                         "input.i8, node<NNN>.i8, out_<name>.i8. This is what "
                         "the MLRift runner is compared against, byte for byte.")
    ap.add_argument("--no-float-ref", action="store_true",
                    help="skip the float reference (much faster; the int8 "
                         "path does not depend on it)")
    args = ap.parse_args()

    import importlib.util, pathlib
    here = pathlib.Path(__file__).parent
    spec = importlib.util.spec_from_file_location("fr", here / "float_ref.py")
    fr = importlib.util.module_from_spec(spec); spec.loader.exec_module(fr)

    ranges = json.load(open(args.ranges))
    m = onnx.load(args.model); g = m.graph
    init = {t.name: numpy_helper.to_array(t).astype(np.float64) for t in g.initializer}

    inp_name = g.input[0].name
    s_in0, z_in0 = act_qparams(*ranges.get(inp_name, (0.0, 255.0)))
    dims = [d.dim_value for d in g.input[0].type.tensor_type.shape.dim]

    if args.input_i8:
        _, C, H, W = dims
        raw = np.frombuffer(open(args.input_i8, "rb").read(), dtype=np.int8)
        if raw.size != C * H * W:
            raise SystemExit(f"--input-i8: {raw.size} bytes, expected {C*H*W}")
        qin = raw.reshape(1, H, W, C).transpose(0, 3, 1, 2).astype(np.int64)
        x = (qin - z_in0) * s_in0          # for the float reference only
    elif args.image:
        from PIL import Image
        im = Image.open(args.image).convert("RGB").resize((args.size, args.size))
        x = np.asarray(im, dtype=np.float64).transpose(2, 0, 1)[None]
        qin = np.clip(np.rint(x / s_in0) + z_in0,
                      INT8_MIN, INT8_MAX).astype(np.int64)
    else:
        x = np.random.default_rng(args.seed).random(
            (1, 3, args.size, args.size)) * 255.0
        qin = np.clip(np.rint(x / s_in0) + z_in0,
                      INT8_MIN, INT8_MAX).astype(np.int64)

    float_out = None
    if not args.no_float_ref:
        float_out, _ = fr.run(args.model, x, collect_ranges=False)

    env_q = {inp_name: qin}
    env_s = {inp_name: (s_in0, z_in0)}

    # NHWC flat is the MLRift runner's memory layout. Tensors still in ONNX's
    # native NCHW get transposed on the way out; a tensor that has already been
    # through the graph's own Transpose(0,2,3,1) is [1,H,W,C] and its ravel IS
    # NHWC order, so transposing it again would be wrong. env_l tracks which.
    env_l = {inp_name: "nchw"}

    def nhwc_bytes(t, layout):
        t = np.asarray(t)
        if layout == "nchw" and t.ndim == 4:
            t = np.transpose(t, (0, 2, 3, 1))
        return t.astype(np.int8).ravel().tobytes()

    dumpdir = None
    if args.dump:
        import os
        dumpdir = args.dump
        os.makedirs(dumpdir, exist_ok=True)
        open(f"{dumpdir}/input.i8", "wb").write(
            nhwc_bytes(env_q[inp_name], "nchw"))

    def A(n):
        d = {}
        for a in n.attribute:
            if a.type == onnx.AttributeProto.INT:      d[a.name] = a.i
            elif a.type == onnx.AttributeProto.INTS:   d[a.name] = list(a.ints)
            elif a.type == onnx.AttributeProto.STRING: d[a.name] = a.s.decode()
        return d

    for ni, n in enumerate(g.node):
        a = A(n); op = n.op_type; out = n.output[0]
        if out in ranges:
            s_out, z_out = act_qparams(*ranges[out])
        else:
            s_out, z_out = 1.0, 0

        if op == "Conv":
            xi = env_q[n.input[0]]; s_i, z_i = env_s[n.input[0]]
            w = init[n.input[1]]
            b = init[n.input[2]] if len(n.input) > 2 else None
            flat = w.reshape(w.shape[0], -1)
            amax = np.abs(flat).max(axis=1); amax[amax == 0] = 1e-12
            s_w = amax / 127.0
            qw = np.rint(flat / s_w[:, None]).clip(-127, 127).reshape(w.shape)
            qb = None if b is None else np.rint(b / (s_i * s_w)).astype(np.int64)
            acc = fr.conv((xi - z_i).astype(np.float64), qw, None,
                          a.get("strides", [1, 1]), a.get("pads", [0, 0, 0, 0]),
                          a.get("group", 1), a.get("dilations", [1, 1]))
            acc = np.rint(acc).astype(np.int64)
            if qb is not None:
                acc += qb.reshape(1, -1, 1, 1)
            mult_shift = [quantize_multiplier((s_i * sw) / s_out) for sw in s_w]
            y = np.empty_like(acc)
            for c, (mu, sh) in enumerate(mult_shift):
                y[:, c] = requantize(acc[:, c], mu, sh)
            y = np.clip(y + z_out, INT8_MIN, INT8_MAX)
        elif op == "Relu":
            xi = env_q[n.input[0]]; s_i, z_i = env_s[n.input[0]]
            y = np.maximum(xi, z_i); s_out, z_out = s_i, z_i
        elif op == "Add":
            xa, xb = env_q[n.input[0]], env_q[n.input[1]]
            (sa, za), (sb, zb) = env_s[n.input[0]], env_s[n.input[1]]
            y = add_general(xa, za, sa, xb, zb, sb, s_out, z_out).astype(np.int64)
        elif op == "MaxPool":
            xi = env_q[n.input[0]]; s_i, z_i = env_s[n.input[0]]
            y = fr.maxpool(xi.astype(np.float64), a.get("kernel_shape", [2, 2]),
                           a.get("strides", [2, 2]),
                           a.get("pads", [0, 0, 0, 0])).astype(np.int64)
            s_out, z_out = s_i, z_i
        elif op == "Resize":
            xi = env_q[n.input[0]]; s_i, z_i = env_s[n.input[0]]
            src = xi.shape
            sizes = init.get(n.input[3]) if len(n.input) > 3 else None
            sc = init.get(n.input[2]) if len(n.input) > 2 else None
            if sc is None or len(np.atleast_1d(sc)) == 0:
                sc = np.array([sizes[i] / src[i] for i in range(4)])
            y = fr.resize_nearest(xi.astype(np.float64), sc).astype(np.int64)
            s_out, z_out = s_i, z_i
        elif op == "Sigmoid":
            xi = env_q[n.input[0]]; s_i, z_i = env_s[n.input[0]]
            deq = (xi - z_i) * s_i
            y = np.clip(np.rint((1/(1+np.exp(-deq))) / s_out) + z_out,
                        INT8_MIN, INT8_MAX).astype(np.int64)
        elif op == "Transpose":
            xi = env_q[n.input[0]]
            y = np.transpose(xi, a.get("perm")); s_out, z_out = env_s[n.input[0]]
        elif op == "Reshape":
            xi = env_q[n.input[0]]
            y = xi.reshape([int(s) for s in init[n.input[1]].astype(int).tolist()])
            s_out, z_out = env_s[n.input[0]]
        else:
            raise NotImplementedError(op)

        env_q[out] = y
        env_s[out] = (s_out, z_out)
        env_l[out] = "nhwc" if op == "Transpose" else env_l[n.input[0]]
        if dumpdir:
            open(f"{dumpdir}/node{ni:03d}.i8", "wb").write(
                nhwc_bytes(y, env_l[out]))

    if dumpdir:
        for o in g.output:
            open(f"{dumpdir}/out_{o.name}.i8", "wb").write(
                nhwc_bytes(env_q[o.name], env_l[o.name]))
        print(f"  dumped int8 NHWC tensors -> {dumpdir}/")

    if float_out is None:
        return
    print(f"  {'output':<10} {'max abs err':>12} {'rel to range':>13}")
    worst = 0.0
    for o in g.output:
        q = env_q[o.name]; s, z = env_s[o.name]
        deq = (q - z) * s
        f = np.asarray(float_out[o.name], dtype=np.float64)
        err = np.abs(deq - f).max()
        span = max(f.max() - f.min(), 1e-9)
        worst = max(worst, err / span)
        print(f"  {o.name:<10} {err:12.5f} {err/span*100:12.2f}%")
    print(f"\n  worst error as % of output range: {worst*100:.2f}%")


if __name__ == "__main__":
    sys.exit(main())
