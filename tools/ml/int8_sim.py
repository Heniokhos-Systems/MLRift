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
    """float multiplier -> (int32 multiplier in [2^30, 2^31), right shift)."""
    if m == 0.0:
        return 0, 0
    s = 0
    while m < 0.5:
        m *= 2.0; s += 1
    while m >= 1.0:
        m /= 2.0; s -= 1
    q = int(round(m * (1 << 31)))
    if q == (1 << 31):
        q //= 2; s -= 1
    # srdhm() DOUBLES (a*b*2 >> 31), so with q = m_norm * 2^31 it returns
    # 2*a*m_norm. We want a*m_norm*2^-s, hence one extra right shift.
    return q, s + 1


def srdhm(a, b):
    """Saturating rounding doubling high mul, int32 x int32 -> int32."""
    a = np.asarray(a, dtype=np.int64); b = np.int64(b)
    overflow = (a == -(1 << 31)) & (b == -(1 << 31))
    ab = a * b * 2
    nudge = np.where(ab >= 0, 1 << 30, 1 - (1 << 30))
    out = (ab + nudge) >> 31
    return np.where(overflow, (1 << 31) - 1, out).astype(np.int64)


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
    left = max(0, -shift)
    right = max(0, shift)
    if left:
        acc = np.asarray(acc, dtype=np.int64) * (1 << left)
    return rdbpot(srdhm(acc, mult), right)


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
    args = ap.parse_args()

    import importlib.util, pathlib
    here = pathlib.Path(__file__).parent
    spec = importlib.util.spec_from_file_location("fr", here / "float_ref.py")
    fr = importlib.util.module_from_spec(spec); spec.loader.exec_module(fr)

    ranges = json.load(open(args.ranges))
    m = onnx.load(args.model); g = m.graph
    init = {t.name: numpy_helper.to_array(t).astype(np.float64) for t in g.initializer}

    if args.image:
        from PIL import Image
        im = Image.open(args.image).convert("RGB").resize((args.size, args.size))
        x = np.asarray(im, dtype=np.float64).transpose(2, 0, 1)[None]
    else:
        x = np.random.default_rng(args.seed).random(
            (1, 3, args.size, args.size)) * 255.0

    float_out, _ = fr.run(args.model, x, collect_ranges=False)

    inp_name = g.input[0].name
    s_in0, z_in0 = act_qparams(*ranges.get(inp_name, (0.0, 255.0)))
    env_q = {inp_name: np.clip(np.rint(x / s_in0) + z_in0,
                               INT8_MIN, INT8_MAX).astype(np.int64)}
    env_s = {inp_name: (s_in0, z_in0)}

    def A(n):
        d = {}
        for a in n.attribute:
            if a.type == onnx.AttributeProto.INT:      d[a.name] = a.i
            elif a.type == onnx.AttributeProto.INTS:   d[a.name] = list(a.ints)
            elif a.type == onnx.AttributeProto.STRING: d[a.name] = a.s.decode()
        return d

    for n in g.node:
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
            y = np.clip(np.rint(((xa - za) * sa + (xb - zb) * sb) / s_out) + z_out,
                        INT8_MIN, INT8_MAX).astype(np.int64)
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
