#!/usr/bin/env python3
"""Independent reference for std/nn_int8.mlr.

This is NOT a port of the MLRift code — it is written directly from the
gemmlowp / TFLite-Micro definitions, in ordinary Python `int` arithmetic with
explicit 32-bit truncation. In particular it does the 64-bit product with a
plain `a * b`, exactly the operation the MLRift kernels are forbidden from
using and have to synthesise from 16-bit half-products. If the two agree, the
software multiply-high and the whole requantization chain are correct.

Emits the same line sequence as tests/nn/nn_test_core.mlr. Compare with diff.
"""

INT32_MIN = -(1 << 31)
INT32_MAX = (1 << 31) - 1


# --------------------------------------------------------------------------
# gemmlowp fixed-point primitives (reference definitions)
# --------------------------------------------------------------------------

def srdhm(a, b):
    """SaturatingRoundingDoublingHighMul."""
    if a == INT32_MIN and b == INT32_MIN:
        return INT32_MAX
    ab = a * b                      # true 64-bit product
    nudge = (1 << 30) if ab >= 0 else (1 - (1 << 30))
    s = ab + nudge
    # C integer division truncates toward zero.
    q = abs(s) // (1 << 31)
    if s < 0:
        q = -q
    return q


def umulhi32(a, b):
    """High 32 bits of a 32x32 unsigned product."""
    return ((a & 0xFFFFFFFF) * (b & 0xFFFFFFFF)) >> 32


def umullo32(a, b):
    return ((a & 0xFFFFFFFF) * (b & 0xFFFFFFFF)) & 0xFFFFFFFF


def mulhi_i32(a, b):
    """High 32 bits of a 32x32 signed product, as int32.

    The high word of the two's-complement 64-bit product read as signed is
    exactly floor(a*b / 2^32), and Python's >> is an arithmetic shift.
    """
    return (a * b) >> 32


def rdbpot(x, exponent):
    """RoundingDivideByPOT — round half away from zero."""
    if exponent == 0:
        return x
    if exponent > 31:
        exponent = 31
    mask = (1 << exponent) - 1
    remainder = x & mask            # Python & on negatives == two's complement
    threshold = (mask >> 1) + (1 if x < 0 else 0)
    r = x >> exponent               # Python >> is arithmetic
    if remainder > threshold:
        r += 1
    return r


def to_i32(v):
    v &= 0xFFFFFFFF
    return v - (1 << 32) if v & 0x80000000 else v


def mul_by_qm(x, multiplier, shift):
    left_shift = shift if shift > 0 else 0
    right_shift = -shift if shift < 0 else 0
    if left_shift > 31:
        left_shift = 31
    xs = to_i32(x << left_shift) if left_shift > 0 else x
    return rdbpot(srdhm(xs, multiplier), right_shift)


def clamp(v, lo, hi):
    return lo if v < lo else (hi if v > hi else v)


def requantize(acc, multiplier, shift):
    return clamp(mul_by_qm(acc, multiplier, shift), -128, 127)


def requantize_zp(acc, multiplier, shift, out_zp, act_min, act_max):
    return clamp(mul_by_qm(acc, multiplier, shift) + out_zp, act_min, act_max)


def sdiv_trunc(a, b):
    if b == 0:
        return 0
    q = abs(a) // b
    return q if a >= 0 else -q


def out_dim(in_dim, k, stride, pad):
    eff = in_dim + 2 * pad
    if eff < k:
        return 0
    return (eff - k) // stride + 1


# --------------------------------------------------------------------------
# Deterministic test-vector generator — bit-identical to nn_test_core.mlr
# --------------------------------------------------------------------------

class Rng:
    def __init__(self, seed):
        self.s = seed

    def r32(self):
        self.s = (self.s * 1103515245 + 12345) & 0xFFFFFFFF
        return self.s

    def i8(self):
        return ((self.r32() >> 16) & 0xFF) - 128

    def r30(self):
        return (self.r32() >> 2) & 0x3FFFFFFF

    def mult(self):
        # TFLite multipliers are normalised into [2^30, 2^31).
        return (1 << 30) + (self.r30() >> 1)

    def shift(self):
        return -(int(self.r32() >> 28) % 12) - 1

    def bias(self):
        return to_i32(self.r32()) >> 20


OUT = []


def emit(v):
    OUT.append(str(v))


def fill_i8(rng, n):
    return [rng.i8() for _ in range(n)]


# --------------------------------------------------------------------------
# Kernels (reference implementations, NHWC / OHWI as documented)
# --------------------------------------------------------------------------

def conv2d(inp, in_h, in_w, in_c, in_zp, w, kh, kw, bias, mult, shift, q_stride,
           out_c, out_zp, stride_h, stride_w, pad_h, pad_w, act_min, act_max):
    out_h = out_dim(in_h, kh, stride_h, pad_h)
    out_w = out_dim(in_w, kw, stride_w, pad_w)
    out = []
    for oy in range(out_h):
        for ox in range(out_w):
            for oc in range(out_c):
                acc = bias[oc] if bias is not None else 0
                for i in range(kh):
                    iy = oy * stride_h + i - pad_h
                    if iy < 0 or iy >= in_h:
                        continue
                    for j in range(kw):
                        ix = ox * stride_w + j - pad_w
                        if ix < 0 or ix >= in_w:
                            continue
                        for ic in range(in_c):
                            iv = inp[(iy * in_w + ix) * in_c + ic] - in_zp
                            wv = w[((oc * kh + i) * kw + j) * in_c + ic]
                            acc += iv * wv
                qi = oc * q_stride
                out.append(requantize_zp(acc, mult[qi], shift[qi], out_zp,
                                         act_min, act_max))
    return out


def depthwise(inp, in_h, in_w, in_c, in_zp, w, kh, kw, bias, mult, shift,
              q_stride, out_zp, stride_h, stride_w, pad_h, pad_w, depth_mult,
              act_min, act_max):
    out_h = out_dim(in_h, kh, stride_h, pad_h)
    out_w = out_dim(in_w, kw, stride_w, pad_w)
    out_c = in_c * depth_mult
    out = [0] * (out_h * out_w * out_c)
    for oy in range(out_h):
        for ox in range(out_w):
            for ic in range(in_c):
                for m in range(depth_mult):
                    oc = ic * depth_mult + m
                    acc = bias[oc] if bias is not None else 0
                    for i in range(kh):
                        iy = oy * stride_h + i - pad_h
                        if iy < 0 or iy >= in_h:
                            continue
                        for j in range(kw):
                            ix = ox * stride_w + j - pad_w
                            if ix < 0 or ix >= in_w:
                                continue
                            iv = inp[(iy * in_w + ix) * in_c + ic] - in_zp
                            wv = w[(i * kw + j) * out_c + oc]
                            acc += iv * wv
                    qi = oc * q_stride
                    out[(oy * out_w + ox) * out_c + oc] = requantize_zp(
                        acc, mult[qi], shift[qi], out_zp, act_min, act_max)
    return out


def gemv(x, k_dim, x_zp, w, n_dim, bias, mult, shift, q_stride, out_zp,
         act_min, act_max):
    out = []
    for n in range(n_dim):
        acc = bias[n] if bias is not None else 0
        for k in range(k_dim):
            acc += (x[k] - x_zp) * w[n * k_dim + k]
        qi = n * q_stride
        out.append(requantize_zp(acc, mult[qi], shift[qi], out_zp,
                                 act_min, act_max))
    return out


def gemm(a, m_dim, k_dim, a_zp, b, n_dim, bias, mult, shift, q_stride, out_zp,
         act_min, act_max):
    out = []
    for r in range(m_dim):
        out += gemv(a[r * k_dim:], k_dim, a_zp, b, n_dim, bias, mult, shift,
                    q_stride, out_zp, act_min, act_max)
    return out


def maxpool(inp, in_h, in_w, in_c, kh, kw, stride_h, stride_w, pad_h, pad_w,
            act_min, act_max):
    out_h = out_dim(in_h, kh, stride_h, pad_h)
    out_w = out_dim(in_w, kw, stride_w, pad_w)
    out = []
    for oy in range(out_h):
        for ox in range(out_w):
            for c in range(in_c):
                best = -128
                for i in range(kh):
                    iy = oy * stride_h + i - pad_h
                    if iy < 0 or iy >= in_h:
                        continue
                    for j in range(kw):
                        ix = ox * stride_w + j - pad_w
                        if ix < 0 or ix >= in_w:
                            continue
                        v = inp[(iy * in_w + ix) * in_c + c]
                        if v > best:
                            best = v
                out.append(clamp(best, act_min, act_max))
    return out


def avgpool(inp, in_h, in_w, in_c, kh, kw, stride_h, stride_w, pad_h, pad_w,
            act_min, act_max):
    out_h = out_dim(in_h, kh, stride_h, pad_h)
    out_w = out_dim(in_w, kw, stride_w, pad_w)
    out = []
    for oy in range(out_h):
        for ox in range(out_w):
            for c in range(in_c):
                acc = 0
                count = 0
                for i in range(kh):
                    iy = oy * stride_h + i - pad_h
                    if iy < 0 or iy >= in_h:
                        continue
                    for j in range(kw):
                        ix = ox * stride_w + j - pad_w
                        if ix < 0 or ix >= in_w:
                            continue
                        acc += inp[(iy * in_w + ix) * in_c + c]
                        count += 1
                v = 0
                if count:
                    half = count // 2
                    num = acc + half if acc > 0 else acc - half
                    v = sdiv_trunc(num, count)
                out.append(clamp(v, act_min, act_max))
    return out


def add_int8(a, a_zp, a_m, a_s, b, b_zp, b_m, b_s, out_zp, o_m, o_s, n,
             left_shift, act_min, act_max):
    out = []
    for i in range(n):
        av = to_i32((a[i] - a_zp) << left_shift)
        bv = to_i32((b[i] - b_zp) << left_shift)
        s = mul_by_qm(av, a_m, a_s) + mul_by_qm(bv, b_m, b_s)
        out.append(clamp(mul_by_qm(s, o_m, o_s) + out_zp, act_min, act_max))
    return out


# --------------------------------------------------------------------------
# The test programme — must mirror nn_test_core.mlr statement for statement
# --------------------------------------------------------------------------

def main():
    # ---- S1: srdhm fixed edge cases ----------------------------------
    fixed = [
        (1 << 30, 1 << 30),
        (-(1 << 30), 1 << 30),
        (INT32_MAX, INT32_MAX),
        (INT32_MIN, INT32_MIN),
        (INT32_MIN, INT32_MAX),
        (INT32_MAX, INT32_MIN),
        (0, 12345),
        (12345, 0),
        (1, 1),
        (-1, 1),
        (-1, -1),
        (1, -1),
        (3, 1431655765),
        (-3, 1431655765),
        (12345678, -87654321),
        (INT32_MAX, 1),
        (INT32_MIN, 1),
        (INT32_MIN, -1),
        (0x7FFFFFFF, 0x40000000),
        ((1 << 30) - 1, (1 << 30) - 1),
    ]
    for a, b in fixed:
        emit(srdhm(a, b))

    rng = Rng(12345)
    for _ in range(64):
        a = to_i32(rng.r32())
        b = to_i32(rng.r32())
        emit(srdhm(a, b))

    # ---- S1b: the software multiply-high primitives -------------------
    # 0xFFFFFFFF^2 is the case that breaks a naive 16-bit-halves recombination.
    umul_pairs = [
        (0xFFFFFFFF, 0xFFFFFFFF),
        (0x12345678, 0x9ABCDEF0),
        (1000000, 1000000),
        (0, 0xFFFFFFFF),
        (1, 0xFFFFFFFF),
        (0x80000000, 0x80000000),
        (0xFFFF, 0xFFFF),
        (0x10000, 0x10000),
        (0xFFFFFFFF, 1),
        (0x7FFFFFFF, 0x80000001),
    ]
    for a, b in umul_pairs:
        emit(umulhi32(a, b))
        emit(umullo32(a, b))
    for a, b in fixed:
        emit(mulhi_i32(a, b))
    rng1b = Rng(12345)
    for _ in range(64):
        a = to_i32(rng1b.r32())
        b = to_i32(rng1b.r32())
        emit(mulhi_i32(a, b))

    # ---- S2: rdbpot, ties and negatives ------------------------------
    fixed2 = [
        (1, 1), (-1, 1), (2, 1), (-2, 1), (3, 1), (-3, 1),
        (4, 2), (-4, 2), (5, 2), (-5, 2), (6, 2), (-6, 2), (7, 2), (-7, 2),
        (0, 5), (1, 31), (-1, 31), (INT32_MAX, 31), (INT32_MIN, 31),
        (12345, 3), (-12345, 3), (128, 8), (-128, 8), (192, 8), (-192, 8),
    ]
    for x, e in fixed2:
        emit(rdbpot(x, e))

    rng2 = Rng(777)
    for _ in range(48):
        x = to_i32(rng2.r32())
        e = rng2.r32() % 32
        emit(rdbpot(x, e))

    # ---- S3: mul_by_qm ------------------------------------------------
    rng3 = Rng(4242)
    for _ in range(64):
        x = to_i32(rng3.r32()) >> 8
        m = rng3.mult()
        s = rng3.shift()
        emit(mul_by_qm(x, m, s))
    # positive (left) shifts
    for _ in range(16):
        x = to_i32(rng3.r32()) >> 20
        m = rng3.mult()
        s = int(rng3.r32() >> 28) % 4 + 1
        emit(mul_by_qm(x, m, s))

    # ---- S4: requantize, saturation at +-127 --------------------------
    rng4 = Rng(99991)
    for _ in range(48):
        acc = to_i32(rng4.r32()) >> 10
        m = rng4.mult()
        s = rng4.shift()
        emit(requantize(acc, m, s))
    # forced saturation both directions
    for acc in (1 << 30, -(1 << 30), INT32_MAX, INT32_MIN, 1 << 20, -(1 << 20)):
        emit(requantize(acc, 1 << 30, -1))

    # ---- S5: requantize_zp with activation ranges ---------------------
    rng5 = Rng(31337)
    for _ in range(32):
        acc = to_i32(rng5.r32()) >> 12
        m = rng5.mult()
        s = rng5.shift()
        emit(requantize_zp(acc, m, s, -5, -128, 127))
    for _ in range(32):
        acc = to_i32(rng5.r32()) >> 12
        m = rng5.mult()
        s = rng5.shift()
        emit(requantize_zp(acc, m, s, 7, 7, 127))   # fused ReLU

    # ---- S6: conv2d 5x5x2, k3x3, s1, p1, out_c 3, per channel ---------
    r = Rng(2026)
    inp = fill_i8(r, 5 * 5 * 2)
    w = fill_i8(r, 3 * 3 * 3 * 2)
    bias = [r.bias() for _ in range(3)]
    mult = [r.mult() for _ in range(3)]
    shift = [r.shift() for _ in range(3)]
    for v in conv2d(inp, 5, 5, 2, -3, w, 3, 3, bias, mult, shift, 1, 3, 11,
                    1, 1, 1, 1, -128, 127):
        emit(v)

    # ---- S7: conv2d 6x6x2, k3x3, s2, p0, out_c 2, per tensor ----------
    r = Rng(60606)
    inp = fill_i8(r, 6 * 6 * 2)
    w = fill_i8(r, 2 * 3 * 3 * 2)
    bias = [r.bias() for _ in range(2)]
    mult = [r.mult()]
    shift = [r.shift()]
    for v in conv2d(inp, 6, 6, 2, 5, w, 3, 3, bias, mult, shift, 0, 2, -8,
                    2, 2, 0, 0, -128, 127):
        emit(v)

    # ---- S8a: depthwise 5x5x3, k3x3, s1, p1, dm 1 ---------------------
    r = Rng(808)
    inp = fill_i8(r, 5 * 5 * 3)
    w = fill_i8(r, 3 * 3 * 3)
    bias = [r.bias() for _ in range(3)]
    mult = [r.mult() for _ in range(3)]
    shift = [r.shift() for _ in range(3)]
    for v in depthwise(inp, 5, 5, 3, -12, w, 3, 3, bias, mult, shift, 1, 4,
                       1, 1, 1, 1, 1, -128, 127):
        emit(v)

    # ---- S8b: depthwise 4x4x2, k3x3, s2, p1, dm 2 ---------------------
    r = Rng(80808)
    inp = fill_i8(r, 4 * 4 * 2)
    w = fill_i8(r, 3 * 3 * 4)
    bias = [r.bias() for _ in range(4)]
    mult = [r.mult() for _ in range(4)]
    shift = [r.shift() for _ in range(4)]
    for v in depthwise(inp, 4, 4, 2, 0, w, 3, 3, bias, mult, shift, 1, -2,
                       2, 2, 1, 1, 2, -128, 127):
        emit(v)

    # ---- S9: pointwise 4x4x3 -> 4 channels ----------------------------
    r = Rng(1111)
    inp = fill_i8(r, 4 * 4 * 3)
    w = fill_i8(r, 4 * 3)
    bias = [r.bias() for _ in range(4)]
    mult = [r.mult() for _ in range(4)]
    shift = [r.shift() for _ in range(4)]
    for v in conv2d(inp, 4, 4, 3, 6, w, 1, 1, bias, mult, shift, 1, 4, 0,
                    1, 1, 0, 0, -128, 127):
        emit(v)

    # ---- S10: gemv k=16 n=5 -------------------------------------------
    r = Rng(51515)
    x = fill_i8(r, 16)
    w = fill_i8(r, 5 * 16)
    bias = [r.bias() for _ in range(5)]
    mult = [r.mult() for _ in range(5)]
    shift = [r.shift() for _ in range(5)]
    for v in gemv(x, 16, -9, w, 5, bias, mult, shift, 1, 3, -128, 127):
        emit(v)

    # ---- S11: gemm m=3 k=8 n=4 ----------------------------------------
    r = Rng(383838)
    a = fill_i8(r, 3 * 8)
    b = fill_i8(r, 4 * 8)
    bias = [r.bias() for _ in range(4)]
    mult = [r.mult() for _ in range(4)]
    shift = [r.shift() for _ in range(4)]
    for v in gemm(a, 3, 8, 2, b, 4, bias, mult, shift, 1, -1, -128, 127):
        emit(v)

    # ---- S12: maxpool -------------------------------------------------
    r = Rng(1234567)
    inp = fill_i8(r, 5 * 5 * 2)
    for v in maxpool(inp, 5, 5, 2, 2, 2, 2, 2, 0, 0, -128, 127):
        emit(v)
    for v in maxpool(inp, 5, 5, 2, 3, 3, 2, 2, 1, 1, -128, 127):
        emit(v)

    # ---- S13: avgpool -------------------------------------------------
    for v in avgpool(inp, 5, 5, 2, 2, 2, 2, 2, 0, 0, -128, 127):
        emit(v)
    for v in avgpool(inp, 5, 5, 2, 3, 3, 2, 2, 1, 1, -128, 127):
        emit(v)

    # ---- S14: relu / relu6 --------------------------------------------
    r = Rng(20260724)
    vals = fill_i8(r, 24)
    for v in vals:
        emit(v if v >= -7 else -7)
    for v in vals:
        emit(clamp(v, 3, 100))

    # ---- S15: add_int8 ------------------------------------------------
    r = Rng(555)
    a = fill_i8(r, 24)
    b = fill_i8(r, 24)
    for v in add_int8(a, -2, 1 << 30, -1, b, 4, 1 << 30, -2, 6, 1 << 30, -20,
                      24, 20, -128, 127):
        emit(v)

    # ---- S16: end-to-end detector-shaped stack -------------------------
    # conv 3x3/s2 -> relu -> depthwise 3x3 -> pointwise 1x1 -> global avgpool.
    # SYNTHETIC weights (pseudo-random). This is the SHAPE of a tiny
    # MobileNet-style face detector head, not a trained detector.
    r = Rng(777001)
    img = fill_i8(r, 8 * 8 * 1)                    # 8x8 grayscale
    wc = fill_i8(r, 4 * 3 * 3 * 1)
    bc = [r.bias() for _ in range(4)]
    mc = [r.mult() for _ in range(4)]
    sc = [r.shift() for _ in range(4)]
    c1 = conv2d(img, 8, 8, 1, -5, wc, 3, 3, bc, mc, sc, 1, 4, 0,
                2, 2, 1, 1, -128, 127)             # -> 4x4x4
    c1 = [v if v >= 0 else 0 for v in c1]          # ReLU, zero point 0

    wd = fill_i8(r, 3 * 3 * 4)
    bd = [r.bias() for _ in range(4)]
    md = [r.mult() for _ in range(4)]
    sd = [r.shift() for _ in range(4)]
    d1 = depthwise(c1, 4, 4, 4, 0, wd, 3, 3, bd, md, sd, 1, 0,
                   1, 1, 1, 1, 1, -128, 127)       # -> 4x4x4

    wp = fill_i8(r, 2 * 4)
    bp = [r.bias() for _ in range(2)]
    mp = [r.mult() for _ in range(2)]
    sp = [r.shift() for _ in range(2)]
    p1 = conv2d(d1, 4, 4, 4, 0, wp, 1, 1, bp, mp, sp, 1, 2, 0,
                1, 1, 0, 0, -128, 127)             # -> 4x4x2

    g = avgpool(p1, 4, 4, 2, 4, 4, 4, 4, 0, 0, -128, 127)   # -> 1x1x2

    for v in c1:
        emit(v)
    for v in d1:
        emit(v)
    for v in p1:
        emit(v)
    for v in g:
        emit(v)

    # ---- S17: nn_requantize_zp == nn_requantize_zp_ref -----------------
    # The target compares its flattened requantization against the composed
    # definition itself and emits the disagreement count. There is nothing for
    # Python to recompute here: the assertion is that the two MLRift forms are
    # the same function, so the reference value is 0.
    emit(0)

    print('\n'.join(OUT))


if __name__ == '__main__':
    main()
