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


# ===========================================================================
# --emit-binary : the flat .krnn container std/nn_model.mlr walks
# ===========================================================================
# The layout is specified once, in tools/ml/krnn.py, and repeated verbatim in
# std/nn_model.mlr. Everything below is offline work: shape inference in NHWC,
# quantisation-parameter derivation, arena allocation, and packing. The target
# does none of it.
#
# Every arithmetic decision here is deliberately identical to int8_sim.py --
# same weight scales, same activation scales, same multiplier normalisation --
# because the whole point of the exercise is that the two agree BIT-EXACTLY.
# quantize_multiplier is imported from int8_sim.py rather than reimplemented
# so it cannot drift.

import importlib.util as _ilu, pathlib as _pl

_HERE = _pl.Path(__file__).parent


def _load(name):
    spec = _ilu.spec_from_file_location(name, _HERE / f"{name}.py")
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _sim():
    return _load("int8_sim")


def out_dim(in_dim, k, stride, pad):
    return (in_dim + 2 * pad - k) // stride + 1


def scale_q16(s):
    v = int(round(s * 65536.0))
    return max(0, min(0xFFFFFFFF, v))


class Buf:
    __slots__ = ("bid", "size", "off", "first_use", "last_use", "is_output")

    def __init__(self, bid, size, first_use):
        self.bid = bid
        self.size = size
        self.off = None
        self.first_use = first_use
        self.last_use = first_use
        self.is_output = False


def plan_arena(bufs, n_nodes, align=4):
    """TFLite-Micro-style greedy-by-size offline memory planner.

    Buffers are placed largest first at the lowest offset that does not
    collide with an already-placed buffer whose lifetime overlaps. This beats
    a chronological first-fit free-list badly on CNNs, because the big early
    feature maps get to sit at offset 0 and everything small is tucked into
    the gaps rather than fragmenting the front of the arena.

    Returns the peak byte count. Runs on the host; the target only ever sees
    the resulting constants.
    """
    def rounded(n):
        return max(align, (n + align - 1) // align * align)

    order = sorted(bufs, key=lambda b: (-b.size, b.first_use))
    placed = []
    for b in order:
        need = rounded(b.size)
        end = b.last_use
        if b.is_output:
            end = n_nodes
        occupied = []
        for p in placed:
            p_end = n_nodes if p.is_output else p.last_use
            if p.first_use <= end and b.first_use <= p_end:
                occupied.append((p.off, p.off + rounded(p.size)))
        occupied.sort()
        off = 0
        for lo, hi in occupied:
            if off + need <= lo:
                break
            if hi > off:
                off = hi
        b.off = off
        placed.append(b)
    return max((b.off + rounded(b.size)) for b in bufs) if bufs else 0


def emit_binary(model_path, ranges_path, out_path, report=True):
    import krnn as K
    sim = _sim()
    quantize_multiplier = sim.quantize_multiplier

    m = onnx.load(model_path)
    g = m.graph
    init = {t.name: numpy_helper.to_array(t).astype(np.float64)
            for t in g.initializer}
    ranges = json.load(open(ranges_path))

    in_name = g.input[0].name
    idims = [d.dim_value for d in g.input[0].type.tensor_type.shape.dim]
    _, in_c0, in_h0, in_w0 = idims
    s_in0, z_in0 = sim.act_qparams(*ranges.get(in_name, (0.0, 255.0)))

    out_names = [o.name for o in g.output]
    graph_outputs = set(out_names)

    # ---- how many times is each tensor consumed? (drives in-place Relu) ----
    use_count = {}
    for n in g.node:
        for t in n.input:
            use_count[t] = use_count.get(t, 0) + 1
    for t in out_names:
        use_count[t] = use_count.get(t, 0) + 1

    # ---- pass A: shapes, quant params, blob, buffer assignment ------------
    shape = {in_name: (in_h0, in_w0, in_c0)}
    qp = {in_name: (s_in0, z_in0)}
    bufs = []
    buf_of = {}

    cur_idx = [-1]

    def new_buf(name, nbytes):
        b = Buf(len(bufs), nbytes, cur_idx[0])
        bufs.append(b)
        buf_of[name] = b
        return b

    def alias(name, src):
        buf_of[name] = buf_of[src]

    new_buf(in_name, in_h0 * in_w0 * in_c0)

    blob = K.Blob()
    layers = []
    notes = []
    # nn_mul_by_qm masks the pre-multiply left shift to 32 bits; int8_sim.py
    # does it in 64. A positive shift is therefore the one case where the two
    # could legitimately disagree, so it is recorded rather than assumed away.
    left_shift_warn = []

    for idx, n in enumerate(g.node):
        cur_idx[0] = idx
        a = attrs(n)
        op = n.op_type
        oname = n.output[0]
        rec = K.blank_layer()
        ih, iw, ic = shape[n.input[0]]
        s_i, z_i = qp[n.input[0]]
        rec["in_h"], rec["in_w"], rec["in_c"] = ih, iw, ic
        rec["in_zp"] = z_i

        if oname in ranges:
            s_out, z_out = sim.act_qparams(*ranges[oname])
        else:
            s_out, z_out = 1.0, 0

        if op == "Conv":
            w = init[n.input[1]]
            b = init[n.input[2]] if len(n.input) > 2 else None
            O, IC, KH, KW = w.shape
            groups = a.get("group", 1)
            sh, sw = a.get("strides", [1, 1])
            pads = a.get("pads", [0, 0, 0, 0])
            dil = a.get("dilations", [1, 1])
            if dil != [1, 1]:
                raise SystemExit(f"node {idx}: dilation {dil} unsupported")
            if pads[0] != pads[2] or pads[1] != pads[3]:
                raise SystemExit(f"node {idx}: asymmetric pads {pads}")
            ph, pw = pads[0], pads[1]

            # weight quantisation -- byte-for-byte what int8_sim.py does
            flat = w.reshape(O, -1)
            amax = np.abs(flat).max(axis=1)
            amax[amax == 0] = 1e-12
            s_w = amax / 127.0
            qw = np.rint(flat / s_w[:, None]).clip(-127, 127).reshape(w.shape)
            qb = None if b is None else np.rint(b / (s_i * s_w))

            depthwise = groups > 1 and groups == O and IC == 1
            if depthwise:
                rec["op"] = K.OP_DEPTHWISE
                # ONNX (O,1,KH,KW) -> 1HWC [kh][kw][out_c]
                packed = np.transpose(qw[:, 0], (1, 2, 0))
                rec["depth_mult"] = O // ic
            else:
                if groups != 1:
                    raise SystemExit(f"node {idx}: grouped conv g={groups} "
                                     f"is not depthwise and is unsupported")
                rec["op"] = K.OP_POINTWISE if (KH == 1 and KW == 1 and
                                               sh == 1 and sw == 1 and
                                               ph == 0 and pw == 0) \
                    else K.OP_CONV2D
                # ONNX OIHW -> OHWI
                packed = np.transpose(qw, (0, 2, 3, 1))

            oh = out_dim(ih, KH, sh, ph)
            ow = out_dim(iw, KW, sw, pw)
            rec.update(out_h=oh, out_w=ow, out_c=O, kh=KH, kw=KW,
                       stride_h=sh, stride_w=sw, pad_h=ph, pad_w=pw,
                       q_stride=1, out_zp=z_out, n_elem=oh * ow * O)
            rec["w_off"] = blob.add_i8(packed.ravel())
            if qb is not None:
                rec["bias_off"] = blob.add_i32(qb)
            mults, shifts = [], []
            for sw_c in s_w:
                q, s = quantize_multiplier((s_i * sw_c) / s_out)
                mults.append(q)
                shifts.append(s)
                if s > 0:
                    left_shift_warn.append(idx)
            rec["mult_off"] = blob.add_i32(mults)
            rec["shift_off"] = blob.add_i32(shifts)
            shape[oname] = (oh, ow, O)
            qp[oname] = (s_out, z_out)
            new_buf(oname, oh * ow * O)

        elif op == "Relu":
            rec["op"] = K.OP_RELU
            rec.update(out_h=ih, out_w=iw, out_c=ic, out_zp=z_i,
                       n_elem=ih * iw * ic)
            shape[oname] = (ih, iw, ic)
            qp[oname] = (s_i, z_i)
            # In place when the producer's value is not needed afterwards.
            if use_count.get(n.input[0], 0) == 1 and \
                    n.input[0] not in graph_outputs and n.input[0] != in_name:
                alias(oname, n.input[0])
            else:
                new_buf(oname, ih * iw * ic)
                notes.append(f"node {idx} Relu: out-of-place "
                             f"({n.input[0]} has other consumers)")

        elif op == "MaxPool":
            kh, kw = a.get("kernel_shape", [2, 2])
            sh, sw = a.get("strides", [2, 2])
            pads = a.get("pads", [0, 0, 0, 0])
            if pads[0] != pads[2] or pads[1] != pads[3]:
                raise SystemExit(f"node {idx}: asymmetric pool pads {pads}")
            ph, pw = pads[0], pads[1]
            oh, ow = out_dim(ih, kh, sh, ph), out_dim(iw, kw, sw, pw)
            rec["op"] = K.OP_MAXPOOL
            rec.update(out_h=oh, out_w=ow, out_c=ic, kh=kh, kw=kw,
                       stride_h=sh, stride_w=sw, pad_h=ph, pad_w=pw,
                       out_zp=z_i, n_elem=oh * ow * ic)
            shape[oname] = (oh, ow, ic)
            qp[oname] = (s_i, z_i)
            new_buf(oname, oh * ow * ic)

        elif op == "Add":
            bh, bw, bc = shape[n.input[1]]
            s_b, z_b = qp[n.input[1]]
            if (bh, bw, bc) != (ih, iw, ic):
                raise SystemExit(f"node {idx}: Add broadcast unsupported")
            LEFT = 20
            twice = 2.0 * max(s_i, s_b)
            am, ash = quantize_multiplier(s_i / twice)
            bm, bsh = quantize_multiplier(s_b / twice)
            om, osh = quantize_multiplier(twice / ((1 << LEFT) * s_out))
            rec["op"] = K.OP_ADD
            rec.update(out_h=ih, out_w=iw, out_c=ic, out_zp=z_out,
                       n_elem=ih * iw * ic, in1_zp=z_b, left_shift=LEFT,
                       a_mult=am, a_shift=ash, b_mult=bm, b_shift=bsh,
                       o_mult=om, o_shift=osh)
            shape[oname] = (ih, iw, ic)
            qp[oname] = (s_out, z_out)
            new_buf(oname, ih * iw * ic)

        elif op == "Resize":
            mode = a.get("mode", "nearest")
            nm = a.get("nearest_mode", "floor")
            ctm = a.get("coordinate_transformation_mode", "asymmetric")
            if mode != "nearest" or nm != "floor" or ctm != "asymmetric":
                raise SystemExit(f"node {idx}: Resize {mode}/{nm}/{ctm} "
                                 f"unsupported")
            sc = init.get(n.input[2]) if len(n.input) > 2 else None
            if sc is None or len(np.atleast_1d(sc)) == 0:
                sizes = init[n.input[3]]
                sc = np.array([sizes[i] / [1, ic, ih, iw][i] for i in range(4)])
            fh, fw = float(sc[2]), float(sc[3])
            if fh != int(fh) or fw != int(fw) or fh < 1 or fw < 1:
                raise SystemExit(f"node {idx}: non-integer Resize scale "
                                 f"{fh}x{fw}")
            fh, fw = int(fh), int(fw)
            oh, ow = ih * fh, iw * fw
            rec["op"] = K.OP_RESIZE_NN
            rec.update(out_h=oh, out_w=ow, out_c=ic, stride_h=fh, stride_w=fw,
                       out_zp=z_i, n_elem=oh * ow * ic)
            shape[oname] = (oh, ow, ic)
            qp[oname] = (s_i, z_i)
            new_buf(oname, oh * ow * ic)

        elif op == "Sigmoid":
            # 256-entry int8 -> int8 table, one per Sigmoid node because the
            # table folds this tensor's input AND output quantisation into
            # itself. Built from the same expression int8_sim.py evaluates,
            # so it is not an approximation of the simulator -- it IS the
            # simulator, tabulated. No expf on the target.
            q = np.arange(-128, 128, dtype=np.float64)
            deq = (q - z_i) * s_i
            lut = np.clip(np.rint((1.0 / (1.0 + np.exp(-deq))) / s_out) + z_out,
                          -128, 127).astype(np.int8)
            rec["op"] = K.OP_SIGMOID_LUT
            rec.update(out_h=ih, out_w=iw, out_c=ic, out_zp=z_out,
                       n_elem=ih * iw * ic)
            rec["lut_off"] = blob.add_bytes(lut.tobytes())
            shape[oname] = (ih, iw, ic)
            qp[oname] = (s_out, z_out)
            new_buf(oname, ih * iw * ic)

        elif op in ("Transpose", "Reshape"):
            # Both are free in NHWC for this graph. The ONNX graph is NCHW, so
            # its Transpose(0,2,3,1) is exactly "reinterpret as NHWC" -- which
            # is the layout our buffers are already in. The Reshapes that
            # follow flatten [1,H,W,C] to [1,H*W,C]; identical flat order.
            # Anything else would need a real COPY, and we refuse instead of
            # silently producing garbage.
            n_elem = ih * iw * ic
            if op == "Transpose":
                perm = a.get("perm", [])
                if list(perm) != [0, 2, 3, 1]:
                    raise SystemExit(f"node {idx}: Transpose perm {perm} is "
                                     f"not the NCHW->NHWC no-op")
            else:
                tgt = [int(v) for v in init[n.input[1]].astype(int).tolist()]
                resolved = list(tgt)
                if -1 in resolved:
                    known = 1
                    for v in resolved:
                        if v != -1:
                            known *= v
                    resolved[resolved.index(-1)] = n_elem // known
                if int(np.prod(resolved)) != n_elem or resolved[-1] != ic:
                    raise SystemExit(f"node {idx}: Reshape {tgt} is not a flat "
                                     f"NHWC no-op (h,w,c={ih},{iw},{ic})")
            rec["op"] = K.OP_NOP
            rec.update(out_h=ih, out_w=iw, out_c=ic, out_zp=z_i, n_elem=n_elem)
            shape[oname] = (ih, iw, ic)
            qp[oname] = (s_i, z_i)
            alias(oname, n.input[0])

        else:
            raise SystemExit(f"node {idx}: unsupported op {op}")

        rec["_idx"] = idx
        rec["_op_name"] = op
        rec["_out"] = oname
        rec["_in0"] = n.input[0]
        rec["_in1"] = n.input[1] if op == "Add" else None
        layers.append(rec)

    for t in graph_outputs:
        buf_of[t].is_output = True

    # ---- pass B: liveness -------------------------------------------------
    for idx, n in enumerate(g.node):
        for t in n.input:
            b = buf_of.get(t)
            if b is not None:
                b.last_use = max(b.last_use, idx)
        b = buf_of[n.output[0]]
        b.last_use = max(b.last_use, idx)

    # ---- pass C: arena allocation ----------------------------------------
    peak = plan_arena(bufs, len(g.node))

    # Sanity: no two live buffers may overlap. Cheap, and the one bug class
    # in an offline planner that produces plausible-looking wrong numbers.
    for i, b1 in enumerate(bufs):
        for b2 in bufs[i + 1:]:
            e1 = len(g.node) if b1.is_output else b1.last_use
            e2 = len(g.node) if b2.is_output else b2.last_use
            if b1.first_use <= e2 and b2.first_use <= e1:
                s1 = (b1.off, b1.off + max(4, (b1.size + 3) // 4 * 4))
                s2 = (b2.off, b2.off + max(4, (b2.size + 3) // 4 * 4))
                if s1[0] < s2[1] and s2[0] < s1[1]:
                    raise SystemExit(f"arena planner bug: buffers {b1.bid} "
                                     f"{s1} and {b2.bid} {s2} overlap while "
                                     f"both live")

    # ---- pack -------------------------------------------------------------
    for rec in layers:
        rec["in0_off"] = buf_of[rec["_in0"]].off
        rec["out_off"] = buf_of[rec["_out"]].off
        if rec["_in1"] is not None:
            rec["in1_off"] = buf_of[rec["_in1"]].off
        if buf_of[rec["_out"]].is_output:
            rec["flags"] |= 1

    layers_off = K.HEADER_BYTES
    outputs_off = layers_off + len(layers) * K.LAYER_STRIDE
    blob_off = outputs_off + len(out_names) * K.OUTPUT_STRIDE

    body = b"".join(K.pack_layer(r) for r in layers)
    otbl = b""
    for name in out_names:
        s, z = qp[name]
        h, w, c = shape[name]
        otbl += K.pack_output(buf_of[name].off, h * w * c, z, scale_q16(s))

    hdr = K.pack_header(len(layers), peak, layers_off, outputs_off,
                        len(out_names), blob_off, len(blob),
                        buf_of[in_name].off, in_h0, in_w0, in_c0,
                        z_in0, scale_q16(s_in0))
    data = hdr + body + otbl + bytes(blob.buf)
    open(out_path, "wb").write(data)

    if report:
        from collections import Counter
        c = Counter(K.OP_NAMES[r["op"]] for r in layers)
        print(f"  wrote        : {out_path} ({len(data):,} bytes)")
        print(f"  layers       : {len(layers)}  {dict(c)}")
        print(f"  blob         : {len(blob):,} bytes")
        print(f"  layer table  : {len(body):,} bytes "
              f"({K.LAYER_STRIDE} B/record)")
        print(f"  PEAK ARENA   : {peak:,} bytes")
        print(f"  input        : {in_h0}x{in_w0}x{in_c0} NHWC "
              f"at arena+{buf_of[in_name].off}, "
              f"scale={s_in0:.6g} zp={z_in0}")
        print(f"  outputs      : {len(out_names)}")
        for i, name in enumerate(out_names):
            s, z = qp[name]
            h, w, cc = shape[name]
            print(f"    [{i:2d}] {name:<9} n={h*w*cc:<6} "
                  f"arena+{buf_of[name].off:<7} scale={s:.6g} zp={z}")
        for nt in notes:
            print(f"  note: {nt}")
        if left_shift_warn:
            print(f"  WARNING: positive (left) requant shifts at nodes "
                  f"{sorted(set(left_shift_warn))}")
    return {
        "arena_bytes": peak,
        "outputs": [{"name": nm, "off": buf_of[nm].off,
                     "n": int(np.prod(shape[nm]))} for nm in out_names],
        "input_off": buf_of[in_name].off,
        "input_scale": s_in0, "input_zp": z_in0,
        "layer_out_off": [buf_of[r["_out"]].off for r in layers],
        "layer_n_elem": [r["n_elem"] for r in layers],
        "layer_op": [K.OP_NAMES[r["op"]] for r in layers],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("-o", "--out", required=True, help="output prefix")
    ap.add_argument("--emit-binary", metavar="OUT.krnn",
                    help="also emit the flat on-target container")
    ap.add_argument("--ranges", help="activation ranges json "
                                     "(required by --emit-binary)")
    args = ap.parse_args()

    if args.emit_binary:
        if not args.ranges:
            raise SystemExit("--emit-binary needs --ranges")
        sys.path.insert(0, str(_HERE))
        emit_binary(args.model, args.ranges, args.emit_binary)
        return 0

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
