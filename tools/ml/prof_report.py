#!/usr/bin/env python3
"""tools/ml/prof_report.py — turn a nn_prof_esp32.mlr capture into a table.

    prof_report.py <model.krnn> <capture.txt> [--mhz 40] [--compare other.txt]

The capture is whatever came off the UART. Lines may be prefixed with a
host-side "<unix_ts> " timestamp (that is what tests capture with), which is
stripped. Only "P <index> <op> <cycles>" and "PTOT <cycles>" lines are used;
everything else is ignored, so a capture that also contains the ROM banner and
the 1344 output values works unchanged.

The layer geometry comes from the .krnn itself, so the table can say WHAT each
expensive layer is (shape, kernel, stride) without the target having to send
it.

Exit status is 0 always — this is a reporting tool, not a test.
"""
import sys
import os
import struct

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import krnn


def read_layers(path):
    """Parse a .krnn's header + layer table into a list of field dicts."""
    blob = open(path, "rb").read()
    magic, version, layer_count, arena_bytes, layers_off, layer_stride = \
        struct.unpack_from("<IIIIII", blob, 0)
    if magic != krnn.MAGIC:
        sys.exit("%s: not a KRNN file (magic %#x)" % (path, magic))
    if layer_stride != krnn.LAYER_STRIDE:
        sys.exit("%s: layer stride %d, expected %d"
                 % (path, layer_stride, krnn.LAYER_STRIDE))
    layers = []
    for i in range(layer_count):
        base = layers_off + i * layer_stride
        d = {}
        for j, name in enumerate(krnn.LAYER_FIELDS):
            fmt = "<i" if name in krnn.SIGNED_FIELDS else "<I"
            d[name] = struct.unpack_from(fmt, blob, base + 4 * j)[0]
        layers.append(d)
    return layers, arena_bytes


def read_capture(path, want=None):
    """{layer_index: (op, cycles)} plus the reported PTOT, for ONE run.

    The on-target driver loops, so a capture holds several blocks and BOTH ends
    are typically truncated — the first because the capture attached late, the
    last because the capture was stopped mid-run. So split on the "P 0 ..." line
    that starts each block and return the LARGEST block (or, if `want` layers
    are expected, the last block that has exactly that many). Taking "the last
    block" is exactly the bug this replaced: it silently reported a 22-layer
    fragment as if it were the whole 106-layer profile.
    """
    blocks = []          # list of (per, tot)
    per, tot = {}, None
    for raw in open(path, "r", errors="replace"):
        f = raw.split()
        # Strip a leading host timestamp if present.
        if f and f[0].replace(".", "", 1).isdigit() and len(f) > 1:
            f = f[1:]
        if not f:
            continue
        if f[0] == "P" and len(f) == 4:
            try:
                idx, op, cyc = int(f[1]), int(f[2]), int(f[3])
            except ValueError:
                continue
            if idx == 0 and per:
                blocks.append((per, tot))
                per, tot = {}, None
            per[idx] = (op, cyc)
        elif f[0] == "PTOT" and len(f) == 2:
            try:
                tot = int(f[1])
            except ValueError:
                pass
    if per:
        blocks.append((per, tot))
    if not blocks:
        return {}, None
    full = [b for b in blocks if want is not None and len(b[0]) == want]
    if full:
        return full[-1]
    return max(blocks, key=lambda b: len(b[0]))


def shape(d):
    op = d["op"]
    s = "%dx%dx%d -> %dx%dx%d" % (d["in_h"], d["in_w"], d["in_c"],
                                  d["out_h"], d["out_w"], d["out_c"])
    if op in (krnn.OP_CONV2D, krnn.OP_DEPTHWISE, krnn.OP_MAXPOOL,
              krnn.OP_AVGPOOL):
        s += "  k%dx%d s%d p%d" % (d["kh"], d["kw"], d["stride_h"], d["pad_h"])
    return s


def macs(d):
    """Multiply-accumulates, as the natural work unit for a cycles/MAC figure."""
    op = d["op"]
    ohw = d["out_h"] * d["out_w"]
    if op == krnn.OP_CONV2D or op == krnn.OP_POINTWISE:
        return ohw * d["out_c"] * d["in_c"] * max(d["kh"], 1) * max(d["kw"], 1)
    if op == krnn.OP_DEPTHWISE:
        return ohw * d["out_c"] * d["kh"] * d["kw"]
    return 0


def main():
    args = [a for a in sys.argv[1:]]
    mhz = 40.0
    compare = None
    out = []
    i = 0
    while i < len(args):
        if args[i] == "--mhz":
            mhz = float(args[i + 1]); i += 2
        elif args[i] == "--compare":
            compare = args[i + 1]; i += 2
        else:
            out.append(args[i]); i += 1
    if len(out) != 2:
        sys.exit(__doc__)
    model, capture = out

    layers, arena = read_layers(model)
    per, tot = read_capture(capture, want=len(layers))
    if not per:
        sys.exit("%s: no 'P <i> <op> <cycles>' lines found" % capture)
    base = None
    if compare:
        base, _ = read_capture(compare, want=len(layers))

    measured = sum(c for _, c in per.values())
    print("model   %s  (%d layers, arena %d B)" % (model, len(layers), arena))
    print("capture %s  (%d layers timed)" % (capture, len(per)))
    print("total   %d cycles = %.3f s at %.1f MHz%s"
          % (measured, measured / (mhz * 1e6), mhz,
             "" if tot in (None, measured) else "  [PTOT says %d]" % tot))
    if len(per) != len(layers):
        print("WARNING: only %d of %d layers present — capture is a fragment"
              % (len(per), len(layers)))
    print()

    # ---- per op type ----
    agg = {}
    for idx, (op, cyc) in per.items():
        a = agg.setdefault(op, [0, 0, 0])
        a[0] += cyc
        a[1] += 1
        a[2] += macs(layers[idx]) if idx < len(layers) else 0
    print("%-12s %5s %14s %7s %14s %9s" %
          ("op", "n", "cycles", "%", "MACs", "cyc/MAC"))
    print("-" * 68)
    for op in sorted(agg, key=lambda o: -agg[o][0]):
        cyc, n, mc = agg[op]
        print("%-12s %5d %14d %6.2f%% %14d %9s"
              % (krnn.OP_NAMES.get(op, "op%d" % op), n, cyc,
                 100.0 * cyc / measured, mc,
                 "%.2f" % (cyc / mc) if mc else "-"))
    print("-" * 68)
    print("%-12s %5d %14d %6.2f%%" % ("TOTAL", len(per), measured, 100.0))
    print()

    # ---- per layer, worst first ----
    print("worst layers:")
    hdr = "%4s %-11s %12s %6s %8s %-34s" % \
        ("idx", "op", "cycles", "%", "cyc/MAC", "shape")
    if base:
        hdr += " %12s %8s" % ("was", "speedup")
    print(hdr)
    print("-" * (len(hdr) + 2))
    shown = 0
    for idx, (op, cyc) in sorted(per.items(), key=lambda kv: -kv[1][1]):
        if shown >= 25:
            break
        d = layers[idx] if idx < len(layers) else krnn.blank_layer()
        mc = macs(d)
        line = "%4d %-11s %12d %5.2f%% %8s %-34s" % (
            idx, krnn.OP_NAMES.get(op, "op%d" % op), cyc,
            100.0 * cyc / measured,
            "%.2f" % (cyc / mc) if mc else "-", shape(d))
        if base:
            b = base.get(idx, (op, 0))[1]
            line += " %12d %7s" % (b, "%.2fx" % (b / cyc) if cyc else "-")
        print(line)
        shown += 1

    if base:
        bt = sum(c for _, c in base.values())
        print()
        print("overall: %d -> %d cycles  (%.3fx)"
              % (bt, measured, bt / measured if measured else 0))


if __name__ == "__main__":
    main()
