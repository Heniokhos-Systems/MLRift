#!/usr/bin/env python3
"""Generate the deterministic validation frame, byte for byte.

This mirrors nn_mt_rand()/nn_mt_fill_input() in tests/nn/nn_model_core.mlr
exactly: the same LCG, the same seed, the same NHWC fill order. The MLRift
runner GENERATES this frame on target rather than having it baked in (49 KB
of store8 literals would have been larger than the model), so the only way
both sides can be proved to start from the same tensor is for Python to
reproduce the generator rather than the other way round.

    gen_frame.py <h> <w> <c> <out.i8> [seed]

Writes raw bytes; read them back with int8_sim.py --input-i8.
"""
import sys


def frame_bytes(n, seed=12345):
    s = seed
    out = bytearray(n)
    for i in range(n):
        s = (s * 1103515245 + 12345) & 0xFFFFFFFF
        out[i] = (s >> 16) & 0xFF
    return bytes(out)


def main():
    if len(sys.argv) < 5:
        print(__doc__)
        return 2
    h, w, c = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    seed = int(sys.argv[5]) if len(sys.argv) > 5 else 12345
    data = frame_bytes(h * w * c, seed)
    open(sys.argv[4], "wb").write(data)
    print(f"  {h}x{w}x{c} NHWC, {len(data):,} bytes, seed {seed} "
          f"-> {sys.argv[4]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
