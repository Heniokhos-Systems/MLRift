#!/usr/bin/env python3
"""Turn an image into the int8 NHWC frame the MLRift runner expects.

    frame_from_image.py <image> <side> <out.i8> [--scale S] [--zp Z]

The frame the runner is handed is ALREADY quantised: the .krnn header carries
the input (scale, zero_point) and the caller is responsible for applying it,
exactly as an MCU camera driver would. For a YuNet converted from a range
file that does not name the graph input, that quantisation is the identity
minus 128 (scale 1.0, zp -128), i.e. pixel - 128 — which is what --scale/--zp
default to.

Needs pillow; nothing on the target side does.
"""
import argparse, sys
import numpy as np
from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("side", type=int)
    ap.add_argument("out")
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--zp", type=int, default=-128)
    args = ap.parse_args()

    im = Image.open(args.image).convert("RGB").resize((args.side, args.side))
    x = np.asarray(im, dtype=np.float64)              # H, W, C -- already NHWC
    q = np.clip(np.rint(x / args.scale) + args.zp, -128, 127).astype(np.int8)
    q.tofile(args.out)
    print(f"  {args.image} -> {args.out}: {args.side}x{args.side}x3 NHWC int8, "
          f"{q.size:,} bytes (scale {args.scale}, zp {args.zp})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
