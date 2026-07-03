#!/usr/bin/env python3
"""Numpy-side reference for recall_probe.mlr's single-trial byte-exact check.

Reproduces Noesis Phase-0's own `recall()` (m8_phase0_spiking_killgate.py) for
ONE fixed (seed, trial) using the SAME replay inputs `recall_probe.mlr` loads
from `examples/m8_phase1/data_public/` — this is Phase-0's own recall function
called on the dumped/replayed data, not a re-derivation, so any divergence
against the .mlr probe's output is a real port bug, not a reference drift.

Requires local access to the private Noesis repo (path via M8P0_GATE env var
or the default below); NOT required to run the committed .mlr example itself.

Usage:
    $PY recall_reference.py > /tmp/recall_ref.txt
    diff /tmp/recall_ref.txt <(/tmp/recall_probe)
"""
import importlib.util
import os
import sys

import numpy as np

GATE = os.environ.get(
    "M8P0_GATE",
    "/home/pantelis/Desktop/Projects/Work/Noesis/scripts/m8_phase0_spiking_killgate.py",
)
spec = importlib.util.spec_from_file_location("m8p0", GATE)
m8 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m8)

SEED = 20
TRIAL = 0
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_public")


def load_int_table(fname):
    return np.loadtxt(os.path.join(DATA, fname), dtype=np.int64, ndmin=1)


def main():
    seq = load_int_table(f"seed{SEED}_seq.txt")
    sym_code = load_int_table(f"seed{SEED}_symcode.txt")
    cue_code = load_int_table(f"seed{SEED}_cuecode.txt")
    theta = np.fromfile(os.path.join(DATA, f"seed{SEED}_theta.bin"), dtype="<f8")
    with open(os.path.join(DATA, f"seed{SEED}_cueidx.txt")) as f:
        lines = f.readlines()
    cue_idx = np.array([int(x) for x in lines[TRIAL].split()], dtype=np.int64)

    W = m8.build_store(seq, cue_code, sym_code)
    mm, mg, _sc, _mff = m8.recall(W, cue_idx, theta, sym_code)
    print(f"{int(mm)} {int(mg)}")


if __name__ == "__main__":
    main()
