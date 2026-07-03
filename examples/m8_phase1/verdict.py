#!/usr/bin/env python3
"""MLRift — M8 Phase 1 pre-registered verdict.

Runs the full byte-exact harness (compare_gate.sh) and adjudicates the
result against the pre-registered rule (spec: M8 Phase 1 plan, "Verdict
algorithm"):

  1. Parse compare_gate.sh's output into the set of divergent
     (seed, trial) pairs. Empty -> BYTE-EXACT PASS.
  2. Otherwise, for EVERY divergent trial, prove the divergence is
     terminal-ULP: both the gate (`--trace` mode, see
     examples/m8_phase1_spiking_gate.mlr) and a from-scratch numpy
     replay of the same trial (this file's `numpy_trace`, using
     Phase-0's own LIF/WTA op-order verbatim) must show that, at the
     FIRST differing settle step, the two per-neuron membrane
     potentials bracket that neuron's own spike threshold and differ
     by at most one ULP at that threshold. Any non-terminal-ULP step
     (bigger gap, no bracket, or a systematic same-sign pattern across
     many trials / a whole seed) is a logic/op-order/codegen bug, not
     rounding -> INVESTIGATE.
  3. Require the divergent fraction < 1% AND a host re-score: import
     the (untouched, on-disk) Phase-0 gate, monkeypatch its
     `build_lifetime` to substitute the .mlr's own (m, margin) for
     every seed we have a full lifetime of .mlr output for (wherever
     Phase-0 calls it at the exact CENTROID defaults our replay dump
     was generated from), and re-run its `verdict()` — require the
     same all-checks-pass ('PASS-necessary-not-sufficient').
  4. Print exactly one of:
       VERDICT: BYTE-EXACT PASS
       VERDICT: TOLERANCE PASS
       VERDICT: INVESTIGATE
     A TOLERANCE PASS requires every divergent trial proven
     terminal-ULP AND the re-score to hold. Anything unproven prints
     INVESTIGATE — this script never asserts a PASS it hasn't earned.

Given the gate is already byte-exact on every probed seed, the expected
(and target) outcome is BYTE-EXACT PASS, in which case steps 2-3 above
never execute — but they are real, runnable logic (not stubs), so a
future divergence is adjudicated honestly rather than rubber-stamped.

Usage:
    python3 verdict.py [data_dir]
      data_dir — optional override; defaults to the full local
                 (gitignored) 26-seed dump at m8_phase1/data/ if
                 present, else the committed 2-seed m8_phase1/data_public/.

Env:
    M8P0_GATE — path to Noesis's local-only Phase-0 gate script
                (m8_phase0_spiking_killgate.py). Only required if step 3
                (host re-score) actually runs, i.e. only on a
                divergence. No baked default — this is a local dev tool,
                never a public MLRift dependency.
"""
import importlib.util
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MLRIFT_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
COMPARE_SH = os.path.join(HERE, "compare_gate.sh")
GATE_BIN = "/tmp/m8p1_gate"                # built by compare_gate.sh
GATE_SORTED = "/tmp/m8p1_gate_sorted.txt"  # written by compare_gate.sh
REF_SORTED = "/tmp/m8p1_ref_sorted.txt"    # written by compare_gate.sh

# Gate constants (verbatim from Phase-0 / examples/m8_phase1_spiking_gate.mlr).
A = 8
N_SYM, K_SYM = 160, 20
Mc, K_CTX = 300, 40
L = 300
DT, TAU_M = 0.1, 10.0
V_REST, V_THRESH, V_RESET = 0.0, 1.0, 0.0
REFRAC_STEPS = 20
WIN_STEPS = 120
INPUT_SCALE = 20.0
G_INH = 0.05
INH_DECAY = 0.8
DIVERGENT_FRACTION_LIMIT = 0.01


def pick_data_dir(explicit):
    """Mirror compare_gate.sh's own data-dir selection exactly, so the
    trace/re-score steps analyze the same run compare_gate.sh just did."""
    if explicit:
        return os.path.abspath(explicit)
    local = os.path.join(HERE, "data")
    if os.path.isfile(os.path.join(local, "reference_mmargin.txt")):
        return local
    return os.path.join(HERE, "data_public")


def run_compare(data_dir):
    """Run compare_gate.sh (which builds the gate + writes GATE_SORTED /
    REF_SORTED), streaming its own PASS/DIVERGENCE line to our stdout."""
    result = subprocess.run(["bash", COMPARE_SH, data_dir], cwd=MLRIFT_ROOT,
                             capture_output=True, text=True, check=True)
    sys.stdout.write(result.stdout)
    if result.stderr:
        sys.stderr.write(result.stderr)


def load_rows(path):
    rows = {}
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) != 4:
                continue
            sd, t, m, mg = (int(x) for x in parts)
            rows[(sd, t)] = (m, mg)
    return rows


def divergent_pairs():
    """Diff GATE_SORTED against REF_SORTED at the (seed, trial) level."""
    gate = load_rows(GATE_SORTED)
    ref = load_rows(REF_SORTED)
    keys = sorted(set(gate) | set(ref))
    diverging = [k for k in keys if gate.get(k) != ref.get(k)]
    total = len(ref) if ref else len(keys)
    return diverging, gate, ref, total


def load_ints(path):
    with open(path) as f:
        return [int(x) for x in f.read().split()]


def load_cueidx_lines(path):
    with open(path) as f:
        return [[int(x) for x in line.split()] for line in f]


def numpy_trace(data_dir, seed, trial):
    """Recompute one trial's per-step membrane-potential trace directly
    from the dumped replay inputs, using Phase-0's recall() LIF/WTA
    op-order VERBATIM (mult-then-divide, refractory-freeze, product-first
    inhibition — see m8_phase0_spiking_killgate.py:recall) with a V
    snapshot inserted after every step. This IS the reference dynamics
    (not a re-derivation of it): the golden reference_mmargin.txt was
    produced by exactly this vectorized formula.

    Returns (trace, vth): trace is (WIN_STEPS, N_SYM) f64, vth is the
    per-neuron threshold V_THRESH + theta[i] used at every step.
    """
    seq = np.array(load_ints(f"{data_dir}/seed{seed}_seq.txt"))
    symcode = np.array(load_ints(f"{data_dir}/seed{seed}_symcode.txt")).reshape(A, K_SYM)
    cuecode = np.array(load_ints(f"{data_dir}/seed{seed}_cuecode.txt")).reshape(L, K_CTX)
    cueidx_all = load_cueidx_lines(f"{data_dir}/seed{seed}_cueidx.txt")
    theta = np.fromfile(f"{data_dir}/seed{seed}_theta.bin", dtype="<f8")

    W = np.zeros((N_SYM, Mc), dtype=np.int64)
    for t in range(seq.shape[0]):
        si = symcode[seq[t]]
        W[np.ix_(si, cuecode[t])] += 1

    cue_idx = np.array(cueidx_all[trial])
    raw = W[:, cue_idx].sum(1).astype(np.float64)
    row_sum = W.sum(1).astype(np.float64)
    row_sum[row_sum == 0] = 1.0
    ff = raw / row_sum * INPUT_SCALE

    V = np.full(N_SYM, V_REST)
    refr = np.zeros(N_SYM, dtype=int)
    vth = V_THRESH + theta
    inh = 0.0
    trace = np.zeros((WIN_STEPS, N_SYM))
    for step in range(WIN_STEPS):
        active = refr <= 0
        drive = ff - G_INH * inh
        V[active] += DT * (V_REST - V[active] + drive[active]) / TAU_M
        spiked = active & (V >= vth)
        n_spiked = int(spiked.sum())
        V[spiked] = V_RESET
        refr[spiked] = REFRAC_STEPS
        refr[~active] -= 1
        inh = inh * INH_DECAY + float(n_spiked)
        trace[step] = V
    return trace, vth


def mlr_trace(gate_bin, data_dir, seed, trial):
    out_path = f"/tmp/m8p1_trace_{seed}_{trial}.bin"
    subprocess.run([gate_bin, "--trace", str(seed), str(trial), out_path, data_dir],
                    cwd=MLRIFT_ROOT, check=True, capture_output=True)
    flat = np.fromfile(out_path, dtype="<f8")
    if flat.size != WIN_STEPS * N_SYM:
        raise ValueError(f"trace size mismatch: {flat.size} != {WIN_STEPS * N_SYM}")
    return flat.reshape(WIN_STEPS, N_SYM)


def first_diff_step(np_trace, trace_mlr):
    step_diffs = np.where((np_trace != trace_mlr).any(axis=1))[0]
    return int(step_diffs[0]) if len(step_diffs) else None


def is_terminal_ulp(gate_bin, data_dir, seed, trial):
    """Returns (ok: bool, message: str, sign: float|None).

    sign is +1 if numpy's V ran ahead of .mlr's at the first bad neuron,
    -1 if behind, None if not applicable — used by the caller to check
    for a systematic same-sign bias across trials (a real bug looks the
    same way every time; rounding noise doesn't)."""
    np_tr, vth = numpy_trace(data_dir, seed, trial)
    tr = mlr_trace(gate_bin, data_dir, seed, trial)

    step = first_diff_step(np_tr, tr)
    if step is None:
        return False, (f"seed {seed} trial {trial}: (m,margin) differ but the full "
                        f"{WIN_STEPS}-step V trace is IDENTICAL -> divergence is "
                        f"downstream of the settle loop (readout/tie-break bug), not rounding"), None

    diff = np.abs(np_tr[step] - tr[step])
    bad = np.where(diff > 0)[0]
    signs = []
    for i in bad:
        v_np, v_mlr, vth_i = np_tr[step, i], tr[step, i], vth[i]
        bracket = (v_np - vth_i) * (v_mlr - vth_i) < 0
        ulp_i = np.nextafter(vth_i, np.inf) - vth_i
        gap = abs(v_np - v_mlr)
        if not (bracket and gap <= ulp_i):
            return False, (f"seed {seed} trial {trial} step {step} neuron {i}: "
                            f"np_V={v_np!r} mlr_V={v_mlr!r} vth={vth_i!r} bracket={bracket} "
                            f"|diff|={gap:.3e} 1ulp@vth={ulp_i:.3e} -> NOT terminal-ULP"), None
        signs.append(1.0 if v_np > v_mlr else -1.0)

    return True, (f"seed {seed} trial {trial}: first divergence step {step}, "
                  f"{len(bad)} neuron(s), all bracket their own vth within 1 ULP "
                  f"-> terminal-ULP"), (signs[0] if signs else None)


def host_rescore(data_dir, gate_rows):
    """Substitute the .mlr's own (m, margin) into Phase-0's build_lifetime
    output (in-process monkeypatch only — the on-disk gate script is never
    modified) wherever Phase-0 calls it at the exact CENTROID defaults
    (b=3, L=300, cue_noise=0.15, win=120, theta_scale=0.05) for a seed we
    have every trial of .mlr output for, then re-run verdict() and require
    the same all-checks-pass result."""
    gate_path = os.environ.get("M8P0_GATE")
    if not gate_path:
        return False, "M8P0_GATE not set (local dev tool, no baked path) -> cannot host re-score"
    if not os.path.isfile(gate_path):
        return False, f"M8P0_GATE={gate_path} does not exist"

    spec = importlib.util.spec_from_file_location("m8p0", gate_path)
    m8 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m8)

    orig_build_lifetime = m8.build_lifetime

    def patched_build_lifetime(b, seed, Lc=m8.L, cue_noise=m8.CUE_NOISE,
                                win=m8.WIN_STEPS, theta_scale=m8.THETA_SCALE):
        lf = orig_build_lifetime(b, seed, Lc=Lc, cue_noise=cue_noise, win=win, theta_scale=theta_scale)
        is_centroid_default = (b == 3 and Lc == m8.L and cue_noise == m8.CUE_NOISE
                                and win == m8.WIN_STEPS and theta_scale == m8.THETA_SCALE)
        have_full_lifetime = all((seed, t) in gate_rows for t in range(lf["L"]))
        if is_centroid_default and have_full_lifetime:
            recs = lf["recs"].copy()
            margins = lf["margins"].copy()
            for t in range(lf["L"]):
                m_mlr, mg_mlr = gate_rows[(seed, t)]
                recs[t] = m_mlr
                margins[t] = float(mg_mlr)
            correct = (recs == lf["seq"]).astype(int)
            thr = np.quantile(margins, 2 / 3)
            confident = margins >= thr
            cw_mask = (recs != lf["seq"]) & confident
            lf = dict(lf)
            lf.update(recs=recs, margins=margins, correct=correct, confident=confident, cw_mask=cw_mask,
                      dump=[(t, int(recs[t]), float(margins[t]), lf["dump"][t][3]) for t in range(lf["L"])])
        return lf

    m8.build_lifetime = patched_build_lifetime
    try:
        res = m8.verdict()
    finally:
        m8.build_lifetime = orig_build_lifetime
    return res == "PASS-necessary-not-sufficient", f"host re-score verdict() = {res!r}"


def main():
    explicit = sys.argv[1] if len(sys.argv) > 1 else None
    data_dir = pick_data_dir(explicit)
    run_compare(data_dir)

    diverging, gate, ref, total = divergent_pairs()
    if not diverging:
        print("VERDICT: BYTE-EXACT PASS")
        return 0

    frac = len(diverging) / total
    print(f"{len(diverging)}/{total} trials diverge ({frac:.4%})")

    # Systematic-pattern pre-check: an entire seed diverging is a
    # logic/op-order bug, never rounding noise.
    per_seed_divergent, per_seed_total = {}, {}
    for (sd, _t) in gate:
        per_seed_total[sd] = per_seed_total.get(sd, 0) + 1
    for (sd, _t) in diverging:
        per_seed_divergent[sd] = per_seed_divergent.get(sd, 0) + 1
    whole_seeds = [sd for sd, n in per_seed_divergent.items() if n == per_seed_total.get(sd)]
    if whole_seeds:
        print(f"VERDICT: INVESTIGATE (entire seed(s) {whole_seeds} diverge in full -> systematic, not rounding)")
        return 1

    if frac >= DIVERGENT_FRACTION_LIMIT:
        print(f"VERDICT: INVESTIGATE (divergent fraction {frac:.4%} >= {DIVERGENT_FRACTION_LIMIT:.0%} threshold)")
        return 1

    signs = []
    for (sd, t) in diverging:
        ok, msg, sign = is_terminal_ulp(GATE_BIN, data_dir, sd, t)
        print(f"  {msg}")
        if not ok:
            print("VERDICT: INVESTIGATE (a divergent trial is NOT terminal-ULP)")
            return 1
        if sign is not None:
            signs.append(sign)

    if len(signs) >= 3 and (all(s > 0 for s in signs) or all(s < 0 for s in signs)):
        print("VERDICT: INVESTIGATE (systematic same-sign divergence across every trial -> not rounding)")
        return 1

    ok, msg = host_rescore(data_dir, gate)
    print(msg)
    if not ok:
        print("VERDICT: INVESTIGATE (host re-score did not reproduce the all-checks-pass verdict)")
        return 1

    print("VERDICT: TOLERANCE PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
