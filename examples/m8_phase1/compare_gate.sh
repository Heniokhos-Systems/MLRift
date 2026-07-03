#!/usr/bin/env bash
# MLRift — M8 Phase 1 full byte-exact harness.
#
# Builds examples/m8_phase1_spiking_gate.mlr and runs it over every
# seed present in the chosen data directory, concatenates + sorts its
# stdout by (seed, trial), and byte-diffs the result against that
# directory's reference_mmargin.txt (the golden numpy reference).
#
# Data directory selection: defaults to the full local (gitignored)
# 26-seed dump at examples/m8_phase1/data/ if present (the author's
# full run); a public clone — which only has the committed 2-seed
# examples/m8_phase1/data_public/ fixture — automatically falls back
# to that. Pass a directory explicitly as $1 to override either
# default. Either way, the seed list is discovered by globbing
# "seed*_seq.txt" in the chosen directory rather than hardcoding
# 20..45, so this runs correctly against a partial (2-seed) subset.
#
# Usage: compare_gate.sh [data_dir]
set -euo pipefail
cd "$(dirname "$0")/../.."   # MLRift/

LOCAL_DATA=examples/m8_phase1/data
PUBLIC_DATA=examples/m8_phase1/data_public
if [ $# -ge 1 ]; then
    DATA_DIR="$1"
elif [ -f "$LOCAL_DATA/reference_mmargin.txt" ]; then
    DATA_DIR="$LOCAL_DATA"
else
    DATA_DIR="$PUBLIC_DATA"
fi

if [ ! -f "$DATA_DIR/reference_mmargin.txt" ]; then
    echo "compare_gate.sh: no reference_mmargin.txt under $DATA_DIR" >&2
    exit 1
fi

GATE_BIN=/tmp/m8p1_gate
GATE_ALL=/tmp/m8p1_gate_all.txt
GATE_SORTED=/tmp/m8p1_gate_sorted.txt
REF_SORTED=/tmp/m8p1_ref_sorted.txt

./build/mlrc --arch=x86_64 --target=linux --emit=elfexe \
    examples/m8_phase1_spiking_gate.mlr -o "$GATE_BIN"

: > "$GATE_ALL"
shopt -s nullglob
for f in "$DATA_DIR"/seed*_seq.txt; do
    base=$(basename "$f")
    sd=${base#seed}
    sd=${sd%_seq.txt}
    "$GATE_BIN" "$sd" "$DATA_DIR" >> "$GATE_ALL"
done
shopt -u nullglob

sort -n -k1,1 -k2,2 "$GATE_ALL" > "$GATE_SORTED"
sort -n -k1,1 -k2,2 "$DATA_DIR/reference_mmargin.txt" > "$REF_SORTED"

if diff -q "$GATE_SORTED" "$REF_SORTED" >/dev/null; then
    n=$(wc -l < "$REF_SORTED")
    echo "BYTE-EXACT PASS: all $n (seed,trial,m,margin) bit-identical"
else
    n=$(diff "$GATE_SORTED" "$REF_SORTED" | grep -c '^[<>]' || true)
    echo "DIVERGENCE: $n differing lines (see: diff $GATE_SORTED $REF_SORTED)"
fi
