#!/bin/bash
# MLRift vs C vs Rust benchmark suite.
#
# Auto-detects host arch; runs whichever of {mlrc, gcc, rustc} is on PATH.
# Missing toolchains are reported as "N/A" rather than failing.
#
# Env knobs:
#   MLRC=<path>       compiler binary (default: ../build/mlrc, then `mlrc`)
#   MLRC_ARCH=<flag>  override `--arch=...` (default: host arch)
#   RESULTS=<path>    output md file (default: results.md beside this script)
#   PLATFORM=<label>  string for the result header (e.g. "Pi 400 / aarch64")

set -u

DIR="$(cd "$(dirname "$0")" && pwd)"
MLRC="${MLRC:-$DIR/../build/mlrc}"
[ -x "$MLRC" ] || MLRC="$(command -v mlrc 2>/dev/null || true)"
RESULTS="${RESULTS:-$DIR/results.md}"

host_arch="$(uname -m)"
case "$host_arch" in
    x86_64|amd64)  default_mlrc_arch="--arch=x86_64" ;;
    aarch64|arm64) default_mlrc_arch="--arch=arm64" ;;
    *)             default_mlrc_arch="" ;;
esac
MLRC_ARCH="${MLRC_ARCH:-$default_mlrc_arch}"

HAVE_MLRC=0; [ -x "$MLRC" ] && HAVE_MLRC=1
HAVE_GCC=0;  command -v gcc   > /dev/null 2>&1 && HAVE_GCC=1
HAVE_RUSTC=0;command -v rustc > /dev/null 2>&1 && HAVE_RUSTC=1

PLATFORM="${PLATFORM:-$(uname -s) / $host_arch}"

cpu_model=""
if [ -r /proc/cpuinfo ]; then
    cpu_model="$(grep -m1 'model name' /proc/cpuinfo | sed 's/.*: //')"
    [ -z "$cpu_model" ] && cpu_model="$(grep -m1 'Hardware' /proc/cpuinfo | sed 's/.*: //')"
fi
[ -z "$cpu_model" ] && cpu_model="$(uname -p 2>/dev/null || echo unknown)"

{
    echo "# MLRift benchmark — $PLATFORM"
    echo
    echo "**Date:** $(date -u +'%Y-%m-%d %H:%M:%S UTC')"
    echo "**Host:** $cpu_model"
    echo "**Kernel:** $(uname -srm)"
    echo "**Toolchains:** mlrc=$([ "$HAVE_MLRC" = 1 ] && echo yes || echo no), gcc=$([ "$HAVE_GCC" = 1 ] && echo yes || echo no), rustc=$([ "$HAVE_RUSTC" = 1 ] && echo yes || echo no)"
    echo "**mlrc flags:** \`$MLRC_ARCH\`"
    echo
} > "$RESULTS"

if [ "$HAVE_MLRC" != 1 ]; then
    echo "ERROR: mlrc not found (MLRC=$MLRC, PATH miss)" >&2
    exit 2
fi

# Median of 3 runs in milliseconds. Echoes the median to stdout.
median3_ms() {
    local t0 t1 t2 a b c m
    for i in 0 1 2; do
        local start end
        start=$(date +%s%N)
        "$@" > /dev/null 2>&1 || true
        end=$(date +%s%N)
        eval "t$i=$(( (end - start) / 1000000 ))"
    done
    a=$t0; b=$t1; c=$t2
    # Median of 3 without external sort
    if   [ "$a" -le "$b" ] && [ "$a" -le "$c" ]; then m=$(( b<=c ? b : c ))
    elif [ "$b" -le "$a" ] && [ "$b" -le "$c" ]; then m=$(( a<=c ? a : c ))
    else                                              m=$(( a<=b ? a : b ))
    fi
    echo "$m"
}

# Compile-time of one command (single run, ms).
compile_ms() {
    local start end
    start=$(date +%s%N)
    "$@" > /dev/null 2>&1
    local rc=$?
    end=$(date +%s%N)
    if [ "$rc" -ne 0 ]; then echo "FAIL"; else echo $(( (end - start) / 1000000 )); fi
}

bench_one() {
    local name="$1"

    local mlrc_compile="N/A" gcc_O0_compile="N/A" gcc_O2_compile="N/A" rs_dbg_compile="N/A" rs_rel_compile="N/A"
    local mlrc_size="N/A"    gcc_O0_size="N/A"    gcc_O2_size="N/A"    rs_dbg_size="N/A"    rs_rel_size="N/A"
    local mlrc_run="N/A"     gcc_O0_run="N/A"     gcc_O2_run="N/A"     rs_dbg_run="N/A"     rs_rel_run="N/A"

    local mlr_src="$DIR/$name.mlr"
    local c_src="$DIR/$name.c"
    local rs_src="$DIR/$name.rs"

    local mlr_bin="$DIR/$name.mlrc.bin"
    local c0_bin="$DIR/$name.c.O0.bin"
    local c2_bin="$DIR/$name.c.O2.bin"
    local rd_bin="$DIR/$name.rs.dbg.bin"
    local rr_bin="$DIR/$name.rs.rel.bin"

    if [ -f "$mlr_src" ]; then
        mlrc_compile=$(compile_ms $MLRC $MLRC_ARCH "$mlr_src" -o "$mlr_bin")
        [ -f "$mlr_bin" ] && chmod +x "$mlr_bin" 2>/dev/null
        [ -f "$mlr_bin" ] && mlrc_size=$(stat -c%s "$mlr_bin" 2>/dev/null || wc -c < "$mlr_bin")
        [ -x "$mlr_bin" ] && mlrc_run="$(median3_ms "$mlr_bin")"
    fi

    if [ "$HAVE_GCC" = 1 ] && [ -f "$c_src" ]; then
        gcc_O0_compile=$(compile_ms gcc -O0 -o "$c0_bin" "$c_src")
        gcc_O2_compile=$(compile_ms gcc -O2 -o "$c2_bin" "$c_src")
        [ -f "$c0_bin" ] && gcc_O0_size=$(stat -c%s "$c0_bin")
        [ -f "$c2_bin" ] && gcc_O2_size=$(stat -c%s "$c2_bin")
        [ -x "$c0_bin" ] && gcc_O0_run="$(median3_ms "$c0_bin")"
        [ -x "$c2_bin" ] && gcc_O2_run="$(median3_ms "$c2_bin")"
    fi

    if [ "$HAVE_RUSTC" = 1 ] && [ -f "$rs_src" ]; then
        rs_dbg_compile=$(compile_ms rustc -o "$rd_bin" "$rs_src")
        rs_rel_compile=$(compile_ms rustc -C opt-level=2 -o "$rr_bin" "$rs_src")
        [ -f "$rd_bin" ] && rs_dbg_size=$(stat -c%s "$rd_bin")
        [ -f "$rr_bin" ] && rs_rel_size=$(stat -c%s "$rr_bin")
        [ -x "$rd_bin" ] && rs_dbg_run="$(median3_ms "$rd_bin")"
        [ -x "$rr_bin" ] && rs_rel_run="$(median3_ms "$rr_bin")"
    fi

    {
        echo "## $name"
        echo
        echo "| Compiler | Compile (ms) | Binary (B) | Runtime median-of-3 (ms) |"
        echo "|---|---|---|---|"
        echo "| mlrc (self-hosted) | $mlrc_compile | $mlrc_size | $mlrc_run |"
        echo "| gcc -O0 | $gcc_O0_compile | $gcc_O0_size | $gcc_O0_run |"
        echo "| gcc -O2 | $gcc_O2_compile | $gcc_O2_size | $gcc_O2_run |"
        echo "| rustc (debug) | $rs_dbg_compile | $rs_dbg_size | $rs_dbg_run |"
        echo "| rustc -O opt2 | $rs_rel_compile | $rs_rel_size | $rs_rel_run |"
        echo
    } | tee -a "$RESULTS"

    rm -f "$mlr_bin" "$c0_bin" "$c2_bin" "$rd_bin" "$rr_bin"
}

for prog in fib sort sieve matmul; do
    bench_one "$prog"
done

echo "Done. Results: $RESULTS"
