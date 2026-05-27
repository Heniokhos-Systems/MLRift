#!/system/bin/sh
# Android mlrc-only bench. mksh/bionic-compatible (no bash, no `local`).
set -u
DIR="${DIR:-/data/local/tmp/mlrift-bench}"
MLRC="$DIR/mlrc"
RESULTS="$DIR/results-android.md"
PLATFORM="${PLATFORM:-Android / aarch64}"

cpu="$(grep -m1 'Hardware' /proc/cpuinfo 2>/dev/null | sed 's/.*: //')"
[ -z "$cpu" ] && cpu="$(uname -m)"

{
    echo "# MLRift benchmark -- $PLATFORM"
    echo
    echo "**Date:** $(date -u +'%Y-%m-%d %H:%M:%S UTC')"
    echo "**Host:** $cpu"
    echo "**Kernel:** $(uname -srm)"
    echo "**Toolchains:** mlrc=yes (gcc/rustc skipped -- release scope)"
    echo "**Android props:** $(getprop ro.product.model 2>/dev/null) ($(getprop ro.build.version.release 2>/dev/null))"
    echo
} > "$RESULTS"

median3_ms() {
    bin="$1"
    t0=0; t1=0; t2=0
    i=0
    while [ "$i" -lt 3 ]; do
        s=$(date +%s%N)
        "$bin" > /dev/null 2>&1 || true
        e=$(date +%s%N)
        ms=$(( (e - s) / 1000000 ))
        eval "t$i=$ms"
        i=$((i + 1))
    done
    a=$t0; b=$t1; c=$t2
    if   [ "$a" -le "$b" ] && [ "$a" -le "$c" ]; then m=$([ "$b" -le "$c" ] && echo $b || echo $c)
    elif [ "$b" -le "$a" ] && [ "$b" -le "$c" ]; then m=$([ "$a" -le "$c" ] && echo $a || echo $c)
    else                                              m=$([ "$a" -le "$b" ] && echo $a || echo $b)
    fi
    echo "$m"
}

compile_ms() {
    s=$(date +%s%N)
    "$@" > /dev/null 2>&1
    rc=$?
    e=$(date +%s%N)
    if [ "$rc" -ne 0 ]; then echo "FAIL"; else echo $(( (e - s) / 1000000 )); fi
}

bench_one() {
    name="$1"
    src="$DIR/$name.mlr"
    bin="$DIR/$name.mlrc.bin"
    mc="N/A"; sz="N/A"; rt="N/A"
    if [ -f "$src" ]; then
        mc="$(compile_ms "$MLRC" --emit=android "$src" -o "$bin")"
        if [ -f "$bin" ]; then
            chmod +x "$bin" 2>/dev/null
            sz="$(stat -c%s "$bin" 2>/dev/null || wc -c < "$bin")"
            rt="$(median3_ms "$bin")"
        fi
    fi
    {
        echo "## $name"
        echo
        echo "| Compiler | Compile (ms) | Binary (B) | Runtime median-of-3 (ms) |"
        echo "|---|---|---|---|"
        echo "| mlrc (self-hosted) | $mc | $sz | $rt |"
        echo
    } >> "$RESULTS"
    rm -f "$bin"
}

for p in fib sort sieve matmul; do
    bench_one "$p"
done
cat "$RESULTS"
