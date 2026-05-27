#!/bin/bash
# Per-platform smoke corpus for MLRift.
#
# Runs every *.mlr under tests/smoke/ against the current host target,
# prints PASS / FAIL per test, and exits non-zero if any test fails.
#
# Optional env knobs:
#   MLRC=<path>     — compiler to test (default: ../build/mlrc2)
#   MLRC_FLAGS=...  — extra flags (e.g. "--arch=x86_64 --emit=elfexe" for
#                    cross-compile + host-run, or "--arch=arm64 --emit=android"
#                    for Android)
#   SMOKE_RUNNER=<cmd>  — wrapper to invoke binaries under (e.g.
#                         "qemu-aarch64-static" for ARM64 via emulation).
#
# Expected exit code for each test is encoded in a single-line comment at
# the top of each .mlr as "// expected: <N>". Defaults to 0.

set -u
DIR="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$DIR/.." && pwd)"
MLRC="${MLRC:-$REPO/../build/mlrc2}"
# Default resolution: prefer build/mlrc2 if MLRC wasn't explicitly set and
# the default doesn't exist.
if [ ! -x "$MLRC" ]; then
    MLRC="$REPO/build/mlrc2"
fi
MLRC_FLAGS="${MLRC_FLAGS:-}"
RUNNER="${SMOKE_RUNNER:-}"

# Default to host-native single-arch output if the caller didn't specify.
# Without --arch/--emit, mlrc produces a .mlrbo fat bundle which isn't
# directly runnable — that's the right default for end users but wrong
# for a smoke harness that exec()s the output.
if [ -z "$MLRC_FLAGS" ]; then
    host_arch="$(uname -m)"
    host_os="$(uname -s)"
    case "$host_arch" in
        x86_64|amd64)  MLRC_FLAGS="--arch=x86_64" ;;
        aarch64|arm64) MLRC_FLAGS="--arch=arm64" ;;
        *) echo "smoke: unrecognised host arch $host_arch, defaulting to x86_64" >&2; MLRC_FLAGS="--arch=x86_64" ;;
    esac
    case "$host_os" in
        Darwin)  MLRC_FLAGS="$MLRC_FLAGS --emit=macho" ;;
        Linux)   : ;;   # default ELF
        *)       : ;;
    esac
fi

if [ ! -x "$MLRC" ]; then
    echo "smoke: mlrc not found (tried: $MLRC)" >&2
    exit 2
fi

PASS=0
FAIL=0
TOTAL=0
FAIL_LIST=""

for src in "$DIR"/*.mlr; do
    name="$(basename "$src" .mlr)"
    TOTAL=$((TOTAL + 1))
    expected="$(grep -oE '// expected:[[:space:]]*[0-9]+' "$src" | head -1 | grep -oE '[0-9]+$')"
    expected="${expected:-0}"
    out="$(mktemp /tmp/mlrc_smoke_${name}_XXXX)"
    rm -f "$out"
    build_log="$(mktemp)"
    if ! $MLRC $MLRC_FLAGS "$src" -o "$out" > "$build_log" 2>&1; then
        echo "FAIL: $name (compile failed)"
        sed 's/^/  /' "$build_log" | tail -5
        FAIL=$((FAIL + 1))
        FAIL_LIST="$FAIL_LIST $name"
        rm -f "$build_log"
        continue
    fi
    rm -f "$build_log"
    chmod +x "$out" 2>/dev/null || true
    abs_out="$(readlink -f "$out" 2>/dev/null || echo "$out")"
    tmpwd="$(mktemp -d)"
    (
        cd "$tmpwd"   # fresh tmpdir so file_rw / file_size side-files
                     # don't collide between runs
        $RUNNER "$abs_out" > /dev/null 2>&1
    )
    got=$?
    rm -rf "$tmpwd"
    if [ "$got" = "$expected" ]; then
        PASS=$((PASS + 1))
    else
        echo "FAIL: $name (expected $expected, got $got)"
        FAIL=$((FAIL + 1))
        FAIL_LIST="$FAIL_LIST $name"
    fi
    rm -f "$out"
done

echo ""
echo "=== smoke: $PASS passed, $FAIL failed (total $TOTAL) ==="
if [ "$FAIL" -gt 0 ]; then
    echo "  failed:$FAIL_LIST" >&2
    exit 1
fi
