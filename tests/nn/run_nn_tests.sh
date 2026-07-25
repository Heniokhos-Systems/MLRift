#!/bin/bash
# tests/nn/run_nn_tests.sh — numeric validation for std/nn_int8.mlr.
#
# Emits one "PASS: <name> (<detail>)" / "FAIL: ..." / "SKIP: ..." line per
# check, so tests/run_tests.sh can tally it directly. Run standalone with:
#
#     MLRC=./build/mlrc bash tests/nn/run_nn_tests.sh
#
# What is being proved:
#   nn_numeric_host   the kernels' output on the host is EXACTLY equal to
#                     tests/nn/nn_ref.py, an independent Python reference
#                     written from the gemmlowp/TFLite definitions using
#                     native 64-bit products — the very operation the MLRift
#                     kernels are forbidden to use and synthesise instead.
#   nn_numeric_xtensa the same 896 values, byte-identical, from a freestanding
#                     xtensa image booted under qemu-system-xtensa -M lx60.
#   nn_numeric_riscv32 ditto for a freestanding riscv32 flat image under
#                     qemu-system-riscv32 -M virt.
#   nn_build_mcu      both MCU images compile even when no qemu is installed.

DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$DIR/../.."
MLRC="${MLRC:-$ROOT/build/mlrc}"
TMP="${TMPDIR:-/tmp}/nn_$$"
mkdir -p "$TMP"
trap 'rm -rf "$TMP"' EXIT

# Run a qemu image that ends in `loop { }`, stopping as soon as it has printed
# its END sentinel. A plain `timeout N qemu ...` would always burn the full N
# seconds, and piping to head loses buffered output when the kill lands.
qemu_until_end() {
    local outf="$1"; shift
    : > "$outf"
    "$@" > "$outf" 2>/dev/null &
    local qpid=$!
    local i
    for i in $(seq 1 300); do
        grep -q '^END$' "$outf" 2>/dev/null && break
        sleep 0.1
    done
    kill $qpid 2>/dev/null
    wait $qpid 2>/dev/null
}

# ---------------------------------------------------------------------------
# 1. Python reference
# ---------------------------------------------------------------------------
if ! command -v python3 > /dev/null 2>&1; then
    echo "SKIP: nn_numeric_host (no python3 for the reference)"
    echo "SKIP: nn_numeric_xtensa (no python3 for the reference)"
    echo "SKIP: nn_numeric_riscv32 (no python3 for the reference)"
else
    if ! python3 "$DIR/nn_ref.py" > "$TMP/expected.txt" 2> "$TMP/pyerr.txt"; then
        echo "FAIL: nn_numeric_host (reference nn_ref.py failed)"
        head -5 "$TMP/pyerr.txt"
    else
        NVALS=$(wc -l < "$TMP/expected.txt" | tr -d ' ')

        # ------------------------------------------------------------------
        # 2. Host
        # ------------------------------------------------------------------
        if ! $MLRC --arch="$(uname -m)" "$DIR/nn_host.mlr" -o "$TMP/nn_host" \
                > "$TMP/hostbuild.txt" 2>&1; then
            echo "FAIL: nn_numeric_host (host build failed)"
            head -5 "$TMP/hostbuild.txt"
        else
            chmod +x "$TMP/nn_host"
            "$TMP/nn_host" > "$TMP/host.txt" 2>&1
            if cmp -s "$TMP/expected.txt" "$TMP/host.txt"; then
                echo "PASS: nn_numeric_host ($NVALS values exactly equal to the Python reference)"
            else
                echo "FAIL: nn_numeric_host (host output != nn_ref.py)"
                diff "$TMP/expected.txt" "$TMP/host.txt" | head -10
            fi
        fi

        # ------------------------------------------------------------------
        # 3. xtensa under qemu
        # ------------------------------------------------------------------
        if ! command -v qemu-system-xtensa > /dev/null 2>&1; then
            echo "SKIP: nn_numeric_xtensa (no qemu-system-xtensa)"
        elif ! $MLRC --arch=xtensa --freestanding "$DIR/nn_mcu.mlr" \
                -o "$TMP/nn_xt.elf" > "$TMP/xtbuild.txt" 2>&1; then
            echo "FAIL: nn_numeric_xtensa (xtensa build failed)"
            head -5 "$TMP/xtbuild.txt"
        else
            qemu_until_end "$TMP/xt.txt" \
                qemu-system-xtensa -M lx60 -nographic -kernel "$TMP/nn_xt.elf"
            head -n "$NVALS" "$TMP/xt.txt" > "$TMP/xtvals.txt"
            if ! grep -q '^END$' "$TMP/xt.txt"; then
                echo "FAIL: nn_numeric_xtensa (image did not reach END under qemu)"
            elif cmp -s "$TMP/expected.txt" "$TMP/xtvals.txt"; then
                echo "PASS: nn_numeric_xtensa ($NVALS values byte-identical to host and reference)"
            else
                echo "FAIL: nn_numeric_xtensa (xtensa output != nn_ref.py)"
                diff "$TMP/expected.txt" "$TMP/xtvals.txt" | head -10
            fi
        fi

        # ------------------------------------------------------------------
        # 4. riscv32 under qemu
        # ------------------------------------------------------------------
        if ! command -v qemu-system-riscv32 > /dev/null 2>&1; then
            echo "SKIP: nn_numeric_riscv32 (no qemu-system-riscv32)"
        elif ! $MLRC --arch=riscv32 --freestanding "$DIR/nn_mcu_riscv.mlr" \
                -o "$TMP/nn_rv.bin" > "$TMP/rvbuild.txt" 2>&1; then
            echo "FAIL: nn_numeric_riscv32 (riscv32 build failed)"
            head -5 "$TMP/rvbuild.txt"
        else
            qemu_until_end "$TMP/rv.txt" \
                qemu-system-riscv32 -M virt -nographic -bios "$TMP/nn_rv.bin"
            head -n "$NVALS" "$TMP/rv.txt" > "$TMP/rvvals.txt"
            if ! grep -q '^END$' "$TMP/rv.txt"; then
                echo "FAIL: nn_numeric_riscv32 (image did not reach END under qemu)"
            elif cmp -s "$TMP/expected.txt" "$TMP/rvvals.txt"; then
                echo "PASS: nn_numeric_riscv32 ($NVALS values byte-identical to host and reference)"
            else
                echo "FAIL: nn_numeric_riscv32 (riscv32 output != nn_ref.py)"
                diff "$TMP/expected.txt" "$TMP/rvvals.txt" | head -10
            fi
        fi
    fi
fi

# ---------------------------------------------------------------------------
# 5. Cross-compilation of the library itself (no qemu, no python needed)
# ---------------------------------------------------------------------------
BUILD_OK=1
BUILD_MSG=""
for SPEC in "xtensa:$DIR/nn_mcu.mlr" "riscv32:$DIR/nn_mcu_riscv.mlr"; do
    A="${SPEC%%:*}"
    F="${SPEC#*:}"
    if $MLRC --arch="$A" --freestanding "$F" -o "$TMP/xc_$A.bin" \
            > "$TMP/xc_$A.log" 2>&1; then
        BUILD_MSG="$BUILD_MSG $A($(wc -c < "$TMP/xc_$A.bin" | tr -d ' ')B)"
    else
        BUILD_OK=0
        echo "  --- $A build log ---"
        head -5 "$TMP/xc_$A.log"
    fi
done
if [ "$BUILD_OK" = 1 ]; then
    echo "PASS: nn_build_mcu (freestanding images build:$BUILD_MSG)"
else
    echo "FAIL: nn_build_mcu"
fi
