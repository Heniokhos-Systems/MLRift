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

# --- the REAL-SILICON driver builds, and its model lands where it says ------
# tests/nn/nn_model_esp32.mlr hardcodes 0x3FFB0000 for the model because
# `--model` puts the blob at the bottom of the DRAM window and moves .data up
# above it. Nothing in CI can boot an ESP32, so this asserts the two things
# that would silently break that contract: that the driver still compiles for
# --target=esp32 (a 73728-byte arena + an 89 KB model has to FIT the 192 KiB
# window), and that segment 0 really is the blob at 0x3FFB0000.
ESP_BLOB="$TMP/esp_blob.bin"
ESP_IMG="$TMP/nn_esp32.bin"
head -c 89304 /dev/zero > "$ESP_BLOB" 2>/dev/null
if ! $MLRC --arch=xtensa --freestanding --target=esp32 --model "$ESP_BLOB" \
        "$DIR/nn_model_esp32.mlr" -o "$ESP_IMG" > "$TMP/esp32.log" 2>&1; then
    echo "FAIL: nn_build_esp32 (build failed)"
    head -5 "$TMP/esp32.log"
else
    ESP_NSEG=$(od -An -tu1 -j 1 -N 1 "$ESP_IMG" | tr -d ' ')
    ESP_S0=$(od -An -tu4 -j 24 -N 4 "$ESP_IMG" | tr -d ' ')
    ESP_S0LEN=$(od -An -tu4 -j 28 -N 4 "$ESP_IMG" | tr -d ' ')
    if [ "$ESP_NSEG" = "3" ] && [ "$ESP_S0" = "$((0x3FFB0000))" ] \
       && [ "$ESP_S0LEN" = "89304" ]; then
        echo "PASS: nn_build_esp32 ($(wc -c < "$ESP_IMG" | tr -d ' ')B image," \
             "89304B model segment at 0x3FFB0000 where the driver reads it)"
    else
        echo "FAIL: nn_build_esp32 (segs=$ESP_NSEG seg0=$ESP_S0 len=$ESP_S0LEN," \
             "want 3 / $((0x3FFB0000)) / 89304)"
    fi
fi

# ---------------------------------------------------------------------------
# 5b. Sub-range equivalence of the partitioned kernel variants
# ---------------------------------------------------------------------------
# Everything above only ever calls the kernels over their FULL range, so a
# `_rows`/`_range` variant that is wrong for a proper SUB-range passes all of
# it and first misbehaves once the work is actually split across two cores.
# tests/nn/nn_split_equiv.mlr closes that gap on the host: for every split
# point k it asserts kernel(0,h) == kernel(0,k) then kernel(k,h), byte for
# byte, over every converted kernel.
#
# The harness is run TWICE. The second run passes an argument that deliberately
# shifts one band boundary, which must make it fail — a check that has never
# been observed failing proves nothing, so a perturbed run that comes back
# green means the harness itself is broken and this reports FAIL.
if ! $MLRC --arch="$(uname -m)" "$DIR/nn_split_equiv.mlr" -o "$TMP/nn_se" \
        > "$TMP/sebuild.txt" 2>&1; then
    echo "FAIL: nn_split_equiv (build failed)"
    head -5 "$TMP/sebuild.txt"
else
    chmod +x "$TMP/nn_se"
    SE_CLEAN_OUT=$("$TMP/nn_se" 2>&1); SE_CLEAN_RC=$?
    "$TMP/nn_se" perturb > "$TMP/se_perturb.txt" 2>&1; SE_DIRTY_RC=$?
    SE_N=$(printf '%s\n' "$SE_CLEAN_OUT" | sed -n 's/^checks=\([0-9]*\).*/\1/p')
    if [ "$SE_CLEAN_RC" != 0 ]; then
        echo "FAIL: nn_split_equiv (a split differs from the full range)"
        printf '%s\n' "$SE_CLEAN_OUT" | head -10
    elif [ "$SE_DIRTY_RC" = 0 ]; then
        echo "FAIL: nn_split_equiv (negative control passed — the harness cannot detect a bad band)"
    else
        echo "PASS: nn_split_equiv ($SE_N splits byte-identical to the full range;" \
             "perturbed control correctly fails)"
    fi
fi

# ---------------------------------------------------------------------------
# 6. The model runner (std/nn_model.mlr) on a synthetic .krnn
# ---------------------------------------------------------------------------
# tests/nn/nn_model_ref.py builds a small model covering every opcode and
# says what running it must print. It needs only the standard library — no
# ONNX, no numpy, no model file — so this runs everywhere the kernel test
# does. The 106-node YuNet check that proves the NUMBERS on a real graph is
# opt-in below via NN_KRNN_MODEL.
#
# The model is NOT baked into the MCU images: qemu's generic loader drops it
# at a fixed address, which is what a flash partition is on real silicon.
# xtensa's loader address is PHYSICAL (0x00800000, read back through the
# dc232b cached window at 0xd0800000); riscv32 machine mode has no
# translation so 0x81000000 is both.
XT_MODEL_PADDR=0x00800000
RV_MODEL_PADDR=0x81000000

run_krnn() {
    # run_krnn <label> <model.krnn> <expected.txt>
    local label="$1" model="$2" expected="$3"
    local n
    n=$(wc -l < "$expected" | tr -d ' ')

    if ! $MLRC --arch="$(uname -m)" "$DIR/nn_model_host.mlr" \
            -o "$TMP/nm_host" > "$TMP/nmhost.txt" 2>&1; then
        echo "FAIL: ${label}_host (build failed)"
        head -5 "$TMP/nmhost.txt"
        return
    fi
    chmod +x "$TMP/nm_host"
    "$TMP/nm_host" "$model" > "$TMP/nm_host.txt" 2>&1
    if cmp -s "$expected" "$TMP/nm_host.txt"; then
        echo "PASS: ${label}_host ($n values exactly equal to the reference)"
    else
        echo "FAIL: ${label}_host (output != reference)"
        diff "$expected" "$TMP/nm_host.txt" | head -10
    fi

    if ! command -v qemu-system-xtensa > /dev/null 2>&1; then
        echo "SKIP: ${label}_xtensa (no qemu-system-xtensa)"
    elif ! $MLRC --arch=xtensa --freestanding "$DIR/nn_model_mcu.mlr" \
            -o "$TMP/nm_xt.elf" > "$TMP/nmxt.txt" 2>&1; then
        echo "FAIL: ${label}_xtensa (build failed)"
        head -5 "$TMP/nmxt.txt"
    else
        qemu_until_end "$TMP/nm_xt.txt" \
            qemu-system-xtensa -M lx60 -nographic -kernel "$TMP/nm_xt.elf" \
            -device "loader,file=$model,addr=$XT_MODEL_PADDR,force-raw=on"
        head -n "$n" "$TMP/nm_xt.txt" > "$TMP/nm_xtvals.txt"
        if ! grep -q '^END$' "$TMP/nm_xt.txt"; then
            echo "FAIL: ${label}_xtensa (image did not reach END under qemu)"
        elif cmp -s "$expected" "$TMP/nm_xtvals.txt"; then
            echo "PASS: ${label}_xtensa ($n values byte-identical to host and reference)"
        else
            echo "FAIL: ${label}_xtensa (output != reference)"
            diff "$expected" "$TMP/nm_xtvals.txt" | head -10
        fi
    fi

    if ! command -v qemu-system-riscv32 > /dev/null 2>&1; then
        echo "SKIP: ${label}_riscv32 (no qemu-system-riscv32)"
    elif ! $MLRC --arch=riscv32 --freestanding "$DIR/nn_model_mcu_riscv.mlr" \
            -o "$TMP/nm_rv.bin" > "$TMP/nmrv.txt" 2>&1; then
        echo "FAIL: ${label}_riscv32 (build failed)"
        head -5 "$TMP/nmrv.txt"
    else
        qemu_until_end "$TMP/nm_rv.txt" \
            qemu-system-riscv32 -M virt -nographic -bios "$TMP/nm_rv.bin" \
            -device "loader,file=$model,addr=$RV_MODEL_PADDR,force-raw=on"
        head -n "$n" "$TMP/nm_rv.txt" > "$TMP/nm_rvvals.txt"
        if ! grep -q '^END$' "$TMP/nm_rv.txt"; then
            echo "FAIL: ${label}_riscv32 (image did not reach END under qemu)"
        elif cmp -s "$expected" "$TMP/nm_rvvals.txt"; then
            echo "PASS: ${label}_riscv32 ($n values byte-identical to host and reference)"
        else
            echo "FAIL: ${label}_riscv32 (output != reference)"
            diff "$expected" "$TMP/nm_rvvals.txt" | head -10
        fi
    fi
}

if ! command -v python3 > /dev/null 2>&1; then
    echo "SKIP: nn_model_host (no python3 for the reference)"
    echo "SKIP: nn_model_xtensa (no python3 for the reference)"
    echo "SKIP: nn_model_riscv32 (no python3 for the reference)"
elif ! python3 "$DIR/nn_model_ref.py" "$TMP/smoke.krnn" \
        > "$TMP/smoke_expected.txt" 2> "$TMP/smokeerr.txt"; then
    echo "FAIL: nn_model_host (reference nn_model_ref.py failed)"
    head -5 "$TMP/smokeerr.txt"
else
    run_krnn nn_model "$TMP/smoke.krnn" "$TMP/smoke_expected.txt"
fi

# ---------------------------------------------------------------------------
# 7. OPT-IN: the same runner on a real converted graph
# ---------------------------------------------------------------------------
# Set NN_KRNN_MODEL to a .krnn produced by
#   tools/ml/onnx_to_mlrift.py <model.onnx> -o <pfx> \
#       --emit-binary <out.krnn> --ranges <ranges.json>
# and NN_KRNN_EXPECTED to the matching int8_sim.py dump rendered one decimal
# per line (tools/ml/README.md spells out the two commands). Skipped when
# unset, because the ONNX model and its calibration are not in this repo.
if [ -n "$NN_KRNN_MODEL" ] && [ -n "$NN_KRNN_EXPECTED" ]; then
    run_krnn nn_krnn "$NN_KRNN_MODEL" "$NN_KRNN_EXPECTED"
fi
