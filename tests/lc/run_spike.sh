#!/bin/bash
# MLRift -- living-compiler verified-migrations, Task 1
#
# Re-entrancy spike: can compile() run twice on different sources in one
# process? Pins std/hip.mlr (17 @dynamic declarations) against
# std/sha256.mlr (none) -- see task-1-brief.md for why this pairing
# matters. Do not substitute inputs.
#
# Two boundaries are asserted here, both discovered empirically (see
# task-1-report.md):
#
#   1. --emit=obj (emit_mode=3) is safe by construction: compile()
#      returns before either dyn_sym_count-gated branch in src/main.mlr
#      (:3108, :3239), both of which require emit_mode==0. Both
#      orderings therefore MATCH under obj mode; only one is checked
#      here since the code path can't distinguish them.
#
#   2. --emit=elfexe (emit_mode=0, the real default build path) leaks
#      dyn_sym_count across compile() calls because dyn_sym_init() only
#      runs once per process (guarded by `dyn_sym_tokens == 0`). The
#      leak is ONE-DIRECTIONAL: a @dynamic-declaring module compiled
#      before a clean module corrupts the clean module's output (hip ->
#      sha256: sha256 gets routed down the dynamic-ELF path it should
#      never take). The reverse order does not corrupt anything (sha256
#      -> hip: sha256 contributes zero symbols, so nothing leaks
#      forward, and hip's own registrations are unaffected).
set -e
MLRC=./build/mlrc
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORK=$(mktemp -d)

# Driver files live outside the repo (mktemp -d), but MLRift resolves
# `import "std/X.mlr"` relative to the *importing file's* directory, not
# cwd. Without this symlink, that lookup misses and falls back to
# mlrc's installed-stdlib search paths (e.g. ~/.local/share/mlrift/),
# which may be a stale copy that silently diverges from this worktree's
# std/ -- or may be missing the module entirely (observed: no
# std/sha256.mlr installed there at all, only std/hip.mlr). Symlinking
# guarantees the spike measures *this worktree's* std/hip.mlr and
# std/sha256.mlr, not whatever happens to be installed on the host.
ln -s "$REPO_ROOT/std" "$WORK/std"

FAIL=0

# check <label> <expected: MATCH|DIFFER> <fileA> <fileB>
check() {
  local label="$1" expected="$2" a="$3" b="$4"
  local actual
  if cmp -s "$a" "$b"; then actual=MATCH; else actual=DIFFER; fi
  if [ "$actual" = "$expected" ]; then
    echo "  $label: $actual (expected)"
  else
    echo "  $label: $actual -- MISMATCH, expected $expected"
    FAIL=1
  fi
}

for M in hip sha256; do
  printf 'import "std/%s.mlr"\nfn main() { exit(0) }\n' "$M" > "$WORK/drv_$M.mlr"
done

# --- Boundary 1: --emit=obj (emit_mode=3), ordering hip -> sha256 -------

for M in hip sha256; do
  $MLRC --arch=x86_64 --emit=obj "$WORK/drv_$M.mlr" -o "$WORK/ref_$M.o" >/dev/null 2>&1
done
echo "reference objects built (--emit=obj):"
for M in hip sha256; do
  echo "  ref_$M.o: $(md5sum "$WORK/ref_$M.o" | cut -d' ' -f1)"
done

sed 's/^fn main()/fn orig_main()/' build/mlrc.mlr > "$WORK/twice_obj.mlr"
cat >> "$WORK/twice_obj.mlr" <<EOF
fn main() {
    compile("$WORK/drv_hip.mlr", "$WORK/obj_hip.o", 0, 3)
    compile("$WORK/drv_sha256.mlr", "$WORK/obj_sha256.o", 0, 3)
    exit(0)
}
EOF
$MLRC --arch=x86_64 "$WORK/twice_obj.mlr" -o "$WORK/twice_obj" >/dev/null 2>&1
chmod +x "$WORK/twice_obj" && "$WORK/twice_obj"

echo "obj mode (--emit=obj), two-in-process, ordering hip -> sha256:"
check "hip.o   " MATCH "$WORK/ref_hip.o" "$WORK/obj_hip.o"
check "sha256.o" MATCH "$WORK/ref_sha256.o" "$WORK/obj_sha256.o"

# --- Boundary 2: --emit=elfexe (emit_mode=0), both orderings ------------

for M in hip sha256; do
  $MLRC --arch=x86_64 --emit=elfexe "$WORK/drv_$M.mlr" -o "$WORK/ref_${M}_exe" >/dev/null 2>&1
done
echo "reference executables built (--emit=elfexe):"
for M in hip sha256; do
  echo "  ref_${M}_exe: $(stat -c%s "$WORK/ref_${M}_exe") bytes"
done

# Ordering A: hip -> sha256 (the @dynamic module runs first).
sed 's/^fn main()/fn orig_main()/' build/mlrc.mlr > "$WORK/twice_exe_a.mlr"
cat >> "$WORK/twice_exe_a.mlr" <<EOF
fn main() {
    compile("$WORK/drv_hip.mlr", "$WORK/a_hip_exe", 0, 0)
    compile("$WORK/drv_sha256.mlr", "$WORK/a_sha256_exe", 0, 0)
    exit(0)
}
EOF
$MLRC --arch=x86_64 "$WORK/twice_exe_a.mlr" -o "$WORK/twice_exe_a" >/dev/null 2>&1
chmod +x "$WORK/twice_exe_a" && "$WORK/twice_exe_a" > "$WORK/twice_exe_a.log" 2>&1

# Ordering B: sha256 -> hip (the clean module runs first).
sed 's/^fn main()/fn orig_main()/' build/mlrc.mlr > "$WORK/twice_exe_b.mlr"
cat >> "$WORK/twice_exe_b.mlr" <<EOF
fn main() {
    compile("$WORK/drv_sha256.mlr", "$WORK/b_sha256_exe", 0, 0)
    compile("$WORK/drv_hip.mlr", "$WORK/b_hip_exe", 0, 0)
    exit(0)
}
EOF
$MLRC --arch=x86_64 "$WORK/twice_exe_b.mlr" -o "$WORK/twice_exe_b" >/dev/null 2>&1
chmod +x "$WORK/twice_exe_b" && "$WORK/twice_exe_b" > "$WORK/twice_exe_b.log" 2>&1

echo "elfexe mode, two-in-process, ordering hip -> sha256:"
echo "  a_hip_exe:    $(stat -c%s "$WORK/a_hip_exe") bytes"
echo "  a_sha256_exe: $(stat -c%s "$WORK/a_sha256_exe") bytes (ref is $(stat -c%s "$WORK/ref_sha256_exe") bytes)"
check "hip_exe   " MATCH  "$WORK/ref_hip_exe" "$WORK/a_hip_exe"
check "sha256_exe" DIFFER "$WORK/ref_sha256_exe" "$WORK/a_sha256_exe"

echo "elfexe mode, two-in-process, ordering sha256 -> hip:"
echo "  b_sha256_exe: $(stat -c%s "$WORK/b_sha256_exe") bytes"
echo "  b_hip_exe:    $(stat -c%s "$WORK/b_hip_exe") bytes"
check "sha256_exe" MATCH "$WORK/ref_sha256_exe" "$WORK/b_sha256_exe"
check "hip_exe   " MATCH "$WORK/ref_hip_exe" "$WORK/b_hip_exe"

# --- Boundary 3: the verification harness's OWN call pattern (Task 6) ----
#
# Boundary 1 above measures two compiles. The Task 5 harness makes FOUR per
# build unit (original + rewrite, x86_64 + arm64), and Task 6 lets one file
# belong to several units (src/bcj.mlr is in both build/mlrc.mlr and
# build/mlr-runner.mlr), so a single `lc --fix` run can compile several
# DIFFERENT sources, eight or more times, in one process. That is the exact
# shape of the emit_mode-0 hazard -- a @dynamic-declaring source compiled
# before a clean one -- so it is measured here rather than assumed safe by
# extrapolation from the two-compile case.
#
# The question this answers: does verifying a std/ module in-process need an
# outer per-compile driver process? At emit_mode 3 the answer is no.

for M in hip sha256; do
  $MLRC --arch=arm64 --emit=obj "$WORK/drv_$M.mlr" -o "$WORK/ref_${M}_arm.o" >/dev/null 2>&1
done

# gen_harness <outfile> <first module> <second module>
# Emits the mig_verify_unit sequence for two units, in the given order.
gen_harness() {
  local out="$1" first="$2" second="$3"
  sed 's/^fn main()/fn orig_main()/' build/mlrc.mlr > "$out.mlr"
  cat >> "$out.mlr" <<EOF
fn main() {
    compile("$WORK/drv_$first.mlr", "$WORK/h_${first}_x86_a.o", 0, 3)
    compile("$WORK/drv_$first.mlr", "$WORK/h_${first}_x86_b.o", 0, 3)
    compile("$WORK/drv_$first.mlr", "$WORK/h_${first}_arm_a.o", 1, 3)
    compile("$WORK/drv_$first.mlr", "$WORK/h_${first}_arm_b.o", 1, 3)
    compile("$WORK/drv_$second.mlr", "$WORK/h_${second}_x86_a.o", 0, 3)
    compile("$WORK/drv_$second.mlr", "$WORK/h_${second}_x86_b.o", 0, 3)
    compile("$WORK/drv_$second.mlr", "$WORK/h_${second}_arm_a.o", 1, 3)
    compile("$WORK/drv_$second.mlr", "$WORK/h_${second}_arm_b.o", 1, 3)
    exit(0)
}
EOF
  $MLRC --arch=x86_64 "$out.mlr" -o "$out" >/dev/null 2>&1
  chmod +x "$out" && "$out" >/dev/null 2>&1
}

# Ordering A: the @dynamic unit first -- the direction that corrupts at mode 0.
gen_harness "$WORK/harness_a" hip sha256
echo "obj mode, EIGHT compiles in one process, ordering hip -> sha256:"
for M in hip sha256; do
  check "$M x86 #1" MATCH "$WORK/ref_$M.o"      "$WORK/h_${M}_x86_a.o"
  check "$M x86 #2" MATCH "$WORK/ref_$M.o"      "$WORK/h_${M}_x86_b.o"
  check "$M arm #1" MATCH "$WORK/ref_${M}_arm.o" "$WORK/h_${M}_arm_a.o"
  check "$M arm #2" MATCH "$WORK/ref_${M}_arm.o" "$WORK/h_${M}_arm_b.o"
done

# Ordering B: the clean unit first. Harmless at mode 0 too, kept as the
# control so an all-MATCH result above cannot be read as "the check is inert".
gen_harness "$WORK/harness_b" sha256 hip
echo "obj mode, EIGHT compiles in one process, ordering sha256 -> hip:"
for M in sha256 hip; do
  check "$M x86 #1" MATCH "$WORK/ref_$M.o"      "$WORK/h_${M}_x86_a.o"
  check "$M x86 #2" MATCH "$WORK/ref_$M.o"      "$WORK/h_${M}_x86_b.o"
  check "$M arm #1" MATCH "$WORK/ref_${M}_arm.o" "$WORK/h_${M}_arm_a.o"
  check "$M arm #2" MATCH "$WORK/ref_${M}_arm.o" "$WORK/h_${M}_arm_b.o"
done

echo
if [ "$FAIL" = 0 ]; then
  echo "PASS: boundary confirmed as expected -- obj mode is safe by construction (returns before"
  echo "the dyn_sym_count-gated branches); elfexe mode leaks dyn_sym_count one-directionally --"
  echo "a @dynamic module compiled BEFORE a clean module corrupts the clean module's output,"
  echo "the reverse order does not."
  echo
  echo "Boundary 3 (Task 6): the safety of obj mode holds at the verification harness's own"
  echo "scale -- eight compiles of two different sources in one process, both orderings, both"
  echo "targets, every object byte-identical to a single-shot reference. std/ therefore needs"
  echo "NO outer per-compile driver process; lc --fix verifies std/ modules in-process."
else
  echo "FAIL: re-entrancy behavior did not match the recorded boundary -- see MISMATCH lines above."
fi
# Keep the artifacts only when there is something to inspect -- the pattern
# prove_full_scale.sh uses. A passing run used to leave its mktemp -d behind.
if [ "$FAIL" = 0 ]; then
  rm -rf "$WORK"
else
  echo "artifacts kept for inspection: WORK=$WORK"
fi
exit "$FAIL"
