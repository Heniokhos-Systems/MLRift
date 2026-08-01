#!/bin/bash
# MLRift -- living-compiler verified-migrations, Task 7
#
# The acceptance criterion: run the type-alias migration at FULL SCALE --
# every long-form uint8/16/32/64 and int8/16/32/64 keyword in the whole
# compiler, ~24.7k sites -- and prove the compiler emits byte-identical
# code afterwards.
#
# Two rules this script exists to honour:
#
#   1. PROVE IT ON A COPY. A migrated src/ must never be committed. The
#      migration runs on $WORK/unit.mlr, a mktemp -d copy of build/mlrc.mlr;
#      Step 3 asserts the tracked tree came through untouched.
#
#   2. OBJECT IDENTITY IS NOT THE WHOLE PROOF. Comparing .o files covers
#      emitted code and string data, but a comment is invisible to it -- and
#      comment corruption is the ORIGINAL DEFECT this project exists to close
#      (the old byte-scanner rewrote the literal "uint64" inside
#      match_keyword(start, len, "uint64", 6), and the length argument 6
#      stayed behind). Step 4 therefore asserts the comments directly.
set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# lc resolves the Makefile, the build-unit map and its own scratch relative
# to the CWD, so run from the repo root rather than wherever the caller is.
cd "$REPO_ROOT"
MLRC="$REPO_ROOT/build/mlrc"
WORK=$(mktemp -d)
FAIL=0

finish() {
  if [ "$FAIL" = 0 ]; then
    rm -rf "$WORK"
  else
    echo "artifacts kept for inspection: WORK=$WORK"
  fi
}
trap finish EXIT

fail() { echo "  FAIL: $*"; FAIL=1; }

if [ ! -x "$MLRC" ]; then
  echo "error: $MLRC not built -- run 'make' first"; FAIL=1; exit 1
fi
if [ ! -f build/mlrc.mlr ]; then
  echo "error: build/mlrc.mlr missing -- run 'make' first"; FAIL=1; exit 1
fi

# The one file the proof must not disturb. build/ is gitignored, so
# `git status` can never report a change to build/mlrc.mlr -- Step 3 needs a
# checksum, not just a porcelain check.
SRC_MD5_BEFORE=$(md5sum build/mlrc.mlr | cut -d' ' -f1)

cp build/mlrc.mlr "$WORK/unit.mlr"
echo "unit: build/mlrc.mlr -> \$WORK/unit.mlr ($(stat -c%s "$WORK/unit.mlr") bytes)"

# --- Step 2: rewrite the copy, compare emitted objects -------------------

for A in x86_64 arm64; do
  "$MLRC" --arch=$A --emit=obj "$WORK/unit.mlr" -o "$WORK/before_$A.o" >/dev/null 2>&1
done

"$MLRC" lc --fix=types "$WORK/unit.mlr" > "$WORK/fix.log" 2>&1 || {
  echo "lc --fix=types failed:"; cat "$WORK/fix.log"; FAIL=1; exit 1; }
SITES=$(grep -o '[0-9]* migration site' "$WORK/fix.log" | grep -o '[0-9]*' || true)
if [ -z "$SITES" ]; then
  echo "could not read a site count out of lc's output:"; cat "$WORK/fix.log"
  FAIL=1; exit 1
fi
echo "rewrote $SITES sites"

# The guard that matters. A count materially below ~24,000 means the scanner
# is not seeing some type kinds -- all four KwUint* AND all four signed
# KwInt* kinds must be handled, and the two families have DIFFERENT long-form
# lengths (5/6/6/6 vs 4/5/5/5).
if [ "$SITES" -lt 24000 ]; then
  fail "only $SITES sites -- expected ~24k; check mig_long_form_len covers all four KwUint* AND all four KwInt* kinds"
fi

for A in x86_64 arm64; do
  "$MLRC" --arch=$A --emit=obj "$WORK/unit.mlr" -o "$WORK/after_$A.o" >/dev/null 2>&1
  if cmp -s "$WORK/before_$A.o" "$WORK/after_$A.o"; then
    echo "  $A: byte-identical ($(stat -c%s "$WORK/after_$A.o") bytes)"
  else
    fail "$A: object MISMATCH"
  fi
done

# Linked executables too. Not required by the criterion (--emit=obj is the
# contract the verifier itself checks) but it is nearly free, and it covers
# the layout/linking stages that a relocatable .o does not reach.
for A in x86_64 arm64; do
  "$MLRC" --arch=$A build/mlrc.mlr  -o "$WORK/exe_before_$A" >/dev/null 2>&1
  "$MLRC" --arch=$A "$WORK/unit.mlr" -o "$WORK/exe_after_$A" >/dev/null 2>&1
  if cmp -s "$WORK/exe_before_$A" "$WORK/exe_after_$A"; then
    echo "  $A: linked executable byte-identical ($(stat -c%s "$WORK/exe_after_$A") bytes)"
  else
    fail "$A: linked executable MISMATCH"
  fi
done

# And the migrated compiler must still BE a compiler.
chmod +x "$WORK/exe_after_x86_64"
printf 'fn main() { println("ok") exit(0) }\n' > "$WORK/hello.mlr"
"$WORK/exe_after_x86_64" --arch=x86_64 "$WORK/hello.mlr" -o "$WORK/hello" >/dev/null 2>&1
if [ "$("$WORK/hello" 2>&1)" = "ok" ]; then
  echo "  migrated compiler builds and runs a program"
else
  fail "the compiler built from migrated source does not work"
fi

# --- Step 3: the tracked tree must be untouched --------------------------

DIRTY=$(git status --short src/ build/mlrc.mlr)
if [ -n "$DIRTY" ]; then
  fail "the proof leaked into the tree:"; echo "$DIRTY"
fi
SRC_MD5_AFTER=$(md5sum build/mlrc.mlr | cut -d' ' -f1)
if [ "$SRC_MD5_BEFORE" = "$SRC_MD5_AFTER" ]; then
  echo "  src/ clean, build/mlrc.mlr unchanged ($SRC_MD5_AFTER)"
else
  fail "build/mlrc.mlr was modified ($SRC_MD5_BEFORE -> $SRC_MD5_AFTER)"
fi

# --- Step 4: comments survived, at scale ---------------------------------

C_BEFORE=$(grep -c '^//' build/mlrc.mlr)
C_AFTER=$(grep -c '^//' "$WORK/unit.mlr")
if [ "$C_BEFORE" = "$C_AFTER" ]; then
  echo "  $C_AFTER top-level comment lines, count unchanged"
else
  fail "comment line count changed: $C_BEFORE -> $C_AFTER"
fi

# Stronger than a count: a whole-line comment contains no code, so every one
# of them must come through byte-for-byte. This is what catches a rewriter
# that edits inside a comment without changing how many there are.
if diff -q <(grep -E '^[[:space:]]*//' build/mlrc.mlr) \
           <(grep -E '^[[:space:]]*//' "$WORK/unit.mlr") >/dev/null; then
  echo "  $(grep -cE '^[[:space:]]*//' build/mlrc.mlr) whole-line comments byte-identical"
else
  fail "whole-line comment text changed"
fi

# The spot-check the brief asks for, pinned to specific comments rather than
# whichever one happens to sort first.
for PAT in '// VarDecl: uint64 IDENT = START' '// defaulting to uint64\*\.'; do
  if grep -qE "$PAT" "$WORK/unit.mlr"; then
    echo "  comment still reads uint64: $(grep -oE "$PAT" "$WORK/unit.mlr" | head -1)"
  else
    fail "a comment containing 'uint64' did not survive: $PAT"
  fi
done

# Every long-form spelling that REMAINS must be inside a comment or a string
# literal -- those are exactly the places the migration must not touch. If a
# leftover shows up on a line with neither, the scanner skipped real code.
LEFT=$(grep -oE '\b(uint8|uint16|uint32|uint64|int8|int16|int32|int64)\b' "$WORK/unit.mlr" | wc -l)
STRAY=$(grep -E '\b(uint8|uint16|uint32|uint64|int8|int16|int32|int64)\b' "$WORK/unit.mlr" \
        | grep -v '//' | grep -v '"' | wc -l)
if [ "$STRAY" = 0 ]; then
  echo "  $LEFT long-form spellings left, all inside comments or string literals"
else
  fail "$STRAY line(s) still hold a long-form type keyword outside any comment or string"
  grep -nE '\b(uint8|uint16|uint32|uint64|int8|int16|int32|int64)\b' "$WORK/unit.mlr" \
    | grep -v '//' | grep -v '"' | head -10
fi

# The exact site the old byte-scanner corrupted. `"uint64", 6` must still be
# a six-character literal paired with the length 6.
if grep -q 'match_keyword(start, len, "uint64", 6)' "$WORK/unit.mlr"; then
  echo "  the lexer's own match_keyword(start, len, \"uint64\", 6) is intact"
else
  fail "the lexer keyword table was corrupted -- the original defect is back"
fi

echo
if [ "$FAIL" = 0 ]; then
  echo "PROVEN: $SITES sites, emitted code unchanged on both targets"
else
  echo "NOT PROVEN -- see FAIL lines above"
fi
exit "$FAIL"
