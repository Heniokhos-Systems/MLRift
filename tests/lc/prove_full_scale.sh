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
#
# Step 4 is POSITION-AWARE, not line-based, and that distinction is the whole
# point. Of the 32,484 comment-bearing lines in build/mlrc.mlr only 25,029 are
# whole-line comments; the other ~7,455 carry code AND a trailing comment on
# the same line -- and those are the dangerous ones, because the line holds a
# token that MUST migrate right next to comment text that must NOT:
#
#     static uint64 ir_live_words = 0   // uint64 words per bitset = ...
#
# A whole-line-comment diff cannot see the right-hand `uint64` corrupted; nor
# can a comment-line count; nor can object identity, since comments never
# reach emitted code. So every check below splits each line at the first `//`
# that is OUTSIDE a string literal (four lines in this source contain `//`
# only inside a hip_emit_cstr string -- a naive split misclassifies them) and
# asserts what each half must satisfy: the comment half byte-identical, the
# code half holding zero long-form spellings afterwards.
set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# lc resolves the Makefile, the build-unit map and its own scratch relative
# to the CWD, so run from the repo root rather than wherever the caller is.
cd "$REPO_ROOT"
MLRC="$REPO_ROOT/build/mlrc"

if [ ! -x "$MLRC" ]; then
  echo "error: $MLRC not built -- run 'make' first"; exit 1
fi
if [ ! -f build/mlrc.mlr ]; then
  echo "error: build/mlrc.mlr missing -- run 'make' first"; exit 1
fi

WORK=$(mktemp -d)
FAIL=0
VERDICT=0   # set to 1 once a verdict line has been printed

# Preserve $WORK on ANY non-zero exit, not just on a recorded FAIL. `set -e`
# can abort mid-script (a compile that dies, a command not found) without any
# assertion having run, and that is exactly when the artifacts are worth
# keeping -- an earlier version deleted them in that case and printed nothing
# at all, so a real failure looked like silence.
finish() {
  local rc=$?
  if [ "$VERDICT" = 0 ] && { [ "$rc" != 0 ] || [ "$FAIL" != 0 ]; }; then
    echo
    echo "NOT PROVEN -- aborted with status $rc before reaching the verdict"
  fi
  if [ "$rc" = 0 ] && [ "$FAIL" = 0 ]; then
    rm -rf "$WORK"
  else
    echo "artifacts kept for inspection: WORK=$WORK"
  fi
}
trap finish EXIT

fail() { echo "  FAIL: $*"; FAIL=1; }

# run <what> <cmd...> -- every compiler invocation goes through this. An
# unguarded `"$MLRC" ... >/dev/null 2>&1` swallows both the error message and
# the reason, leaving only an exit code for someone to guess from.
run() {
  local what="$1"; shift
  if ! "$@" > "$WORK/last_cmd.log" 2>&1; then
    fail "$what failed:"
    tail -20 "$WORK/last_cmd.log" | sed 's/^/    /'
    return 1
  fi
}

# --- The position-aware line splitter ------------------------------------
#
# mode=comment : print "<lineno>:<W|T>:<comment text>" for every line that
#                has a real comment -- W when the comment is the whole line,
#                T when it trails code. The W/T flag rides in the compared
#                text on purpose: a comment that stayed byte-identical but
#                moved from trailing to whole-line means the code beside it
#                vanished, and that must not pass either.
# mode=code    : print "<lineno>:<code text>" with the comment AND string /
#                char literal CONTENTS removed, so a grep over it sees only
#                real code tokens.
# mode=check   : print how many lines end inside an unterminated literal.
#                MLRift string and char literals do not span lines in this
#                source; this asserts that assumption rather than trusting
#                it, because if it were false the per-line state machine
#                would misclassify everything after the offending line.
cat > "$WORK/split.awk" <<'AWK'
BEGIN { unterminated = 0 }
{
  line = $0
  n = length(line)
  st = 0            # 0 = code, 1 = "string", 2 = 'char'
  code = ""
  comment = ""
  i = 1
  while (i <= n) {
    c = substr(line, i, 1)
    if (st == 0) {
      if (c == "/" && substr(line, i + 1, 1) == "/") {
        comment = substr(line, i)
        i = n + 1
        continue
      }
      code = code c
      if (c == "\"") { st = 1 }
      else if (c == "'") { st = 2 }
      i = i + 1
    } else {
      # Inside a literal: contents are NOT code. A backslash escapes the next
      # byte, so \" and \' do not close the literal.
      if (c == "\\") { i = i + 2; continue }
      if ((st == 1 && c == "\"") || (st == 2 && c == "'")) { st = 0; code = code c }
      i = i + 1
    }
  }
  if (st != 0) { unterminated = unterminated + 1 }
  if (mode == "comment") {
    if (comment != "") {
      bare = code
      gsub(/[ \t]/, "", bare)
      print NR ":" (bare == "" ? "W" : "T") ":" comment
    }
  }
  else if (mode == "code") { print NR ":" code }
}
END { if (mode == "check") { print unterminated } }
AWK

split_mode() { awk -v mode="$1" -f "$WORK/split.awk" "$2"; }
LONGFORM='\b(uint8|uint16|uint32|uint64|int8|int16|int32|int64)\b'
# `grep -o | wc -l`, NOT `grep -c`: a line can hold several sites (`unsafe {
# *(p as uint64) -> v }`) and -c counts matching LINES, which undercounts by
# ~2,750 here. wc also terminates the pipeline, so grep's exit 1 on "no
# matches" cannot abort the script under `set -e` at exactly the assertion
# that wanted to read a zero.
count_longform_in_code() { split_mode code "$1" | grep -oE "$LONGFORM" | wc -l; }
count_longform_raw()     { grep -oE "$LONGFORM" "$1" | wc -l; }

# The one file the proof must not disturb. build/ is gitignored, so
# `git status` can never report a change to build/mlrc.mlr -- Step 3 needs a
# checksum, not just a porcelain check.
SRC_MD5_BEFORE=$(md5sum build/mlrc.mlr | cut -d' ' -f1)

UNTERM=$(split_mode check build/mlrc.mlr)
if [ "$UNTERM" != 0 ]; then
  fail "$UNTERM line(s) end inside an unterminated literal -- the splitter's per-line assumption does not hold and every check below is unreliable"
fi

cp build/mlrc.mlr "$WORK/unit.mlr"
echo "unit: build/mlrc.mlr -> \$WORK/unit.mlr ($(stat -c%s "$WORK/unit.mlr") bytes)"

# --- Step 2: rewrite the copy, compare emitted objects -------------------

for A in x86_64 arm64; do
  run "$A reference object" "$MLRC" --arch=$A --emit=obj "$WORK/unit.mlr" -o "$WORK/before_$A.o"
done

# What the site count MUST be, derived independently: every long-form
# spelling that appears in real code -- not in a comment, not inside a string
# literal -- in the ORIGINAL. Derived by the awk above, which shares nothing
# with the MLRift lexer the migration itself uses, so agreement between the
# two is a genuine cross-check rather than one implementation agreeing with
# itself. Exact equality also closes the ~700-site slack band a ">= 24000"
# floor would leave, inside which a whole class of missed sites could hide.
EXPECTED=$(count_longform_in_code build/mlrc.mlr)

run "lc --fix=types" "$MLRC" lc --fix=types "$WORK/unit.mlr"
cp "$WORK/last_cmd.log" "$WORK/fix.log"
SITES=$(grep -o '[0-9]* migration site' "$WORK/fix.log" | grep -o '[0-9]*' || true)
if [ -z "$SITES" ]; then
  echo "could not read a site count out of lc's output:"
  cat "$WORK/fix.log"
  FAIL=1
  exit 1
fi
echo "rewrote $SITES sites (independently derived expectation: $EXPECTED)"

# A count materially below ~24,000 means the scanner is not seeing some type
# kinds -- all four KwUint* AND all four signed KwInt* kinds must be handled,
# and the two families have DIFFERENT long-form lengths (5/6/6/6 vs 4/5/5/5).
if [ "$EXPECTED" -lt 24000 ]; then
  fail "the independent derivation itself found only $EXPECTED sites -- the splitter or the grep is wrong, not necessarily the migration"
elif [ "$SITES" != "$EXPECTED" ]; then
  fail "site count $SITES != the $EXPECTED derived from the source; check mig_long_form_len covers all four KwUint* AND all four KwInt* kinds"
fi

for A in x86_64 arm64; do
  run "$A object after migration" "$MLRC" --arch=$A --emit=obj "$WORK/unit.mlr" -o "$WORK/after_$A.o"
  if cmp -s "$WORK/before_$A.o" "$WORK/after_$A.o"; then
    echo "  $A: byte-identical ($(stat -c%s "$WORK/after_$A.o") bytes)"
  else
    fail "$A: object MISMATCH"
  fi
done

# Linked executables too. Not required by the criterion (--emit=obj is the
# contract the verifier itself checks) but it is nearly free, and it covers
# the layout and relocation stages a relocatable .o never reaches.
for A in x86_64 arm64; do
  run "$A reference executable"       "$MLRC" --arch=$A build/mlrc.mlr    -o "$WORK/exe_before_$A"
  run "$A executable after migration" "$MLRC" --arch=$A "$WORK/unit.mlr"  -o "$WORK/exe_after_$A"
  if cmp -s "$WORK/exe_before_$A" "$WORK/exe_after_$A"; then
    echo "  $A: linked executable byte-identical ($(stat -c%s "$WORK/exe_after_$A") bytes)"
  else
    fail "$A: linked executable MISMATCH"
  fi
done

# And the migrated compiler must still BE a compiler.
chmod +x "$WORK/exe_after_x86_64"
printf 'fn main() { println("ok") exit(0) }\n' > "$WORK/hello.mlr"
run "the migrated compiler compiling a program" \
    "$WORK/exe_after_x86_64" --arch=x86_64 "$WORK/hello.mlr" -o "$WORK/hello"
if [ "$("$WORK/hello" 2>&1)" = "ok" ]; then
  echo "  migrated compiler builds and runs a program"
else
  fail "the compiler built from migrated source does not work"
fi

# --- Step 3: the tracked tree must be untouched --------------------------

DIRTY=$(git status --short src/ build/mlrc.mlr || true)
if [ -n "$DIRTY" ]; then
  fail "the proof leaked into the tree:"
  echo "$DIRTY"
fi
SRC_MD5_AFTER=$(md5sum build/mlrc.mlr | cut -d' ' -f1)
if [ "$SRC_MD5_BEFORE" = "$SRC_MD5_AFTER" ]; then
  echo "  src/ clean, build/mlrc.mlr unchanged ($SRC_MD5_AFTER)"
else
  fail "build/mlrc.mlr was modified ($SRC_MD5_BEFORE -> $SRC_MD5_AFTER)"
fi

# --- Step 4: comments survived, at scale ---------------------------------

# THE assertion. Every comment, wherever it sits on its line, byte-for-byte,
# carrying line numbers so a failure names the line. This subsumes both a
# whole-line-comment diff and a comment-line count, and unlike either of them
# it covers the ~7,455 code-plus-trailing-comment lines. Verified to FIRE:
# corrupting only the trailing `// uint64 words per bitset` on line 28034 of
# the migrated file, leaving that line's code correctly migrated, is
# invisible to a whole-line diff, to a `^//` count, to a "leftover outside
# comments and strings" scan and to object identity -- and is caught here,
# naming line 28034.
split_mode comment build/mlrc.mlr   > "$WORK/comments_before.txt"
split_mode comment "$WORK/unit.mlr" > "$WORK/comments_after.txt"
NCOM=$(wc -l < "$WORK/comments_before.txt")
NWHOLE=$(grep -cE '^[0-9]+:W:' "$WORK/comments_before.txt" || true)
# Floor, cross-checked against a measure that shares no code with the splitter.
# Without this the diff below is vacuous-proof only by luck: if the splitter's
# comment branch ever emitted nothing, both sides would be empty and `diff -q`
# would report success -- a check that passes precisely because it stopped
# looking. grep is the independent witness. The two legitimately disagree by
# the handful of lines where `//` sits inside STRING DATA
# (hip_emit_cstr("// AUTO-GENERATED ...")), which grep counts and the splitter
# correctly does not, so allow a small gap rather than demanding equality.
NCOM_NAIVE=$(grep -c '//' build/mlrc.mlr || true)
if [ "$NCOM" -lt $(( NCOM_NAIVE - NCOM_NAIVE / 100 )) ]; then
  fail "comment splitter produced $NCOM comments but grep sees $NCOM_NAIVE lines containing '//'"
  echo "    the comment diff below would pass vacuously -- fix the splitter" >&2
fi
if diff -q "$WORK/comments_before.txt" "$WORK/comments_after.txt" >/dev/null; then
  echo "  all $NCOM comments byte-identical ($NWHOLE whole-line, $((NCOM - NWHOLE)) trailing)"
else
  fail "comment text changed:"
  diff "$WORK/comments_before.txt" "$WORK/comments_after.txt" | head -20 | sed 's/^/    /'
fi

# The counterpart assertion, on the other half of every line: after the
# migration ZERO long-form spellings may remain in code. Exact, not a
# heuristic -- the earlier "is there a // or a quote somewhere on this line"
# filter passed a corrupted trailing comment without noticing.
LEFT_CODE=$(count_longform_in_code "$WORK/unit.mlr")
LEFT_TOTAL=$(count_longform_raw "$WORK/unit.mlr")
if [ "$LEFT_CODE" = 0 ]; then
  echo "  0 long-form spellings left in code; the $LEFT_TOTAL that remain are all inside comments or string literals"
else
  fail "$LEFT_CODE long-form type keyword(s) still in real code:"
  split_mode code "$WORK/unit.mlr" | grep -E "$LONGFORM" | head -10 | sed 's/^/    /'
fi

# Spot-checks pinned to named comments -- one whole-line, one TRAILING -- so
# they cannot quietly start checking something else.
for PAT in '// VarDecl: uint64 IDENT = START' '// uint64 words per bitset'; do
  if grep -qF "$PAT" "$WORK/unit.mlr"; then
    echo "  comment still reads uint64: $PAT"
  else
    fail "a comment containing 'uint64' did not survive: $PAT"
  fi
done

# The exact site the old byte-scanner corrupted. `"uint64", 6` must still be
# a six-character literal paired with the length 6.
if grep -qF 'match_keyword(start, len, "uint64", 6)' "$WORK/unit.mlr"; then
  echo "  the lexer's own match_keyword(start, len, \"uint64\", 6) is intact"
else
  fail "the lexer keyword table was corrupted -- the original defect is back"
fi

echo
VERDICT=1
if [ "$FAIL" = 0 ]; then
  echo "PROVEN: $SITES sites, emitted code unchanged on both targets"
else
  echo "NOT PROVEN -- see FAIL lines above"
fi
exit "$FAIL"
