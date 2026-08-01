#!/bin/bash
# MLRift -- living-compiler verified-migrations, Task 1
#
# Re-entrancy spike: can compile() run twice on different sources in one
# process? Pins std/hip.mlr (18 @dynamic declarations) against
# std/sha256.mlr (none) -- see task-1-brief.md for why this pairing
# matters. Do not substitute inputs.
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

# Step 1/2: Reference -- each compiled alone, in its own process.
for M in hip sha256; do
  printf 'import "std/%s.mlr"\nfn main() { exit(0) }\n' "$M" > "$WORK/drv_$M.mlr"
  $MLRC --arch=x86_64 --emit=obj "$WORK/drv_$M.mlr" -o "$WORK/ref_$M.o" >/dev/null 2>&1
done
echo "reference objects built"
for M in hip sha256; do
  echo "  ref_$M.o: $(md5sum "$WORK/ref_$M.o" | cut -d' ' -f1)"
done

# Step 3: Two-in-one-process driver -- build/mlrc.mlr with main renamed
# and a new main() that calls compile() twice in a row.
sed 's/^fn main()/fn orig_main()/' build/mlrc.mlr > "$WORK/twice.mlr"
cat >> "$WORK/twice.mlr" <<'EOF'
fn main() {
    compile("A_PATH", "A_OUT", 0, 3)
    compile("B_PATH", "B_OUT", 0, 3)
    exit(0)
}
EOF
sed -i "s|A_PATH|$WORK/drv_hip.mlr|; s|A_OUT|$WORK/two_hip.o|; s|B_PATH|$WORK/drv_sha256.mlr|; s|B_OUT|$WORK/two_sha256.o|" "$WORK/twice.mlr"
$MLRC --arch=x86_64 "$WORK/twice.mlr" -o "$WORK/twice" && chmod +x "$WORK/twice" && "$WORK/twice"

# Step 4: Compare and report the boundary.
echo "comparison:"
for M in hip sha256; do
  if cmp -s "$WORK/ref_$M.o" "$WORK/two_$M.o"; then
    echo "  $M: MATCH"
  else
    echo "  $M: DIFFER"
  fi
done

echo "WORK=$WORK"
