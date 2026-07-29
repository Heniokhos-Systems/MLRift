#!/bin/bash
# Update every hand-maintained packaging manifest to a published release.
#
#   scripts/update-packaging.sh [version]      # default: version in src/main.mlr
#
# Touches:
#   packaging/aur/PKGBUILD         pkgver + per-arch sha256sums arrays
#   packaging/aur/.SRCINFO         regenerated from the PKGBUILD
#   packaging/homebrew/mlrift.rb   version + the eight binary sha256s + the test
#   bucket/mlrift.json             version + both Windows zip urls and hashes
#
# Binary hashes come from the release's own SHA256SUMS asset, so they are the
# hashes of the artifacts users actually download -- not of a local rebuild that
# merely ought to match. std/*.mlr hashes are computed from the working tree,
# which is what raw.githubusercontent serves for the same tag.
#
# The release must already be published; SHA256SUMS is produced by release.yml.
set -euo pipefail

REPO="${REPO:-Heniokhos-Systems/MLRift}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

VERSION="${1:-}"
if [ -z "$VERSION" ]; then
    VERSION=$(grep -oE 'mlrc [0-9]+\.[0-9]+\.[0-9]+ \(MLRift' src/main.mlr | head -1 | awk '{print $2}')
fi
if ! printf '%s' "$VERSION" | grep -qE '^[0-9]+\.[0-9]+\.[0-9]+$'; then
    echo "FAIL: could not determine a version (got '$VERSION'); pass one explicitly" >&2
    exit 1
fi
echo "=== updating packaging manifests to v$VERSION ==="

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

curl --fail --location --silent --show-error \
     --retry 5 --retry-delay 3 --retry-all-errors --connect-timeout 15 \
     -o "$TMP/SHA256SUMS" \
     "https://github.com/$REPO/releases/download/v$VERSION/SHA256SUMS"

# A 404 page saved as SHA256SUMS would sail through the rewrite and poison every
# manifest with garbage hashes, so require the entries we depend on.
for want in mlrc-linux-x86_64 mlrc-linux-arm64 mlr-linux-x86_64 mlr-linux-arm64 \
            mlrc-macos-x86_64 mlrc-macos-arm64 mlr-macos-x86_64 mlr-macos-arm64 \
            mlrc-windows-x86_64.zip mlrc-windows-arm64.zip LICENSE; do
    grep -qE "^[0-9a-f]{64}  $want\$" "$TMP/SHA256SUMS" || {
        echo "FAIL: SHA256SUMS has no valid entry for '$want'" >&2
        echo "--- first 200 bytes ---" >&2; head -c 200 "$TMP/SHA256SUMS" >&2; echo >&2
        exit 1
    }
done
echo "  fetched SHA256SUMS ($(wc -l < "$TMP/SHA256SUMS") entries)"

VERSION="$VERSION" SUMS="$TMP/SHA256SUMS" python3 - <<'PY'
import json, os, re, hashlib, sys

version = os.environ['VERSION']
sums = {}
for line in open(os.environ['SUMS']):
    line = line.strip()
    if not line:
        continue
    h, name = line.split(None, 1)
    sums[name.strip()] = h

def sha_local(path):
    return hashlib.sha256(open(path, 'rb').read()).hexdigest()

changed = []

# --- AUR PKGBUILD -------------------------------------------------------
p = 'packaging/aur/PKGBUILD'
s = open(p).read()
s = re.sub(r'^pkgver=.*$', f'pkgver={version}', s, count=1, flags=re.M)

std = re.search(r'^_std=\((.*?)\)$', s, flags=re.M).group(1).split()

def sums_block(arch_mlrc, arch_mlr):
    rows = [(sums[arch_mlrc], 'mlrc'), (sums[arch_mlr], 'mlr')]
    for m in std:
        rows.append((sha_local(f'std/{m}.mlr'), m))
    rows.append((sums['LICENSE'], 'LICENSE'))
    return '\n'.join(f"    '{h}'  # {n}" for h, n in rows)

for arr, mlrc, mlr in (('sha256sums_x86_64', 'mlrc-linux-x86_64', 'mlr-linux-x86_64'),
                        ('sha256sums_aarch64', 'mlrc-linux-arm64', 'mlr-linux-arm64')):
    new = f"{arr}=(\n{sums_block(mlrc, mlr)}\n)"
    # Assert the pattern MATCHED, not that the text changed: re-running against
    # an already-current manifest is a legitimate no-op, and the workflow relies
    # on that to decide there is nothing to commit.
    s, n = re.subn(rf'^{arr}=\(.*?^\)', lambda _m: new, s, count=1, flags=re.M | re.S)
    assert n == 1, f'{arr} block not found in PKGBUILD'
open(p, 'w').write(s); changed.append(p)

# --- Homebrew formula ---------------------------------------------------
p = 'packaging/homebrew/mlrift.rb'
s = open(p).read()
s = re.sub(r'^(\s*)version "[^"]+"', rf'\g<1>version "{version}"', s, count=1, flags=re.M)
s = re.sub(r'assert_match "[0-9]+\.[0-9]+\.[0-9]+"',
           f'assert_match "{version}"', s)

# Each `url "#{REL}/<asset>"` is followed by the sha256 of that exact asset.
def fix_pair(m):
    asset = m.group('asset')
    if asset not in sums:
        raise SystemExit(f'FAIL: homebrew references {asset}, absent from SHA256SUMS')
    return f'{m.group("url")}{m.group("gap")}{m.group("ind")}sha256 "{sums[asset]}"'

pat = re.compile(
    r'(?P<url>url "#\{REL\}/(?P<asset>[A-Za-z0-9_.\-]+)")'
    r'(?P<gap>\s*\n\s*)(?P<ind>)sha256 "[0-9a-f]{64}"')
s, n = pat.subn(fix_pair, s)
assert n == 8, f'expected 8 homebrew url/sha256 pairs, rewrote {n}'
open(p, 'w').write(s); changed.append(p)

# --- Scoop bucket -------------------------------------------------------
p = 'bucket/mlrift.json'
d = json.load(open(p))
d['version'] = version
for key, asset in (('64bit', 'mlrc-windows-x86_64.zip'), ('arm64', 'mlrc-windows-arm64.zip')):
    d['architecture'][key]['url'] = (
        f'https://github.com/{os.environ.get("REPO", "Heniokhos-Systems/MLRift")}'
        f'/releases/download/v{version}/{asset}')
    d['architecture'][key]['hash'] = sums[asset]
open(p, 'w').write(json.dumps(d, indent=4) + '\n')
changed.append(p)

print('  rewrote: ' + ', '.join(changed))
PY

# --- .SRCINFO -----------------------------------------------------------
# Generated with the SAME emitter tests/run_tests.sh uses for aur_srcinfo_in_sync,
# so a manifest this script produces cannot fail that test. makepkg is not needed
# (and is not available on a non-Arch runner).
bash -c '
    set -e
    source packaging/aur/PKGBUILD
    printf "pkgbase = %s\n" "$pkgname"
    printf "\tpkgdesc = %s\n" "$pkgdesc"
    printf "\tpkgver = %s\n" "$pkgver"
    printf "\tpkgrel = %s\n" "$pkgrel"
    printf "\turl = %s\n" "$url"
    for a in "${arch[@]}"; do printf "\tarch = %s\n" "$a"; done
    for l in "${license[@]}"; do printf "\tlicense = %s\n" "$l"; done
    for pr in "${provides[@]}"; do printf "\tprovides = %s\n" "$pr"; done
    for o in "${options[@]}"; do printf "\toptions = %s\n" "$o"; done
    for x in "${source_x86_64[@]}"; do printf "\tsource_x86_64 = %s\n" "$x"; done
    for x in "${sha256sums_x86_64[@]}"; do printf "\tsha256sums_x86_64 = %s\n" "$x"; done
    for x in "${source_aarch64[@]}"; do printf "\tsource_aarch64 = %s\n" "$x"; done
    for x in "${sha256sums_aarch64[@]}"; do printf "\tsha256sums_aarch64 = %s\n" "$x"; done
    printf "\npkgname = %s\n" "$pkgname"
' > packaging/aur/.SRCINFO
echo "  regenerated packaging/aur/.SRCINFO"

# --- verify -------------------------------------------------------------
STALE=$(grep -rlE '[0-9]+\.[0-9]+\.[0-9]+' packaging/aur/PKGBUILD packaging/aur/.SRCINFO \
                  packaging/homebrew/mlrift.rb bucket/mlrift.json 2>/dev/null \
        | xargs grep -hoE '\b[0-9]+\.[0-9]+\.[0-9]+\b' | sort -u | grep -v "^$VERSION\$" || true)
if [ -n "$STALE" ]; then
    echo "  note: other version-shaped strings present (check they are intentional):"
    printf '    %s\n' $STALE
fi
echo "=== done: manifests now target v$VERSION ==="
