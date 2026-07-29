#!/bin/bash
# Build .deb packages for MLRift
# Usage: ./build-deb.sh [version]
# Produces: mlrift_VERSION_amd64.deb and mlrift_VERSION_arm64.deb
set -e

REPO="Heniokhos-Systems/MLRift"

if [ -z "${1:-}" ]; then
    echo "Fetching latest version from GitHub..."
    VERSION=$(curl -sSL "https://api.github.com/repos/$REPO/releases/latest" | grep '"tag_name":' | sed -E 's/.*"v?([^"]+)".*/\1/')
    BASE="https://github.com/$REPO/releases/latest/download"
    echo "Latest version is $VERSION"
else
    VERSION="$1"
    BASE="https://github.com/$REPO/releases/download/v$VERSION"
fi

RAW="https://raw.githubusercontent.com/$REPO/main"

build_deb() {
    local arch="$1"      # amd64 or arm64
    local bin_name="$2"  # mlrc-linux-x86_64 or mlrc-linux-arm64
    local mlr_name="$3"  # mlr-linux-x86_64 or mlr-linux-arm64

    local PKG="mlrift_${VERSION}_${arch}"
    rm -rf "$PKG"

    # Create directory structure
    mkdir -p "$PKG/DEBIAN"
    mkdir -p "$PKG/usr/bin"
    mkdir -p "$PKG/usr/share/mlrift/std"
    mkdir -p "$PKG/usr/share/doc/mlrift"

    # Control file
    cat > "$PKG/DEBIAN/control" <<EOF
Package: mlrift
Version: $VERSION
Section: devel
Priority: optional
Architecture: $arch
Maintainer: Pantelis Christou <pantelisworks@gmail.com>
Homepage: https://github.com/Heniokhos-Systems/MLRift
Description: Self-hosted systems language and compiler for machine-learning workloads
 MLRift is a self-hosted systems language compiler forked from KernRift, with
 an SSA IR backend that emits native machine code directly -- no LLVM, no C,
 no external assembler. It produces native executables for x86_64 and
 AArch64 across Linux, Windows, macOS, and Android, and extends the IR with
 ML-specific primitives (tensors, event streams, sparse CSR ops, plasticity
 rules). It includes a native AMDGCN GPU backend that talks to /dev/kfd
 directly, with zero ROCm DSO dependencies on Linux.
 .
 The compiler is a single static binary with zero dependencies.
EOF

    # Download mlrc binary
    echo "  Downloading $bin_name..."
    curl -sSLf -o "$PKG/usr/bin/mlrc" "$BASE/$bin_name"
    chmod 755 "$PKG/usr/bin/mlrc"

    # Download mlr runner
    echo "  Downloading $mlr_name..."
    curl -sSLf -o "$PKG/usr/bin/mlr" "$BASE/$mlr_name"
    chmod 755 "$PKG/usr/bin/mlr"

    # Download stdlib (curated general-purpose subset -- see
    # packaging/aur/PKGBUILD for why this isn't all of std/)
    for mod in alloc string io math math_float fmt mem memfast vec map color fb fixedpoint font widget time log net sha256; do
        echo "  Downloading std/$mod.mlr..."
        curl -sSLf -o "$PKG/usr/share/mlrift/std/$mod.mlr" "$RAW/std/$mod.mlr"
    done

    # Copyright files
    echo "  Downloading LICENSE and NOTICE..."
    curl -sSLf -o "$PKG/usr/share/doc/mlrift/LICENSE" "$BASE/LICENSE"
    curl -sSLf -o "$PKG/usr/share/doc/mlrift/NOTICE" "$BASE/NOTICE"

    # Build .deb
    dpkg-deb --build --root-owner-group "$PKG"
    echo "  Built: ${PKG}.deb"
    rm -rf "$PKG"
}

echo "=== Building MLRift $VERSION .deb packages ==="
# KNOWN BUG: mlrc's Linux stdlib search paths are still hardcoded (in
# src/main.mlr) to /usr/local/share/kernrift/, /usr/share/kernrift/ and
# $HOME/.local/share/kernrift/ -- not mlrift. Until that upstream rename
# lands, `import "std/..."` will not resolve against the
# /usr/share/mlrift/std installed below. See packaging/aur/PKGBUILD for
# the full note.
build_deb "amd64" "mlrc-linux-x86_64" "mlr-linux-x86_64"
build_deb "arm64" "mlrc-linux-arm64" "mlr-linux-arm64"
echo "=== Done ==="
