#!/bin/sh
# MLRift Installer — download and install mlrc + mlr from GitHub releases
# Usage: curl -sSf https://raw.githubusercontent.com/Pantelis23/MLRift/main/install.sh | sh
#
# Alternative installation methods:
#   PowerShell (Windows):    irm https://raw.githubusercontent.com/Pantelis23/MLRift/main/install.ps1 | iex
#
set -e

REPO="Pantelis23/MLRift"

# Detect platform
ARCH=$(uname -m)
IS_ANDROID=0
IS_TERMUX=0
if [ -f "/system/bin/linker64" ]; then
    IS_ANDROID=1
    if [ -d "/data/data/com.termux/files" ]; then
        IS_TERMUX=1
    fi
fi

case "$ARCH" in
    x86_64|amd64) ARCH_NAME="x86_64" ;;
    aarch64|arm64) ARCH_NAME="arm64" ;;
    *) echo "error: unsupported architecture: $ARCH"; exit 1 ;;
esac

OS=$(uname -s)
case "$OS" in
    Linux)
        if [ "$IS_ANDROID" = "1" ]; then
            OS_NAME="android"
        else
            OS_NAME="linux"
        fi
        ;;
    Darwin) OS_NAME="macos" ;;
    *)      echo "error: unsupported OS: $OS"; exit 1 ;;
esac

# Set install directory based on environment
if [ "$IS_TERMUX" = "1" ]; then
    INSTALL_DIR="${INSTALL_DIR:-$HOME/.local/bin}"
    STD_DIR="$HOME/.local/share/mlrift/std"
elif [ "$IS_ANDROID" = "1" ]; then
    INSTALL_DIR="${INSTALL_DIR:-/data/local/tmp/mlrift}"
    STD_DIR="/data/local/tmp/mlrift/std"
else
    INSTALL_DIR="${INSTALL_DIR:-$HOME/.local/bin}"
    STD_DIR="$HOME/.local/share/mlrift/std"
fi

echo "=== MLRift Installer ==="
if [ "$IS_TERMUX" = "1" ]; then
    echo "Platform: Android (Termux) ARM64"
elif [ "$IS_ANDROID" = "1" ]; then
    echo "Platform: Android (adb) ARM64"
else
    echo "Platform: $OS_NAME $ARCH_NAME"
fi
echo "Install to: $INSTALL_DIR"
echo ""

mkdir -p "$INSTALL_DIR"

BASE="https://github.com/$REPO/releases/latest/download"

# Pick the right release asset names per platform.
if [ "$IS_ANDROID" = "1" ]; then
    MLRC_ASSET="mlrc-android-$ARCH_NAME"
    MLR_ASSET="mlr-android-$ARCH_NAME"
elif [ "$OS_NAME" = "macos" ]; then
    MLRC_ASSET="mlrc-macos-$ARCH_NAME"
    MLR_ASSET="mlr-macos-$ARCH_NAME"
else
    MLRC_ASSET="mlrc-linux-$ARCH_NAME"
    MLR_ASSET="mlr-linux-$ARCH_NAME"
fi

echo "Downloading $MLRC_ASSET..."
curl -sL -o "$INSTALL_DIR/mlrc" "$BASE/$MLRC_ASSET"
chmod +x "$INSTALL_DIR/mlrc"

echo "Downloading $MLR_ASSET..."
if [ "$IS_TERMUX" = "1" ]; then
    # Termux on Android 14+ denies raw execve of files in
    # /data/data/com.termux/. The runner extracts the slice and exits 120;
    # the canonical packaging/mlr.sh shell wrapper catches that and re-execs
    # ./mlr-exec from the user's shell context (where Termux's libc
    # LD_PRELOAD makes execve succeed). Install both mlr-bin and mlr.
    curl -sL -o "$INSTALL_DIR/mlr-bin" "$BASE/$MLR_ASSET"
    chmod +x "$INSTALL_DIR/mlr-bin"
    curl -sL -o "$INSTALL_DIR/mlr" \
        "https://raw.githubusercontent.com/$REPO/main/packaging/mlr.sh"
    chmod +x "$INSTALL_DIR/mlr"
else
    curl -sL -o "$INSTALL_DIR/mlr" "$BASE/$MLR_ASSET"
    chmod +x "$INSTALL_DIR/mlr"
fi

# Download standard library
echo "Installing standard library..."
mkdir -p "$STD_DIR"
for mod in string io math fmt mem vec map color fb fixedpoint font memfast widget time log net; do
    curl -sL -o "$STD_DIR/$mod.mlr" \
        "https://raw.githubusercontent.com/$REPO/main/std/$mod.mlr"
done
echo "Standard library: $STD_DIR"

echo ""

# Verify
if "$INSTALL_DIR/mlrc" --version 2>/dev/null; then
    echo ""
fi

# Check PATH
case ":$PATH:" in
    *":$INSTALL_DIR:"*)
        echo "mlrc is in your PATH."
        ;;
    *)
        if [ "$IS_TERMUX" = "1" ]; then
            echo "Add to PATH:"
            echo "  echo 'export PATH=\"$INSTALL_DIR:\$PATH\"' >> ~/.bashrc"
            echo "  source ~/.bashrc"
        elif [ "$IS_ANDROID" = "1" ]; then
            echo "Run directly:"
            echo "  $INSTALL_DIR/mlrc --version"
        else
            echo "Add to PATH:  export PATH=\"$INSTALL_DIR:\$PATH\""
            echo "Or add that line to ~/.bashrc"
        fi
        ;;
esac

echo ""
echo "Usage:"
echo "  mlrc hello.mlr -o hello.mlrbo  # compile (fat binary)"
echo "  mlr hello.mlrbo                # run on any platform"
if [ "$IS_ANDROID" = "1" ]; then
    echo "  mlrc --emit=android hello.mlr -o hello   # native Android ARM64"
else
    echo "  mlrc --arch=$ARCH_NAME hello.mlr         # native binary"
fi
echo "  mlrc check module.mlr          # safety analysis"
echo ""
echo "=== Done ==="
