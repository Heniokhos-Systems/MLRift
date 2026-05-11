#!/usr/bin/env bash
# mlr — MLRift fat binary launcher
#
# Termux on Android 14+ denies raw execve of files in /data/data/<app>/.
# The runner binary (mlr-bin) does a raw syscall and gets EACCES when it
# tries to launch the extracted slice. To work around this, mlr-bin still
# extracts and chmod's the slice to ./mlr-exec, then exits 120. We catch
# that here and re-exec ./mlr-exec via the user's shell, where the Termux
# libc LD_PRELOAD wrapper makes exec succeed.
#
# Other platforms (Linux/macOS) raise no such restriction; the runner
# exec's directly and we never reach the post-call return.

set -u

MLRBIN="${MLRBIN:-${0%/*}/mlr-bin}"
[ -x "$MLRBIN" ] || MLRBIN=$(command -v mlr-bin)
if [ -z "${MLRBIN:-}" ] || [ ! -x "$MLRBIN" ]; then
    echo "mlr: mlr-bin not found alongside this wrapper" >&2
    exit 1
fi

"$MLRBIN" "$@"
status=$?

# Termux fallback: 120 = "extracted ./mlr-exec, exec was denied"
if [ "$status" -eq 120 ] && [ -x ./mlr-exec ]; then
    shift  # drop the .mlrbo path; remaining args go to the slice
    exec ./mlr-exec "$@"
fi

exit $status
