#!/bin/sh
# Builds the Debian package with cargo-deb (install once: cargo install cargo-deb).
#
#   scripts/build-deb.sh [rocm|vulkan|cuda|cpu]
#
# The variant selects the model runtime that is compiled in. Without an argument it is `rocm` when
# ROCm is installed (ROCM_PATH, default /opt/rocm) and `cpu` otherwise. A package without a variant
# cannot load models, so there is no such build.
# The unit file in packaging/ is written for /usr/local/bin; the package installs to /usr/bin.
set -eu
cd "$(dirname "$0")/.."

command -v cargo-deb >/dev/null || { echo "cargo-deb is missing: cargo install cargo-deb" >&2; exit 1; }

rocm="${ROCM_PATH:-/opt/rocm}"
if [ $# -gt 0 ] && [ "${1#-}" = "$1" ]; then
    variant=$1
    shift
elif [ -e "$rocm/lib/libamdhip64.so" ]; then
    variant=rocm
else
    variant=cpu
fi
case $variant in
    rocm | vulkan | cuda | cpu) ;;
    *) echo "unknown variant '$variant' (rocm, vulkan, cuda or cpu)" >&2; exit 2 ;;
esac
echo "Building variant: $variant"

if [ "$variant" = rocm ]; then
    # Flags from the M1 experiment (tests/runtime/llamacpp/README.md): the HIP objects must be
    # position independent. The GPU architecture is taken from the installed card.
    export CMAKE_POSITION_INDEPENDENT_CODE=ON CMAKE_HIP_FLAGS=-fPIC
    if [ -z "${AMDGPU_TARGETS:-}" ] && command -v rocminfo >/dev/null; then
        AMDGPU_TARGETS=$(rocminfo 2>/dev/null | grep -o -m1 'gfx[0-9a-f]*' || true)
        [ -n "$AMDGPU_TARGETS" ] && export AMDGPU_TARGETS && echo "GPU architecture: $AMDGPU_TARGETS"
    fi
fi

mkdir -p target/deb-assets
sed 's|/usr/local/bin/eliezer|/usr/bin/eliezer|' packaging/eliezer.service > target/deb-assets/eliezer.service

cargo deb -p eliezer --locked --variant "$variant" "$@"

deb=$(ls -t target/debian/eliezer_*.deb | head -n 1)
echo "Package: $deb"
dpkg-deb --info "$deb"
dpkg-deb --contents "$deb"
