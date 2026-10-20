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

# Keep the previous deliverable intact until bundling succeeds.
stage=$(mktemp -d target/deb-stage.XXXXXX)
trap 'rm -rf "$stage"' EXIT
cargo deb -p eliezer --locked --variant "$variant" --output "$stage/" "$@"
deb=$(find "$stage" -maxdepth 1 -name 'eliezer_*.deb' -print -quit)
[ -n "$deb" ] || { echo "No package produced in the staging directory" >&2; exit 1; }
if [ "$variant" = rocm ]; then
    # The image runtime (sd-server) and the music runtime (audiocpp_server) are part of the ROCm package;
    # SKIP_SDCPP=1 and SKIP_AUDIOCPP=1 leave them out.
    set --
    if [ "${SKIP_SDCPP:-0}" != 1 ]; then
        set -- --sd-server "$(scripts/build-sdcpp.sh)"
    fi
    if [ "${SKIP_AUDIOCPP:-0}" != 1 ]; then
        set -- "$@" --audiocpp-server "$(scripts/build-audiocpp.sh)"
    fi
    python3 scripts/bundle-rocm.py "$deb" "$@"
fi
mkdir -p target/debian
final="target/debian/$(basename "$deb")"
mv "$deb" "$final"
echo "Package: $final"
dpkg-deb --info "$final"
dpkg-deb --contents "$final"
