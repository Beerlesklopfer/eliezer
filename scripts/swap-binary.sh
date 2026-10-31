#!/bin/sh
# Builds the Eliezer program the way the ROCm package contains it and, on request, swaps it into the installed
# system. For a change of the program alone this replaces a rebuild of the package (about 1 GB with the kernel files of one architecture). A change of the
# libraries, of sd-server, of the unit file, the default configuration or the maintainer scripts needs the package
# (scripts/build-deb.sh).
#
#   scripts/swap-binary.sh             build target/swap/eliezer (not installed)
#   scripts/swap-binary.sh --install   build, then install it as /usr/bin/eliezer and restart the service (sudo)
#
# The run path of the packaged program points at the bundled libraries (`$ORIGIN/../lib/<triplet>/eliezer`); it is
# set the same way here, so the swapped program finds them without LD_LIBRARY_PATH. The package manager still
# knows the old file: `dpkg --verify eliezer` reports it as changed until the next package install.
set -eu
cd "$(dirname "$0")/.."

triplet=$(dpkg-architecture -qDEB_HOST_MULTIARCH 2>/dev/null || echo x86_64-linux-gnu)
command -v patchelf >/dev/null || { echo "patchelf is missing (apt install patchelf)" >&2; exit 1; }
[ -d "/usr/lib/$triplet/eliezer" ] || echo "Note: /usr/lib/$triplet/eliezer does not exist; the package has to be installed first." >&2

# Flags of the ROCm build (tests/runtime/llamacpp/README.md); the architecture comes from the installed card.
export CMAKE_POSITION_INDEPENDENT_CODE=ON CMAKE_HIP_FLAGS=-fPIC
if [ -z "${AMDGPU_TARGETS:-}" ] && command -v rocminfo >/dev/null; then
    AMDGPU_TARGETS=$(rocminfo 2>/dev/null | grep -o -m1 'gfx[0-9a-f]*' || true)
    [ -n "$AMDGPU_TARGETS" ] && export AMDGPU_TARGETS
fi
export LD_LIBRARY_PATH="${ROCM_PATH:-/opt/rocm}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

cargo build --release --locked -p eliezer --features rocm
mkdir -p target/swap
cp target/release/eliezer target/swap/eliezer
strip --strip-unneeded target/swap/eliezer 2>/dev/null || true
patchelf --set-rpath "\$ORIGIN/../lib/$triplet/eliezer" target/swap/eliezer
echo "Built target/swap/eliezer ($(du -h target/swap/eliezer | cut -f1)), version: $(LD_LIBRARY_PATH="/usr/lib/$triplet/eliezer" target/swap/eliezer --version)"

if [ "${1:-}" = --install ]; then
    sudo install -m 0755 -o root -g root target/swap/eliezer /usr/bin/eliezer
    sudo /usr/bin/eliezer --check-config /etc/eliezer/config.toml
    sudo systemctl restart eliezer
    echo "Installed and restarted."
fi
