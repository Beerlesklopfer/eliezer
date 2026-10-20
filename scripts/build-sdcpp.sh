#!/bin/sh
# Builds `sd-server` of stable-diffusion.cpp at the commit that was checked (docs/Runtimes.md) with HIP for the
# installed card and prints its path. The result is cached in target/sdcpp; nothing is installed.
#
#   scripts/build-sdcpp.sh [--rebuild]
#
# Needs git, cmake, ninja and ROCm (ROCM_PATH, default /opt/rocm). The sources are cloned from GitHub once. The
# program gets the run path of the package's private multiarch directory, so it starts without LD_LIBRARY_PATH
# once the package has bundled the libraries there (scripts/build-deb.sh does that).
set -eu
cd "$(dirname "$0")/.."
commit=b13a48219d5cdf408edcb93da6cd69b612ed03d3
rocm="${ROCM_PATH:-/opt/rocm}"
triplet=$(dpkg-architecture -qDEB_HOST_MULTIARCH 2>/dev/null || echo x86_64-linux-gnu)
dir=target/sdcpp
binary=$dir/build/bin/sd-server
stamp=$dir/built-$commit-rpath

if [ "${1:-}" = --rebuild ]; then rm -rf "$dir/build" "$stamp"; fi
if [ -x "$binary" ] && [ -e "$stamp" ]; then echo "$binary"; exit 0; fi

command -v cmake >/dev/null && command -v ninja >/dev/null || { echo "cmake and ninja are needed" >&2; exit 1; }
[ -x "$rocm/llvm/bin/clang++" ] || { echo "ROCm not found at $rocm (set ROCM_PATH)" >&2; exit 1; }
mkdir -p "$dir"
if [ ! -d "$dir/src/.git" ] && [ ! -e "$dir/src/.git" ]; then
    git clone -q --recurse-submodules https://github.com/leejet/stable-diffusion.cpp.git "$dir/src"
fi
git -C "$dir/src" checkout -q "$commit"
git -C "$dir/src" submodule update -q --init --recursive

targets=${AMDGPU_TARGETS:-}
if [ -z "$targets" ] && command -v rocminfo >/dev/null; then
    targets=$(rocminfo 2>/dev/null | grep -o -m1 'gfx[0-9a-f]*' || true)
fi
[ -n "$targets" ] || targets=gfx1100
echo "Building sd-server for $targets" >&2
rm -f "$dir"/built-*
cmake -S "$dir/src" -B "$dir/build" -G Ninja \
    -DCMAKE_C_COMPILER="$rocm/llvm/bin/clang" -DCMAKE_CXX_COMPILER="$rocm/llvm/bin/clang++" \
    -DSD_HIPBLAS=ON -DCMAKE_BUILD_TYPE=Release \
    -DGPU_TARGETS="$targets" -DAMDGPU_TARGETS="$targets" \
    -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON -DCMAKE_INSTALL_RPATH="/usr/lib/$triplet/eliezer" \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    -DSD_SERVER_BUILD_FRONTEND=OFF -DSD_WEBP=OFF -DSD_WEBM=OFF >&2
cmake --build "$dir/build" --target sd-server -j "${JOBS:-6}" >&2
touch "$stamp"
echo "$binary"
