#!/bin/sh
# Builds `whisper-server` of whisper.cpp at the tag that was checked (docs/Runtimes.md) with HIP for the installed card
# and prints its path. The result is cached in target/whispercpp; nothing is installed.
#
#   scripts/build-whispercpp.sh [--rebuild]
#
# Needs git, cmake, ninja and ROCm (ROCM_PATH, default /opt/rocm). The sources are fetched from GitHub once.
set -eu
cd "$(dirname "$0")/.."
tag=v1.9.5
commit=d2a764c648779feb36d44363f212a29b00aad1af
rocm="${ROCM_PATH:-/opt/rocm}"
dir=target/whispercpp
src=$dir/src
binary=$src/build/bin/whisper-server
stamp=$dir/built-$commit

if [ "${1:-}" = --rebuild ]; then rm -rf "$src/build" "$stamp"; fi
if [ -x "$binary" ] && [ -e "$stamp" ]; then echo "$binary"; exit 0; fi

command -v cmake >/dev/null && command -v ninja >/dev/null || { echo "cmake and ninja are needed" >&2; exit 1; }
[ -x "$rocm/llvm/bin/clang++" ] || { echo "ROCm not found at $rocm (set ROCM_PATH)" >&2; exit 1; }
mkdir -p "$dir"
if [ ! -d "$src/.git" ]; then
    git init -q "$src"
    git -C "$src" remote add origin https://github.com/ggml-org/whisper.cpp.git
fi
if [ "$(git -C "$src" rev-parse -q --verify HEAD 2>/dev/null || true)" != "$commit" ]; then
    git -C "$src" fetch -q --depth 1 origin "$commit"
    git -C "$src" checkout -q --detach FETCH_HEAD
fi

targets=${AMDGPU_TARGETS:-}
if [ -z "$targets" ] && command -v rocminfo >/dev/null; then
    targets=$(rocminfo 2>/dev/null | grep -o -m1 'gfx[0-9a-f]*' || true)
fi
[ -n "$targets" ] || targets=gfx1100
echo "Building whisper-server $tag for $targets" >&2
rm -f "$dir"/built-*
(cd "$src" && \
    HIPCXX="$rocm/llvm/bin/clang" HIP_PATH="$rocm" cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
        -DGGML_HIP=ON -DAMDGPU_TARGETS="$targets" -DWHISPER_BUILD_TESTS=OFF -DWHISPER_BUILD_EXAMPLES=ON >&2 && \
    cmake --build build --target whisper-server >&2)
[ -x "$binary" ] || { echo "The build did not produce $binary" >&2; exit 1; }
touch "$stamp"
echo "$binary"
