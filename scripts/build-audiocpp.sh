#!/bin/sh
# Builds `audiocpp_server` of audio.cpp at the commit that was checked (docs/Runtimes.md) with HIP for the installed
# card and prints its path. The model families compiled into the program are a setting: AUDIOCPP_MODELS, a comma separated
# list of the family names of audio.cpp (the syntax of `scripts/build_linux.sh --model-set custom --models`; the names
# are the targets and aliases of `audiocpp_add_model` in its CMakeLists.txt). The default is the music model only;
# for music and speech with voice cloning: AUDIOCPP_MODELS=minimax_music3,chatterbox. The result is cached in
# target/audiocpp, per commit and model set; nothing is installed.
#
#   [AUDIOCPP_MODELS=minimax_music3,chatterbox] scripts/build-audiocpp.sh [--rebuild]
#
# Needs git, cmake, ninja and ROCm (ROCM_PATH, default /opt/rocm). The sources are fetched from GitHub once.
# scripts/build-deb.sh bundles the program into the package and gives it the run path of the private multiarch directory.
set -eu
cd "$(dirname "$0")/.."
commit=18bba55495c8991bbbfe7625234d4d8dbc6e82f4
rocm="${ROCM_PATH:-/opt/rocm}"
models="${AUDIOCPP_MODELS:-minimax_music3}"
case "$models" in
    ''|*[!a-z0-9_,]*|,*|*,|*,,*) echo "AUDIOCPP_MODELS is a comma separated list of family names (a-z, 0-9, _)" >&2; exit 1 ;;
esac
dir=target/audiocpp
src=$dir/src
binary=$src/build/linux-hip-release/bin/audiocpp_server
stamp=$dir/built-$commit-$(printf %s "$models" | tr , +)

if [ "${1:-}" = --rebuild ]; then rm -rf "$src/build" "$stamp"; fi
if [ -x "$binary" ] && [ -e "$stamp" ]; then echo "$binary"; exit 0; fi

command -v cmake >/dev/null && command -v ninja >/dev/null || { echo "cmake and ninja are needed" >&2; exit 1; }
[ -x "$rocm/llvm/bin/clang++" ] || { echo "ROCm not found at $rocm (set ROCM_PATH)" >&2; exit 1; }
mkdir -p "$dir"
if [ ! -d "$src/.git" ]; then
    git init -q "$src"
    git -C "$src" remote add origin https://github.com/0xShug0/audio.cpp.git
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
echo "Building audiocpp_server for $targets with the models $models" >&2
rm -f "$dir"/built-*
(cd "$src" && ROCM_PATH="$rocm" scripts/build_linux.sh --backend hip --gpu-targets "$targets" \
    --model-set custom --models "$models" --target audiocpp_server >&2)
[ -x "$binary" ] || { echo "The build did not produce $binary" >&2; exit 1; }
touch "$stamp"
echo "$binary"
