#!/usr/bin/env bash
# Checks local prerequisites without downloads, model loading or secret access.
set -euo pipefail

printf 'Rust tools\n'
for tool in rustc cargo; do
    if [[ -x "$HOME/.cargo/bin/$tool" ]]; then
        "$HOME/.cargo/bin/$tool" --version
    elif command -v "$tool" >/dev/null 2>&1; then
        "$tool" --version
    else
        printf '%s: not found\n' "$tool"
    fi
done

printf '\nGPU devices\n'
for device in /dev/dri /dev/kfd; do
    if [[ -e "$device" ]]; then
        ls -ld "$device"
    else
        printf '%s: missing\n' "$device"
    fi
done

printf '\nGPU data from sysfs (not a runtime measurement)\n'
for card in /sys/class/drm/card[0-9]*; do
    [[ "${card##*/}" =~ ^card[0-9]+$ ]] || continue
    [[ -d "$card/device" ]] || continue
    for field in vendor device mem_info_vram_total mem_info_vram_used; do
        if [[ -r "$card/device/$field" ]]; then
            printf '%s %s: ' "${card##*/}" "$field"
            cat "$card/device/$field"
        fi
    done
done

printf '\nTools in PATH\n'
for tool in hipcc rocminfo vulkaninfo cmake cc glslc hf sd-server podman docker; do
    if command -v "$tool" >/dev/null 2>&1; then
        command -v "$tool"
    else
        printf '%s: not found\n' "$tool"
    fi
done

printf '\nRuntimes in the LM Studio user installation\n'
if [[ -d "$HOME/.lmstudio/extensions/backends" ]]; then
    find "$HOME/.lmstudio/extensions/backends" -maxdepth 2 -type f -name llama-server -print
fi
