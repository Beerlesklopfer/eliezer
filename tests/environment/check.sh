#!/usr/bin/env bash
# Prüft lokale Voraussetzungen ohne Downloads, Modellladen oder Geheimniszugriff.
set -euo pipefail

printf 'Rust-Werkzeuge\n'
for tool in rustc cargo; do
    if [[ -x "$HOME/.cargo/bin/$tool" ]]; then
        "$HOME/.cargo/bin/$tool" --version
    elif command -v "$tool" >/dev/null 2>&1; then
        "$tool" --version
    else
        printf '%s: nicht gefunden\n' "$tool"
    fi
done

printf '\nGPU-Geräte\n'
for device in /dev/dri /dev/kfd; do
    if [[ -e "$device" ]]; then
        ls -ld "$device"
    else
        printf '%s: fehlt\n' "$device"
    fi
done

printf '\nGPU-Angaben aus sysfs (keine Laufzeitmessung)\n'
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

printf '\nWerkzeuge im PATH\n'
for tool in hipcc rocminfo vulkaninfo cmake cc glslc hf sd-server podman docker; do
    if command -v "$tool" >/dev/null 2>&1; then
        command -v "$tool"
    else
        printf '%s: nicht gefunden\n' "$tool"
    fi
done

printf '\nLaufzeiten in der LM-Studio-Benutzerinstallation\n'
if [[ -d "$HOME/.lmstudio/extensions/backends" ]]; then
    find "$HOME/.lmstudio/extensions/backends" -maxdepth 2 -type f -name llama-server -print
fi
