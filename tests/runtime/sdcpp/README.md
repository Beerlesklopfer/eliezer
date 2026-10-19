# Qwen Image 2.1 über stable-diffusion.cpp

Manueller GPU-Versuch; kein Bestandteil der gewöhnlichen netz- und GPU-freien Tests.
Quelle: `leejet/stable-diffusion.cpp`, Commit
`b13a48219d5cdf408edcb93da6cd69b612ed03d3` vom 18.10.2026.
Die Serverhilfe wurde nach dem Bau direkt geprüft.

## Bauen

Quellen einschließlich Submodule in ein lokales Arbeitsverzeichnis auschecken:

```bash
git clone --recurse-submodules https://github.com/leejet/stable-diffusion.cpp.git /tmp/eliezer-m1-sdcpp
git -C /tmp/eliezer-m1-sdcpp checkout b13a48219d5cdf408edcb93da6cd69b612ed03d3
git -C /tmp/eliezer-m1-sdcpp submodule update --init --recursive
cmake -S /tmp/eliezer-m1-sdcpp -B /tmp/eliezer-m1-sdcpp/build -G Ninja \
  -DCMAKE_C_COMPILER=/opt/rocm/llvm/bin/clang \
  -DCMAKE_CXX_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DSD_HIPBLAS=ON -DCMAKE_BUILD_TYPE=Release \
  -DGPU_TARGETS=gfx1100 -DAMDGPU_TARGETS=gfx1100 \
  -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
  -DSD_SERVER_BUILD_FRONTEND=OFF -DSD_WEBP=OFF -DSD_WEBM=OFF
cmake --build /tmp/eliezer-m1-sdcpp/build --target sd-server -j 4
```

Für den lokalen Bau werden zusätzlich die ROCm- und OpenMP-Bibliotheken über
`LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/lib/llvm/lib` verfügbar gemacht.
Die Pfade sind an die Installation anzupassen. Es werden keine Systembibliotheken verändert.

## Probe

```bash
LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/lib/llvm/lib \
python3 tests/runtime/sdcpp/probe.py \
  --server /tmp/eliezer-m1-sdcpp/build/bin/sd-server \
  --diffusion /pfad/qwen-image-2.1-UC-Q4_K_M.gguf \
  --encoder /pfad/qwen3vl_8b_int8_convrot.safetensors \
  --vae /pfad/qwen_image_2.1_vae_bf16.safetensors \
  --chat-probe /pfad/eliezer-m1-probe \
  --chat-model /pfad/Mistral-7B-Instruct-v0.3-Q4_K_M.gguf
```

Das Skript führt Chat → Bild → Chat nacheinander aus, prüft HTTP-Modellliste und
Base64-PNG-Antwort, misst den gesamten VRAM mit einem 50-ms-Abtastintervall und
beendet den gestarteten Server. Ergebnisse, Testbild und synthetische Testlogs
landen in `target/` und werden nicht versioniert. Geprüfte Ergebnisse werden
separat in `results/` festgehalten. Es werden keine Modelle heruntergeladen.

## Geprüfte Schnittstelle

- Start: `--diffusion-model`, `--llm`, `--vae`, `--listen-ip`, `--listen-port`.
- Speichersparender Betrieb: `--offload-to-cpu`.
- Bereitschaft: `GET /v1/models`, erwartet HTTP 200 und eine Modellliste.
- Generierung: `POST /v1/images/generations`, JSON mit `prompt`, `n`, `size`,
  `output_format`; Antwort enthält `created` und `data[].b64_json`.
- Probe: PNG, 512 × 512, 4 Euler-Schritte, CFG 6. Das ist ein Funktionstest,
  keine Qualitätsbewertung und kein Nachweis für andere Auflösungen oder Schrittzahlen.

Qwen Image 2.1 verwendet Qwen3-VL-8B und den eigenen 2.1-VAE. Der frühere
Qwen2.5-VL-Encoder ist für dieses Modell keine austauschbare Alternative.

## Ergebnisse vom 19.10.2026

Vollständige Messwerte: [results/2026-10-19.json](results/2026-10-19.json).

- ROCm-Bau und direkter HTTP-Aufruf erfolgreich.
- Chat → Bild → Chat mit lokalen Dateien erfolgreich; alle gestarteten Prozesse beendet.
- Server antwortbereit nach 0,202 s. Die Modellliste belegt keine geladenen Gewichte:
  das Bildmodell lädt erst mit der ersten Bildanfrage.
- Erste Bildanfrage einschließlich Laden: 8,087 s, PNG mit 512 × 512 Pixeln,
  251 054 Byte; vier Euler-Schritte, CFG 6.
- Spitzenbelegung der gesamten Karte: 8 374 263 808 Byte bei Ausgangsbelegung
  1 194 037 248 Byte (Zuwachs rund 7,18 GB).
- Nach Bildserver-Ende wieder exakt 1 194 037 248 Byte in dieser Messung.
- Beide Chat-Prozessläufe erfolgreich. Nach dem ersten Prozessende Ausgangsbelegung
  wieder erreicht; nach dem zweiten rund 91 MB darüber. Die Messung erfasst auch den Desktop.
- Dieser konkrete Bildversuch ist kein 8-GB-Nachweis: Die Gesamtspitze überschreitet
  8 GB. Für die kleinste Zielklasse sind andere Modelle oder weitere Speichereinstellungen nötig.
