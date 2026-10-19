# Vergleich mit llama-swap

Manueller GPU-Vergleich mit denselben Modelldateien und derselben Bildanfrage
wie im [direkten Versuch](../sdcpp/README.md).
Geprüfter Quellstand: `a2796fb0306215c6883454a8707dfb22d25c3068` vom 14.10.2026.

## Bau und Aufruf

Der Quellstand verlangt Go 1.27.1. Der isolierte Bau nutzt automatische
Go-Toolchain-Auswahl; Cache, Toolchain und Module bleiben unter `/tmp`:

```bash
git clone https://github.com/mostlygeek/llama-swap.git /tmp/eliezer-m1-llama-swap
git -C /tmp/eliezer-m1-llama-swap checkout a2796fb0306215c6883454a8707dfb22d25c3068
cd /tmp/eliezer-m1-llama-swap
GOPATH=/tmp/eliezer-m1-go GOMODCACHE=/tmp/eliezer-m1-go/pkg/mod \
  GOCACHE=/tmp/eliezer-m1-go/cache go build -o llama-swap .
```

Aufruf aus dem Eliezer-Repository:

```bash
LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/lib/llvm/lib \
python3 tests/runtime/llama-swap/probe.py \
  --swap /tmp/eliezer-m1-llama-swap/llama-swap \
  --chat-server /pfad/llama-server \
  --chat-model /pfad/Mistral-7B-Instruct-v0.3-Q4_K_M.gguf \
  --image-server /tmp/eliezer-m1-sdcpp/build/bin/sd-server \
  --diffusion /pfad/qwen-image-2.1-UC-Q4_K_M.gguf \
  --encoder /pfad/qwen3vl_8b_int8_convrot.safetensors \
  --vae /pfad/qwen_image_2.1_vae_bf16.safetensors
```

Das Skript erzeugt eine Konfiguration mit zwei Modellen, validiert sie mit
`llama-swap -validate` und führt Chat → Bild → Chat über den Proxy aus.
Die Standard-Gruppenplanung lässt nur ein Modell gleichzeitig laufen.
Alle Listener sind auf `127.0.0.1` begrenzt, Ports 24606 bis 24608.
Server und Kindprozesse werden am Ende über ihre eigene Prozessgruppe beendet.

## Vergleichsgrenze

Für Chat wird der vorhandene Vulkan-`llama-server` aus LM Studio verwendet,
für Bild derselbe HIP-`sd-server` wie im direkten Versuch. Die Modelldateien und
Anfragen sind gleich; die Chat-Laufzeiten unterscheiden sich (Vulkan-Prozess
gegen eingebettetes ROCm-Binding). Die Messung dient dem Funktions- und
Wechselvergleich, keiner isolierten Geschwindigkeitsbewertung des Verwalters.

Messwerte, synthetische Logs, Konfiguration und Testbild entstehen unter `target/`.
Geprüfte Ergebniszusammenfassungen liegen unter `results/`.

## Ergebnisse vom 19.10.2026

Messwerte: [results/2026-10-19.json](results/2026-10-19.json).

- Quellstand erfolgreich mit automatisch bezogener Go-Toolchain gebaut.
- Konfiguration mit zwei Modellen erfolgreich validiert.
- Chat → Bild → Chat über den Proxy erfolgreich, jeweils einschließlich Laden/Wechsel:
  1,537 s → 8,714 s → 2,090 s.
- Beide Chatantworten melden 16 Prompt- und 32 Ausgabetoken.
- Bildantwort: gültiges PNG, 251 054 Byte, identischer synthetischer Prompt
  und identische Einstellungen wie im direkten Versuch.
- VRAM nach Beenden des Vergleichs: 1 173 655 552 Byte bei Ausgangswert
  1 221 361 664 Byte; kein verbleibender Modellprozess festgestellt.
- Entscheidung: Der fertige Verwalter beherrscht den geprüften Wechsel.
  Eliezer bleibt der Eigenbau, weil die gewählte Architektur die Sprachlaufzeit
  direkt einbettet und die LM-Studio-Verwaltung sowie das eigene VRAM-Buch bereitstellt.
  Für den späteren Betrieb entsteht dadurch keine Abhängigkeit auf llama-swap.
