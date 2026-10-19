# M1 – Machbarkeitsentscheidung

Abgeschlossen am 19.10.2026, Meilensteinfassung 0.1.0.
Geprüfte Hardware: Radeon RX 7900 XTX (`gfx1100`, rund 24 GiB VRAM), Linux,
ROCm/HIP 7.2. Der Mindestanspruch von 8 GB ist keine Aussage, dass jedes Modell
oder die hier geprüfte Bildkonfiguration auf 8 GB passt.

## Ergebnisse

| Prüffrage | Ergebnis und Nachweis |
|---|---|
| Aktueller mistral.rs-Stand | Release und Workspace v0.9.5, Tag und Commit `465191d0b0a7e2432d327b0de06204ab3f94d3a2`; Rust mindestens 1.94 |
| mistral.rs auf AMD | SDK/Core bieten am geprüften Stand kein ROCm-/Vulkan-Backend. Kein AMD-GPU-Modelllauf über diese Backends verfügbar; [Quellenprüfung](../runtime/mistralrs/README.md) |
| Candle vor llama.cpp | Auch der exakt von mistral.rs eingebundene Candle-Commit bietet CPU/CUDA/Metal, kein ROCm/Vulkan; [Nachweis](../runtime/candle/README.md) |
| Eingebetteter AMD-Ausweichweg | `llama-cpp-2 0.1.160` mit `rocm` erfolgreich gebaut, Modell geladen und erneut geladen; [Probeprogramm und Messwerte](../runtime/llamacpp/README.md) |
| Streaming und Tokenzahlen | Tokenweise Ausgabe mit Zeitstempeln und tatsächlichen Tokenzahlen; noch keine OpenAI-/Anthropic-SSE-Implementierung |
| Parallelität | Zwei Sequenzen gleichzeitig in einem gemeinsamen Kontext praktisch geprüft |
| Entladen | Warme Modellzyklen geben den Modellanteil wieder frei. Einmalige Restbelegung gegenüber kaltem Prozess bleibt; vollständige Rückkehr erst nach Prozessende. Der Neustartpfad bleibt erforderlich |
| Bildserver | HIP-Bau von stable-diffusion.cpp, Qwen Image 2.1 mit lokalem Textencoder und VAE; [HTTP-Probe und Messwerte](../runtime/sdcpp/README.md) |
| Chat → Bild → Chat | Direkter Versuch erfolgreich; erste Bildanfrage einschließlich Laden 8,087 s bei 512 × 512 und vier Schritten |
| Fertiger Verwalter | Derselbe Modellwechsel mit llama-swap erfolgreich; [Vergleich](../runtime/llama-swap/README.md) |
| Startparameter | Tatsächliche Hilfen von Ollama, LM Studio und Backend-Programmen geprüft; [Zuordnung](../reference-clients/README.md) |

Die Quellenprüfung ersetzt keinen positiven Laufzeitnachweis für mistral.rs oder Candle.
Für die vorhandene AMD-Karte ist das ein negatives Eignungsergebnis; deshalb wurde der
in der Spezifikation vorgesehene eingebettete Ausweichweg praktisch geprüft.

## Entscheidung für die Umsetzung

1. Eigenbau fortführen. llama-swap funktioniert im Vergleich, wird aber keine
   Laufzeitabhängigkeit von Eliezer. Eliezer erhält seinen eigenen Verwalter,
   das VRAM-Buch und den LM-Studio-kompatiblen Verwaltungsvertrag.
2. Auf AMD das geprüfte eingebettete llama.cpp-Binding einplanen. mistral.rs wird
   auf dieser Karte nicht als GPU-Laufzeit angeboten. Eine Unterstützung anderer
   Hardware mit mistral.rs ist durch diese AMD-Prüfung nicht bestätigt.
3. Bildmodelle mit dem geprüften stable-diffusion.cpp-Server als Kindprozess betreiben.
4. Restbelegung messen und den spezifizierten `unload_leak`-/`needs_restart`-Pfad
   umsetzen. Eine garantierte Rückkehr zur kalten Ausgangsbelegung allein durch
   Handle-Drop ist für das geprüfte ROCm-Binding nicht nachgewiesen.
5. Die niedrige Binding-Schnittstelle erfordert eigene Umsetzung für HTTP/SSE,
   Werkzeugaufrufe, Denkoptionen, Abbruch und Vision. Die Lücken werden in M4
   umgesetzt und geprüft; keine dieser Fähigkeiten wird hier als vorhanden ausgegeben.
6. Die aktuelle mistral.rs-Release verlangt Rust 1.94. Eliezers Vorgabe 1.88
   wird nicht still geändert. Vor Aufnahme dieses optionalen Backends muss die
   Mindestversion ausdrücklich entschieden werden.

## Prüfstatus

- Rust-Probe: Formatprüfung und Clippy mit `-D warnings` grün.
- Normaler Cargo-Testlauf: ein GPU-Test ausdrücklich ignoriert, kein Netz- oder GPU-Zugriff.
- Angeforderter GPU-Test: ein Test bestanden; kein automatischer Beweis vollständiger Speicherfreigabe.
- Python-Prüfprogramme syntaktisch geprüft, direkter Wechsel und Proxy-Vergleich ausgeführt.
- M1 ist eine Machbarkeitsentscheidung mit diesen Grenzen, keine Abnahme des späteren Dienstes.
  M2 beginnt erst nach Freigabe.
