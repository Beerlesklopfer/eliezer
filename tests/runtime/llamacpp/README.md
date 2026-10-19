# Eingebettetes llama.cpp mit ROCm

Manueller GPU-Versuch mit `llama-cpp-2 = 0.1.160`, Feature `rocm`.
Das Probeprogramm ist unabhängig vom späteren Eliezer-Workspace und wird nicht
bei gewöhnlichen Tests gestartet. Es benötigt ein vorhandenes lokales GGUF-Modell.
Es führt weder Downloads noch HTTP-Anfragen aus.

## Umfang

GPU-Auslagerung prüfen, Modell mit allen Schichten laden, 32 Token erzeugen,
Kontext und Modell fallen lassen, bis zu zehn Sekunden Speicherfreigabe beobachten
und den Ablauf im selben Prozess wiederholen. Gemessen wird die gesamte
VRAM-Belegung der Karte über AMD-sysfs; fremde Prozesse können die Werte beeinflussen.
Die Beobachtung verwendet eine Toleranz von 100 MB zur vorherigen Belegung.
Zwei parallele Sequenzen werden geprüft; Werkzeugaufrufe, Vision und parallele
HTTP-Anfragen sind damit noch nicht nachgewiesen.

## Bauen und ausführen

Für die geprüfte Architektur `gfx1100` und eine lokale ROCm-Installation:

```bash
export PATH="$HOME/.cargo/bin:$PATH"
export LD_LIBRARY_PATH=/opt/rocm/lib
CARGO_BUILD_JOBS=4 AMDGPU_TARGETS=gfx1100 \
  CMAKE_POSITION_INDEPENDENT_CODE=ON CMAKE_HIP_FLAGS=-fPIC \
  cargo build --locked --manifest-path tests/runtime/llamacpp/Cargo.toml

tests/runtime/llamacpp/target/debug/eliezer-m1-probe /pfad/zum/modell.gguf
```

Der Gerätezugriff muss für den ausführenden Prozess verfügbar sein. Ausführung
in einer Sandbox ohne GPU-Geräte liefert keine Aussage über die Host-Unterstützung.
Die im Programm verwendete Messquelle `card0` muss zur getesteten GPU passen.

## Erster Versuch vom 19.10.2026 (vor der Erweiterung)

- Erster Bau ohne PIC-Flags: HIP-Objekte konnten nicht in das Rust-Programm
  gelinkt werden (`R_X86_64_32` beziehungsweise `R_X86_64_32S`).
- Frischer Bau mit `CMAKE_POSITION_INDEPENDENT_CODE=ON` und
  `CMAKE_HIP_FLAGS=-fPIC` erfolgreich. Rust-Programm und llama.cpp sind eingebettet;
  kein `llama-server` wurde für diesen Modellversuch gestartet.
- `cargo fmt --check` und `cargo clippy --all-targets -- -D warnings` erfolgreich.
- Manueller GPU-Lauf mit lokalem `Mistral-7B-Instruct-v0.3-Q4_K_M.gguf` erfolgreich:
  zwei Lade-/Antwort-/Entladezyklen im selben Prozess, Exitcode 0.
- Erster Ladevorgang: 1 467 ms; 33/33 Schichten auf der GPU, Modellpuffer 4 097,52 MiB.
- Kontext 512 Token; Prompt 16 Token; Ausgabe jeweils 32 Token.
  Zeit bis zum ersten Token: 57 ms beziehungsweise 23 ms.
  Gesamtzeit der Generierung einschließlich Promptverarbeitung: 299 ms beziehungsweise 255 ms.
- VRAM vor dem ersten Laden: 1 377 079 296 Byte, mit Modell 5 821 272 064 Byte,
  mit Kontext 6 094 217 216 Byte. Nach dem ersten Entladen: 1 620 803 584 Byte
  nach 10 008 ms; die 100-MB-Toleranz zur Ausgangsbelegung wurde nicht erreicht.
- Zweiter Zyklus: mit Kontext 6 109 556 736 Byte, nach Entladen 1 620 811 776 Byte
  nach 100 ms. Erneutes Laden und Antworten funktionieren.
- Die erste Restdifferenz beträgt rund 244 MB. Die Ursache ist noch nicht geklärt
  (gemessen wurde die gesamte Karte). Vollständige Speicherfreigabe ist damit
  nicht nachgewiesen. Das Probeprogramm protokolliert die Werte; Exitcode 0
  ist hier kein automatischer Abnahmenachweis für die Speicherfreigabe.
- Bei Änderung der CMake-Optionen müssen zuvor erzeugte Backend-Objekte neu
  gebaut werden; das Binding erkennt diese Umgebungsänderungen nicht zuverlässig.

## Erweiterte Prüfung und Abschlussbefund

Der aktuelle Versuch verwendet die Chatvorlage aus dem GGUF, gibt Token mit
Zeitstempeln unmittelbar aus und betreibt im dritten Durchlauf zwei Sequenzen
in einem gemeinsamen Kontext mit 1 024 Token. Das Kontextbudget wird geteilt.

Messprotokolle: [drei Durchläufe](results/2026-10-19.txt) und
[Kontrolle mit expliziter HIP-Initialisierung](results/2026-10-19-hip-init.txt).
Die explizite Initialisierung verwendet `hipFree(NULL)` entsprechend dem
installierten HIP-Header. Der einzige FFI-Aufruf ist im Quelltext begründet.

| Durchlauf | Ladezeit | Prompt je Sequenz | Ausgabe | Erstes Token | Zeit inklusive Prompt | Ausgabetoken/s inklusive Prompt |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 292 ms | 16 | 32 | 57 ms | 301 ms | 106,3 |
| 2 | 267 ms | 16 | 32 | 23 ms | 257 ms | 124,5 |
| 3, zwei Sequenzen | 262 ms | 16 | 2 × 32 | je 48 ms | 317 ms | 201,9 gesamt |

Nach den warmen Durchläufen verbleiben gegenüber dem jeweiligen Ausgangswert
4 096 beziehungsweise 0 Byte zusätzlich. Die kalte Ausgangsbelegung wird im
Prozess dagegen nicht vollständig wieder erreicht, auch nicht nach Backend-Drop.
Die zusätzliche HIP-Initialisierung beseitigt diesen Befund nicht. Die genaue
Ursache der einmaligen Restbelegung ist nicht bewiesen. Nach Prozessende ist
in den Wechseltests die Ausgangsbelegung wieder erreicht beziehungsweise liegt
die Abweichung innerhalb der beobachteten Fremdbelegung.

**Folgerung:** Laden, Token-Ausgabe, zwei Sequenzen und Neuladen funktionieren.
Eine Garantie vollständiger Freigabe bis auf die kalte Ausgangsbelegung besteht
nicht; der in der Spezifikation geforderte `unload_leak`-/`needs_restart`-Pfad
bleibt für die spätere Einbettung erforderlich. Die GPU-Prüfung ist ein
Funktionstest, keine automatische Speicherleck-Abnahme.

Der aktuelle Testlauf enthält einen ignorierten Hardwaretest:

```bash
export PATH="$HOME/.cargo/bin:$PATH"
export LD_LIBRARY_PATH=/opt/rocm/lib
# Normaler Testlauf: der GPU-Test bleibt ignoriert.
cargo test --locked --manifest-path tests/runtime/llamacpp/Cargo.toml
# Nur ausdrücklich auf dem GPU-Rechner:
ELIEZER_TEST_MODEL=/pfad/zum/modell.gguf \
  cargo test --locked --manifest-path tests/runtime/llamacpp/Cargo.toml -- --ignored --nocapture
```

Formatprüfung, Clippy mit `-D warnings`, normaler Testlauf (1 ignorierter GPU-Test)
und ausdrücklich gestarteter GPU-Test (1 bestanden) wurden ausgeführt.

## Bekannte Binding-Lücken für M4

- OpenAI-/Anthropic-Antworten und SSE-Ereignisse muss Eliezer selbst aufbauen.
- Tokenzahlen sind über die tatsächlichen Tokenfolgen verfügbar; HTTP-`usage`
  muss der Adapter daraus bilden.
- Das Binding übernimmt keine vollständige Werkzeugaufruf-Schnittstelle.
  Werkzeugdefinitionen, Vorlagenintegration und Erkennung der Ausgabe werden in M4 gebaut.
- Denkoptionen müssen modellabhängig in Vorlage und Parameter übersetzt werden.
- Vision erfordert die zusätzliche `mtmd`-Einbindung und passende Projektoren.
  Diese Kombination wurde hier nicht gebaut oder getestet; sie bleibt ein eigener M4-Nachweis.
- Abbruch und parallele HTTP-Anfragen sind Aufgaben des späteren Adapters.
  Zwei Sequenzen in einem Kontext sind hier praktisch nachgewiesen.
