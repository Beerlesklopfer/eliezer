# Abhängigkeiten und Voraussetzungen

Der Cargo-Workspace ist angelegt. Die Abhängigkeiten des Programmrahmens stehen
in `Cargo.lock`; Versionen und Begründungen sind in
[Abhängigkeitsentscheidungen](../Abhaengigkeiten.md) dokumentiert. Modelllaufzeiten
werden erst bei ihrer tatsächlichen Einbindung hinzugefügt.

## Rust und Entwicklungswerkzeuge

| Komponente | Zweck | Vorgabe / geprüfter Stand |
|---|---|---|
| Rust stable | Compiler, Edition 2024 | Mindestversion 1.88; lokal 1.99.0 unter `~/.cargo/bin` |
| Cargo | Bau und Tests | Lokal 1.99.0 unter `~/.cargo/bin` |
| rustfmt | Einheitliche Formatierung | Lokal 1.10.0-stable |
| Clippy | Statische Codeprüfung | Lokal 0.1.99 |

Die Benutzerinstallation muss vor der älteren Systeminstallation im `PATH` stehen:

```bash
export PATH="$HOME/.cargo/bin:$PATH"
```

Die Projektprüfungen sind `cargo fmt --check`,
`cargo clippy --all-targets -- -D warnings` und `cargo test`.
Sie prüfen den gesamten Workspace; die eigenständigen M1-GPU-Probeprogramme bleiben separat.

## Vorgesehene Rust-Crates

| Crate | Begründung |
|---|---|
| `tokio` | Asynchrone Laufzeit und Kommunikation mit dem Verwalter |
| `axum` | HTTP-Endpunkte und Streaming |
| `reqwest` mit rustls | Ausgehende HTTP-Anfragen ohne OpenSSL-Abhängigkeit im Client |
| `serde` | Typisierte Serialisierung und Konfiguration |
| `tracing` | Strukturierte Betriebslogs |
| `mistralrs` | Vorgesehene eingebettete Modelllaufzeit; aktueller geprüfter Git-Stand 0.9.5 verlangt Rust 1.94. Version und Hardware-Eignung sind vor Aufnahme festzulegen |

Neue Crates werden sparsam ergänzt und mit ihrer Begründung in
[`docs/Abhaengigkeiten.md`](../Abhaengigkeiten.md) festgehalten.
Diese Seite ersetzt weder `Cargo.toml` die Versionsbindung in `Cargo.lock`.

## Modelllaufzeiten

| Komponente | Rolle | Stand |
|---|---|---|
| mistral.rs | Sprach- und Vision-Modelle im Eliezer-Prozess | Vorgesehen; Modelllauf und Speicherfreigabe noch zu prüfen |
| Candle direkt | Prüfung als Rust-Ausweichweg vor llama.cpp | AMD-Eignung noch offen |
| llama.cpp mit Rust-Binding | Optionaler eingebetteter Ausweichweg, besonders für AMD | Binding und Features noch nicht beschlossen |
| stable-diffusion.cpp | Separater Bildmodell-Prozess | Programm und Schnittstelle noch zu prüfen |
| fake-backend | Prüfstand für automatisierte Tests ohne Netz und GPU | In M3 zu implementieren |

Der lokal vorhandene LM-Studio-`llama-server` dient bisher nur der Geräteprüfung.
Sein Nachweis belegt noch keine funktionierende Einbettung von llama.cpp in Rust.

## GPU und Zugriffsrechte

Ziel ist Linux mit einer GPU und mindestens **8 GB VRAM**. Modellgröße,
Quantisierung und Kontext müssen zum verfügbaren Speicher passen.

- NVIDIA: CUDA für die passende Modelllaufzeit; VRAM-Messung über NVML oder `nvidia-smi`.
- AMD: je nach geprüftem Backend ROCm/HIP oder Vulkan; VRAM-Messung über sysfs
  beziehungsweise `rocm-smi` als Rückfallstufe.
- Auf dem aktuellen Host sind eine Radeon RX 7900 XTX (`gfx1100`), HIP 7.2.26015
  und funktionierende ROCm- sowie Vulkan-Geräteerkennung nachgewiesen.
- Der Benutzer gehört zu `video` und `render`. `/dev/dri/card0`,
  `/dev/dri/renderD128` und `/dev/kfd` sind auf dem Host vorhanden.
  Weitere Gruppenänderungen sind dafür derzeit nicht nötig.
- In der Agenten-Sandbox sind die Geräte nicht sichtbar. GPU-Versuche benötigen
  die Ausführung außerhalb dieser Sandbox. Container benötigen ebenfalls den
  passenden Gerätezugriff.

## Systemwerkzeuge und Betrieb

| Komponente | Zweck | Stand |
|---|---|---|
| CMake und C/C++-Compiler | Bau von stable-diffusion.cpp und gegebenenfalls llama.cpp | Lokal vorhanden; erfolgreicher Backend-Bau noch offen |
| GPU-Entwicklungsbibliotheken | Bau des gewählten GPU-Backends | Abhängig von den Ergebnissen in M1 |
| Docker Compose oder Podman mit Compose-Ausführung | Alternative zum nativen Betrieb; persistente Modellablage | Docker-Compose-Konfiguration validiert, Podman-Betrieb mit beiden Ablagearten geprüft |
| systemd | Nativer Dienstbetrieb und Neustart | Unit unter `packaging/eliezer.service` |
| Hugging Face | Modellsuche und ausdrücklich angeforderte Downloads | Laufzeitintegration für M5 vorgesehen |

Die `hf`-CLI ist ein optionales Werkzeug für manuelle Modellprüfungen;
eine Pflicht zur Installation für den späteren Eliezer-Dienst ist bislang nicht festgelegt.
Modelle, Tokenizer, Bildprojektoren, Textencoder und VAE liegen künftig im eigenen
Modellverzeichnis oder im persistenten Container-Volume. Geheimnisse werden separat gehalten.

Die gemessenen Ergebnisse und noch offenen Laufzeitprüfungen stehen in
[Laufzeiten.md](../Laufzeiten.md). Eine lokale Bestandsaufnahme ohne Downloads oder
Modellladen ist mit `bash tests/environment/check.sh` aus dem Repository möglich.
