# Implementierungsgerüst Eliezer

Stand 2026-10-19, Entwurf. Eliezer ersetzt LM Studio als Modellserver von
[MultiGPT](https://github.com/Beerlesklopfer/multi-gpt) (dort Meilenstein M16 = Anbindung).
**Vorab lesen:** [../AGENTS.md](../AGENTS.md) (Regeln) und [Spezifikation.md](Spezifikation.md)
(Aufbau, Konfiguration, Schnittstellen, Planer, Tests). Dieses Dokument ordnet die Arbeit.

Entschieden: alles Rust, Laufzeit mistral.rs, Modelle per Hugging-Face-Token (nur auf dem GPU-Rechner),
im Chat eine einzige Bildauswahl. Zielversion legt der Nutzer fest.

Arbeitsweise: Meilensteine der Reihe nach. Nach jedem Meilenstein `cargo fmt --check`, `cargo clippy -D warnings`,
`cargo test` grün, kurze Notiz in `docs/Stand.md`, dann Freigabe abwarten. Die Spezifikation wird bei
Entscheidungen nachgeführt (nicht still abweichen).

Ergebnisse jedes Meilensteins werden zusätzlich in `debian/changelog` im Debian-Format
festgehalten. Versionsnummern bestimmt der Nutzer; offene Arbeiten werden nicht als
abgeschlossen ausgewiesen. Zielhardware: mindestens 8 GB VRAM, mit angepasster Modellauswahl.

## Meilensteine

### M1 – Versuch auf dem GPU-Rechner (isolierte Probeprogramme, kein Produktionscode)

**Abgeschlossen:** [Ergebnis und Entscheidungen](../tests/m1/README.md),
Meilensteinfassung 0.1.0; Eintrag in `debian/changelog`.

Ziel: Schnittstellen der Fremdprogramme **feststellen**, nicht raten.

Start-, Lade- und Entladeparameter werden anhand der vorhandenen Ollama- und
LM-Studio-Installationen übernommen und auf die eingebetteten Laufzeiten abgebildet.
Geprüfte Kommandozeilen, Zuordnungen und Ergebnisse werden unter
`tests/reference-clients/` dokumentiert.

- **M1-01** mistral.rs **als Crate** in einem kleinen Testprogramm einbinden (kein Kindprozess): Modell laden, antworten, entladen. Feststellen und in `docs/Laufzeiten.md` festhalten: Version, Cargo-Features je Hardware, Builder-Aufrufe (Text, Vision), Ob/wie Entladen und Neuladen im Prozess geht und der VRAM danach frei ist, Streaming und `usage`, Werkzeugaufrufe, Denkoptionen, Chatvorlage, ob `mistralrs-server-core` einbettbar ist. Messwerte: Ladezeit, Token/s, VRAM, zwei gleichzeitige Anfragen.
- **M1-02** stable-diffusion.cpp (Server) mit `qwen-image-2.1-UC-Q4_K_M.gguf`, Textencoder und VAE: Befehlszeile,
  Endpunkte, Zeit je Bild, VRAM. Ebenfalls in `docs/Laufzeiten.md`.
- **M1-02a** Hersteller der Karte klären (`/sys/class/drm/card*/device/vendor`) (NVIDIA oder AMD). mistral.rs und stable-diffusion.cpp je auf dieser Karte bauen und prüfen,
  welche Beschleunigung läuft (CUDA, ROCm/HIP, Vulkan); bei AMD zusätzlich zuerst Candle direkt (reines Rust, laut Architekturplan vorgesehen), dann das llama.cpp-Binding (Vulkan/ROCm bauen und messen). Ergebnis je Backend in `docs/Laufzeiten.md`.
  Läuft das mistral.rs-Crate dort nicht, wird das **eingebettete** Backend `llamacpp` (Binding-Crate, Spezifikation 8) Teil von M4;
  dann das Binding-Crate wählen und seine Lücken (Chatvorlage, Werkzeugaufrufe, Vision, Parallelität) in `docs/Laufzeiten.md` festhalten.
- **M1-03** Wechsel Chat ↔ Bild auf der 24-GB-Karte von Hand; zum Vergleich llama-swap mit denselben Modellen.
- **M1-04** Entscheidung: fertiger Verwalter (llama-swap) oder Eigenbau; mistral.rs eingebettet (entschieden) – Ausweichweg `llamacpp` (ebenfalls eingebettet) für Karten, die das Crate nicht bedient.
  Fällt sie auf „fertig“, wird M2 bis M3 durch eine Anpassung ersetzt und die Spezifikation entsprechend gekürzt.

*Abnahme:* `docs/Laufzeiten.md` enthält je Backend eine **lauffähige Befehlszeile und die gemessenen Werte**; die Entscheidung steht darin.

### M2 – Gerüst

- **M2-01** Workspace nach Spezifikation 3, Lizenz (Nutzer wählt), `rust-toolchain.toml`.
- **M2-02** CI (`.github/workflows/rust.yml`): Format, Clippy, Test, nur Linux.
- **M2-03** Konfiguration (`eliezer-core::config`): Strukturen mit `deny_unknown_fields`, Prüfungen (doppelte Schlüssel,
  Portbereiche, Pfade), deutsche Fehlermeldungen mit Zeile. Test: Beispiel aus Spezifikation 4 lädt; je ein Fehlerfall.
- **M2-03a** Zentrale Modellablage (`storage.models_dir`) und Pfadprüfung: alle Modellbestandteile
  sowie der HF-Cache darunter; Pfade außerhalb der Ablage auch über Symlinks ablehnen.
- **M2-04** Programmrahmen (`eliezer`): Argumente (`--config`, `--check-config`), `tracing`, Start/Beenden mit `SIGTERM`,
  `SIGHUP`-Neuladen als Platzhalter, Einheit `packaging/eliezer.service`.

- **M2-05** Container-Bau und `compose.yaml` einrichten, ergänzt um `compose.volume.yaml`:
  eigenes Modellverzeichnis als Bind-Mount oder alternativ persistentes Docker-/Podman-Volume;
  persistente Zustandsdaten, schreibgeschützte Konfiguration, separate Geheimnisdateien und
  GPU-Zugriff nach M1. Start, Stopp und Neuerstellung mit beiden Ablagevarianten prüfen;
  Docker-/Podman-Voraussetzungen und Unterschiede dokumentieren.

*Abnahme:* Compose-Konfigurationen sind geprüft, der Dienst startet und Modelldateien bleiben
bei Container-Neuerstellung erhalten. `cargo run -- --check-config beispiel.toml` meldet „in Ordnung“; fehlerhafte Dateien nennen Zeile und Feld.

### M3 – Verwalter (ohne echte Backends)

- **M3-01** `eliezer-core`: VRAM-Rechnung (Spezifikation 6) mit allen Architekturtypen; Tests mit den Beispielwerten.
- **M3-02** Zustandsautomat und Planer (Spezifikation 7), `Clock`-Trait; Tests für Verdrängung, Anheften, Bündeln,
  Leerlauf, Warteschlange, Fehlerzustand.
- **M3-03** `eliezer-gpu`: **ein** Trait `GpuProbe` (Spezifikation 6a) mit den Verfahren `nvml`, `nvidia-smi`, `sysfs`, `rocm-smi`, `none` und Prüfstand; Herstellererkennung über `device/vendor`; einheitliche Bedeutung (Byte, gesamt belegt), Zeitlimit und Rückfall; ein gemeinsamer Testsatz.
- **M3-04** `eliezer-backend`: Trait `Backend` und **fake-backend**; Verwalter-Task (`Supervisor`) mit mpsc/oneshot.
- **M3-05** `eliezer-api`: `/api/v1/models`, `/load`, `/unload`, `/api/v1/status`, `/healthz`, Authentifizierung, Fehlerformat;
  Chat-Weiterleitung `/v1/chat/completions` mit **ungepuffertem SSE** gegen das fake-backend.
- **M3-05a** Anthropic-Schnittstelle (Spezifikation 5.1a): Modul `eliezer-api::anthropic` mit Übersetzung hin und zurück, `POST /v1/messages`
  (mit und ohne Streaming), `count_tokens`, Modellliste, Fehlerformat; Tests gegen das fake-backend.
- **M3-06** Vertragstest (Spezifikation 11) mit `tests/contract/`-Dateien.

*Abnahme:* Alle Integrationstests aus Spezifikation 11 laufen ohne Netz und GPU; dieselbe Anfrage liefert über OpenAI- und Anthropic-Schnittstelle denselben Inhalt; MultiGPT (Provider `lmstudio`, `base_url`
auf Eliezer mit fake-backend) listet Modelle, lädt und entlädt.

### M4 – Laufzeit mistral.rs

- **M4-01** Backend `mistralrs` als eingebettetes Crate nach `docs/Laufzeiten.md`: Laden/Entladen über das Handle, Prüfung „VRAM frei nach Entladen“ (`unload_leak`, `needs_restart`), Rechenarbeit auf eigenen Threads, Panic-Abfang, Feature `mistralrs-cuda`.
- **M4-01a** (nur wenn M1 es verlangt) Backend `llamacpp` eingebettet über das gewählte Binding-Crate: Laden/Entladen, Chatvorlage, Streaming, Parallelität, Werkzeugaufrufe, `usage`, Vision soweit möglich; gleiche Prüfungen wie M4-01.
- **M4-02** Chat und Vision, Werkzeugaufrufe, Streaming mit `usage`; Chatvorlage je Modell.
- **M4-03a** Anthropic-Schnittstelle gegen ein echtes Modell: Werkzeugaufruf und Streaming mit einem Anthropic-Client prüfen (Spezifikation 12, Punkt 4).
- **M4-03** Denkoptionen (`reasoning_effort` → Backend), Statistik im letzten SSE-Block, Gegenprobe der VRAM-Rechnung (gemessener Wert wird gespeichert).
- **M4-04** GPU-Tests (`#[ignore]`) und Messwerte in `docs/Stand.md`.

*Abnahme:* Mit nur geänderter `base_url` arbeitet MultiGPT mit einem echten Chat-Modell (Modellliste, Laden mit Kontext und
Parallelität, Chat mit Streaming, Denkoptionen, Statistik).

### M5 – Modellbezug

- **M5-00** Gezielte Suche nach spezialisierten Modellen auf Hugging Face: Suchschnittstelle und
  Ergebnisformat festlegen; Einsatzzweck, Format, Quantisierung, Lizenz, Backend-Unterstützung und
  VRAM-Bedarf berücksichtigen. Kandidaten mit Modellkarte und Zusatzdateien prüfen; Aufnahme in
  die Positivliste und Download bleiben ausdrückliche Verwaltungsschritte.
- **M5-01** Positivliste mit Revision/Commit-Hash (`pins.json`), Abweichungen melden.
- **M5-02** `POST /api/v1/models/download`, Fortschritt in `/api/v1/status`, Prüfung `cache_limit_gb` (507), Lizenzhinweis bei 401/403.
- **M5-03** Token aus Datei (0600, Rechte beim Start prüfen) oder `HF_TOKEN`; nur per Umgebung an Kindprozesse; Prüfskript `scripts/check-secrets.sh`
  (sucht in Logs, `ps`, Fehlermeldungen).

*Abnahme:* Spezialisierte Modelle lassen sich gezielt finden und auf Eignung prüfen. Downloads landen
ausschließlich in der zentralen Modellablage. Modelle lassen sich auf Anforderung laden; kein Token in Logs, Prozessliste oder Meldungen.

### M6 – Bilddienst

- **M6-01** Backend `sdcpp` (Qwen-Image-GGUF) nach `docs/Laufzeiten.md`; `/v1/images/generations`; exklusive Planung (Spezifikation 7, Punkt 4).
- **M6-02** GPU-Test Wechsel Chat ↔ Bild, Messwerte.
- **M6-03** (später) FLUX.2 klein 4B in Rust, sobald die Architektur in mistral.rs/candle vorliegt: eigener Plan, nicht Teil der Abnahme.

*Abnahme (gesamt):* Spezifikation 12.

## Reihenfolge

```
M1 ─> M2 ─> M3 ─┬─> M4 ─> M5
                └─> M6
```
