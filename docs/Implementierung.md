# Implementierungsgerüst Eliezer

Stand 2026-10-19, Entwurf. Eliezer ersetzt LM Studio als Modellserver von
[MultiGPT](https://github.com/Beerlesklopfer/multi-gpt). Die Anbindung auf Seite von MultiGPT steht
dort als Meilenstein M16 in `docs/Implementierung.md`; Hintergrund und Recherche in MultiGPT unter
`docs/plans/Runner-Plotter.md`.

Entschieden: alles Rust, Laufzeit mistral.rs, Modelle per Hugging-Face-Token (nur auf dem GPU-Rechner,
nie im Repository oder in Logs), im Chat eine einzige Bildauswahl. Offen: Verwalter fertig einsetzen
(llama-swap, LocalAI) oder klein in Rust bauen – entscheidet M1. Zielversion legt der Nutzer fest.

## Arbeitspakete je Meilenstein

### M1 – Versuch auf dem GPU-Rechner

- **M1-01** mistral.rs mit einem Chat-Modell: Ladezeit, Token/s, VRAM, `unload`/`reload`, Anfrage an ein entladenes Modell, zwei gleichzeitige Anfragen.
- **M1-02** stable-diffusion.cpp mit `qwen-image-2.1-UC-Q4_K_M.gguf` (Textencoder und VAE prüfen): Zeit je Bild, VRAM.
- **M1-03** Wechsel Chat ↔ Bild auf der 24-GB-Karte; zum Vergleich llama-swap mit denselben Modellen.
- **M1-04** Entscheidung: fertiger Verwalter oder Eigenbau; Einbettung von mistral.rs (Bibliothek oder eigener Prozess).

*Abnahme:* Messwerte liegen vor und die Entscheidung steht in diesem Dokument.

### M2 – Gerüst

- **M2-01** Cargo-Workspace, Lizenz (vom Nutzer zu wählen), `rust-toolchain`.
- **M2-02** CI: Format, Clippy, Tests (wie `runner-rs` in MultiGPT).
- **M2-03** Konfigurationsdatei und Dienst-Unit (`systemd`), Protokollierung ohne Token.

### M3 – Verwalter

- **M3-01** Routing nach dem Feld `model`, Zustände geladen/lädt/entladen.
- **M3-02** `ttl` je Modell (Leerlauf-Entladen), Zeitgeber im Verwalter.
- **M3-03** VRAM-Buch: Modelldatei + KV-Cache + Zuschlag, Reserve für Anzeige/System.
- **M3-04** Warteschlange und Verdrängung nach Priorität; laufende Anfragen werden abgewartet; nichts laden, wenn es nicht passt.
- **M3-05** Schnittstelle für MultiGPT: Modellliste mit Zustand und Speicherschätzung, Laden/Entladen mit Kontext und Parallelität, Status und VRAM, Abbrechen.

### M4 – Laufzeit mistral.rs

- **M4-01** Chat und Vision, OpenAI-kompatibel (`/v1`).
- **M4-02** Denkoptionen je Modell (`none` = aus), Werkzeugaufrufe.
- **M4-03** Statistik je Antwort (erstes Token, Token/s), Speicherschätzung bekannter Architekturen.

### M5 – Modellbezug

- **M5-01** Positivliste (Modell-ID mit Revision), geladen wird nur, was dort steht.
- **M5-02** Cache-Volume (`HF_HOME`) mit Größengrenze; Herunterladen nur auf Anforderung, mit Fortschritt.
- **M5-03** Hugging-Face-Token: Datei 0600 oder Umgebungsvariable des Dienstes, nur an die Laufzeit weitergereicht.

### M6 – Bilddienst

- **M6-01** Übergang: stable-diffusion.cpp (Qwen-Image-GGUF) hinter demselben Vertrag, eine einzige Bildauswahl.
- **M6-02** Ziel: FLUX.2 klein 4B (Apache-2.0) in Rust, sobald die Architektur in mistral.rs/candle vorliegt.

*Abnahme (gesamt):* Ein Chat-Modell und ein Bildauftrag wechseln sich auf der 24-GB-Karte ab, das Chat-Modell kommt danach von selbst zurück, die Statistik stimmt mit der Messung überein, MultiGPT läuft ohne LM Studio.

## Reihenfolge

```
M1 ─> M2 ─> M3 ─┬─> M4 ─> M5
                └─> M6
```
