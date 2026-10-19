# AGENTS.md – Arbeitsanweisung für Codex und andere Agenten

Eliezer ist ein Modellverwalter in **Rust** und ersetzt LM Studio für
[MultiGPT](https://github.com/Beerlesklopfer/multi-gpt). Die vollständige Beschreibung steht in
[docs/Spezifikation.md](docs/Spezifikation.md); die Meilensteine mit Abnahmekriterien in
[docs/Implementierung.md](docs/Implementierung.md). **Lies beide, bevor du Code schreibst.**

## Arbeitsweise

- Meilensteine der Reihe nach (M1 … M6). Je Meilenstein: umsetzen, Tests, kurze Notiz in
  `docs/Stand.md` (was läuft, was nicht, Messwerte), dann stoppen und auf Freigabe warten.
- Nichts erfinden: Schnittstellen und Kommandozeilen fremder Programme (mistral.rs,
  stable-diffusion.cpp) **zuerst am installierten Programm prüfen** (`--help`, Probeaufruf) und das
  Ergebnis in `docs/Laufzeiten.md` festhalten. Die Spezifikation nennt dort nur, was zu klären ist.
- Kleine, nachvollziehbare Commits mit deutscher Beschreibung im Imperativ.
- Versionsnummern legt der Nutzer fest. Keine Tags, keine Releases, kein `cargo publish`.

## Sprache und Stil

- Bezeichner, Typen, Funktionen, Felder der API: **englisch**. Dokumentation, Kommentare,
  Fehlermeldungen für Menschen, Logzeilen: **deutsch**.
- Rust stable, `rust-version = "1.88"`, Edition 2024. Workspace mit mehreren Crates (siehe Spezifikation 3).
- `cargo fmt --check`, `cargo clippy --all-targets -- -D warnings`, `cargo test` müssen grün sein.
- Kein `unsafe` ohne Begründung im Kommentar; kein `unwrap()`/`expect()` außerhalb von Tests und `main`.
- Async mit `tokio`, HTTP-Server `axum`, Client `reqwest` (rustls), Serialisierung `serde`.
  Neue Abhängigkeiten sparsam und mit Begründung in `docs/Abhaengigkeiten.md`.

## Sicherheit (verbindlich)

- Der Hugging-Face-Token und API-Schlüssel stehen **nie** im Repository, in Logs, in Fehlermeldungen,
  in Testdaten oder in Prozess-Argumenten (sichtbar in `ps`). Sie kommen aus einer Datei (0600) oder
  der Umgebung und werden nur an Kindprozesse per Umgebungsvariable weitergereicht.
- Tests laufen **ohne Netz und ohne GPU**: Hintergrundprogramme werden durch einen Prüfstand
  (`fake-backend`, Spezifikation 9) ersetzt. Tests, die eine GPU brauchen, tragen `#[ignore]`
  und eine Begründung; sie laufen nur auf Anforderung (`cargo test -- --ignored`).
- Geladen wird nur, was in der Positivliste der Konfiguration steht.
- Keine Eingabe aus HTTP-Anfragen gelangt in Kommandozeilen, Pfade oder Shell ohne Prüfung gegen
  die Konfiguration (Modellschlüssel sind Namen aus der Liste, keine Pfade).

## Nicht-Ziele

Kein Cluster, keine Benutzerverwaltung, keine Abrechnung, keine Weboberfläche. Rechte, Rangfolge
und Abrechnung liegen in MultiGPT; Eliezer führt nur aus und meldet Tatsachen.
