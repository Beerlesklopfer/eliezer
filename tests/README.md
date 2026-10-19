# Prüfungen

Alle Prüfungen, Probeprogramme, Anleitungen und Ergebnisprotokolle liegen hier.
Abschluss der ersten Laufzeitprüfung: [M1-Ergebnis](m1/README.md).
Automatisierte Tests laufen ohne Netz und GPU; manuelle Hardwareprüfungen werden
ausdrücklich gestartet. Der spätere Cargo-Workspace bindet Hardwareprüfungen nur
als ignorierte Tests mit Begründung ein.

```text
tests/
├── README.md
├── m1/README.md              Abschluss und Entscheidungen
├── reference-clients/
│   └── README.md             Geprüfte Ollama-/LM-Studio-Kommandozeilen
├── environment/
│   ├── README.md             Voraussetzungen und Aufruf
│   └── check.sh              Lokale Bestandsaufnahme
└── runtime/
    ├── mistralrs/README.md   Versions- und Featureprüfung
    ├── candle/README.md      Featureprüfung
    ├── sdcpp/               Bildserver und Chat-/Bildwechsel
    ├── llama-swap/          Vergleich mit fertigem Verwalter
    └── llamacpp/
        ├── README.md         Versuch, Durchführung und Ergebnisse
        ├── Cargo.toml        Isoliertes Probeprogramm
        ├── Cargo.lock        Festgehaltene Abhängigkeiten
        └── src/main.rs       Laden, Generieren, Entladen, Neuladen
```

Weitere Laufzeiten erhalten eigene Verzeichnisse unter `runtime/`.
Vertragstests und Integrationstests werden später unter `contract/` beziehungsweise
`integration/` ergänzt. Rust-Bauausgaben in `target/` bleiben ignoriert.
