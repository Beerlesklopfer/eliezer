# M2 – Programmrahmen und Installation

## Automatisierte Tests

```bash
export PATH="$HOME/.cargo/bin:$PATH"
cargo fmt --all --check
cargo clippy --workspace --all-targets --offline --locked -- -D warnings
cargo test --workspace --offline --locked
```

Die Abhängigkeiten müssen zuvor einmal bezogen worden sein (`cargo fetch --locked`).
Die Tests selbst benötigen kein Netz und keine GPU. Die getrennten M1-Probeprogramme
sind nicht Mitglied des Produkt-Workspaces.

| Datei | Inhalt |
|---|---|
| `config.rs` | Vollständiges Konfigurationsbeispiel, Standardwerte, unbekannte Felder, Typfehler, Quoten, Modellquellen, doppelte Schlüssel, Zeile/Feld und Modellablage |
| `cli.rs` | CLI-Vorrang, Exitcodes, sichere Fehlermeldungen und echte Symlinks einschließlich nicht auflösbarer Ziele |

Ein `PathResolver` hält die Dateisystemauflösung außerhalb der reinen Core-Logik.
Die CLI-Tests verwenden ausschließlich eigene temporäre Verzeichnisse.

## Manuelle Betriebsprüfungen

Die folgenden Aufrufe brauchen einen gebauten Programmrahmen beziehungsweise
lokalen Containerbetrieb. Sie sprechen nur Loopback an und laden keine Modelle:

```bash
cargo build --offline --locked -p eliezer
python3 tests/m2/service.py
python3 tests/m2/verify-unit.py
podman build -t localhost/eliezer:0.2.0 .
python3 tests/m2/container.py
```

- `service.py`: `/healthz`, ungültiger und gültiger SIGHUP, keine Testwerte im Log,
  geordnetes SIGTERM. Gültige SIGHUP-Dateien werden in M2 nur geprüft.
- `verify-unit.py`: Syntaxprüfung über `systemd-analyze verify`. Im temporären
  Prüfexemplar wird nur der noch nicht installierte Programmpfad ersetzt.
  Kein Systembenutzer oder Systemdienst wird dabei angelegt.
- `container.py`: Beide Modellablagen, HTTP-Bereitschaft, Container-Neuerstellung,
  Erhalt eigener Modell- und Zustandsdateien. Verwendet einen eindeutigen
  Testprojektnamen und entfernt anschließend nur dessen eigene Volumes.

## Ergebnisse vom 19.10.2026

- Workspace mit fünf Crates, Edition 2024, Mindest-Rust-Version 1.88.
- Rust 1.99.0: Format, Clippy mit `-D warnings` und neun Tests erfolgreich.
- Dockerfile mit Rust 1.88.0 tatsächlich gebaut: Mindestversion bestätigt.
- HTTP-/Signalprüfung und Unit-Syntaxprüfung erfolgreich.
- Compose-Konfigurationen mit Docker Compose v5.6.0 validiert.
- Podman 5.4.2 / Podman-Compose 1.3.0: Bind-Mount und benanntes Modell-Volume
  einschließlich Container-Neuerstellung praktisch geprüft.
- Die Benutzerzuordnung erfordert beim vorhandenen Provider `x-podman.in_pod: false`.
- Portables tmpfs: `/run` mit Sticky-Bit; der unprivilegierte Einstieg erzeugt
  `/run/eliezer` mit eigener Identität und Modus 0700. Direkte tmpfs-Optionen
  `uid` und `gid` wurden vom vorhandenen Podman abgelehnt und werden nicht verwendet.
- GPU-Ergänzungen sind als Compose-Konfiguration geprüft. Kein GPU-Backend ist
  im M2-Image enthalten. Die rootlose GPU-Variante benötigt laut installierter
  Podman-Dokumentation `crun`; auf dem geprüften Host ist nur `runc` verfügbar.
  Ein GPU-Containerlauf wird daher hier nicht als nachgewiesen ausgegeben.

Ergebnisdateien liegen unter [results/](results/). Installation und Rechte sind
in [INSTALL.MD](../../INSTALL.MD) beschrieben.

## Abgrenzung

M2 lädt keine Modelle. Authentifizierung, Verwalter, Modellrouten und Übernahme
neuer Konfigurationen folgen in M3; echte Backends in M4/M6. Der vorgezogene
`/healthz`-Endpunkt dient der Bereitschaftsprüfung des Programmrahmens.
