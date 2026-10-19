# Eliezer installieren

Eliezer steht unter **GPL-3.0-only**; der vollständige Lizenztext steht in [LICENSE](LICENSE).

Der aktuelle Programmrahmen prüft die Konfiguration, bietet `/healthz` und verarbeitet
SIGTERM sowie SIGHUP. Modellverwaltung, Authentifizierung, Chat und Bildgenerierung
sind darin noch nicht implementiert. Die Laufzeit-Probeprogramme unter `tests/`
sind eigenständige Versuche.

## Voraussetzungen

- Linux; für den späteren Modellbetrieb eine unterstützte GPU mit mindestens 8 GB VRAM.
- Beim Bauen: Rust stable ab 1.88 und Cargo. Die Toolchain-Datei wählt `stable`;
  die deklarierte Mindestversion bleibt 1.88.
- Für Container: Docker mit Compose oder Podman mit einem Compose-Provider.
- Ein eigenes Modellverzeichnis oder ein persistentes Container-Volume.

Details zu GPU-Laufzeiten und Rechten: [Laufzeitvoraussetzungen](docs/Laufzeiten.md).
Für das reine M2-Gerüst wird keine GPU benötigt. GPU-Entwicklungsbibliotheken
werden erst beim Bau der entsprechenden Modellbackends erforderlich.

## Aus dem Quellcode bauen

Im ausgecheckten Repository:

```bash
export PATH="$HOME/.cargo/bin:$PATH"
rustc --version
cargo --version
cargo build --release --locked -p eliezer
```

Das Programm liegt anschließend unter `target/release/eliezer`.
Falls Rust noch fehlt, die Installation über die [offizielle Rust-Seite](https://www.rust-lang.org/tools/install)
vornehmen. Auf einem Rechner mit Rustup verwendet Eliezer die Benutzerinstallation.

Die Entwicklungsprüfungen lassen sich nach einmaligem Bezug der Abhängigkeiten
auch ohne Netz ausführen:

```bash
cargo fetch --locked
cargo fmt --all --check
cargo clippy --workspace --all-targets --offline --locked -- -D warnings
cargo test --workspace --offline --locked
```

GPU-Probeprogramme werden dabei nicht gestartet.

## Konfiguration prüfen und lokal starten

Eine vollständige Beispielkonfiguration liegt unter
[packaging/config.example.toml](packaging/config.example.toml).

```bash
cargo run --locked -- --check-config packaging/config.example.toml
```

Bei Erfolg erscheint `Konfiguration in Ordnung`. Ungültige Konfigurationen
enden mit Code 2 und einer deutschen Meldung mit Zeile und Feld.

Für eine Benutzerinstallation kann die ausführbare Datei lokal installiert werden:

```bash
install -Dm755 target/release/eliezer "$HOME/.local/bin/eliezer"
mkdir -p "$HOME/.config/eliezer" "$HOME/.local/share/eliezer/models"
cp packaging/config.example.toml "$HOME/.config/eliezer/config.toml"
```

In dieser Kopie die folgenden Pfade auf das tatsächlich verwendete, absolute
Modellverzeichnis einstellen:

- `storage.models_dir`, beispielsweise `/home/NAME/.local/share/eliezer/models`.
- `huggingface.cache_dir`, beispielsweise `/home/NAME/.local/share/eliezer/models/hf`.
- Sämtliche `source.file`-, Textencoder- und VAE-Pfade unterhalb der Modellablage.

TOML-Dateien ersetzen weder `$HOME` noch `~` durch das Home-Verzeichnis.
Es sind absolute Pfade einzutragen. Vorhandene Symlinks werden bei der Prüfung
aufgelöst; auch darüber dürfen Modellpfade die gemeinsame Ablage nicht verlassen.
Noch nicht vorhandene Downloadziele sind innerhalb dieser Ablage erlaubt.

```bash
"$HOME/.local/bin/eliezer" --config "$HOME/.config/eliezer/config.toml" --check-config
"$HOME/.local/bin/eliezer" --config "$HOME/.config/eliezer/config.toml"
```

In einem zweiten Terminal:

```bash
curl --fail http://127.0.0.1:1234/healthz
```

Erwartete Antwort: `ok`. Mit Strg+C wird der Dienst geordnet beendet.
Alternativ zum Argument kann `ELIEZER_CONFIG` den Konfigurationspfad angeben;
`--config` hat Vorrang. Ohne beides gilt `/etc/eliezer/config.toml`.

SIGHUP prüft im M2-Gerüst die Datei erneut. Gültige Änderungen werden noch nicht
in den laufenden Dienst übernommen; dafür ist vorerst ein Neustart erforderlich.
Eine ungültige Datei beendet den laufenden Dienst nicht.

## Als systemd-Dienst betreiben

Die folgenden Einrichtungsschritte benötigen Administrationsrechte. Den Benutzer
`eliezer` nur anlegen, wenn er noch nicht existiert:

```bash
sudo useradd --system --user-group --home-dir /var/lib/eliezer --shell /usr/sbin/nologin eliezer
sudo usermod -aG video,render eliezer
sudo install -Dm755 target/release/eliezer /usr/local/bin/eliezer
sudo install -d -m750 -o root -g eliezer /etc/eliezer
sudo install -d -m750 -o eliezer -g eliezer /var/lib/eliezer/models
sudo install -m640 -o root -g eliezer packaging/config.example.toml /etc/eliezer/config.toml
sudo install -m644 packaging/eliezer.service /etc/systemd/system/eliezer.service
```

Die Gerätegruppen müssen auf dem jeweiligen System existieren und zu den
Geräteberechtigungen passen. Bei abweichenden Gruppennamen auch
`SupplementaryGroups` in der Unit anpassen. Die Rechte des angemeldeten Benutzers
ersetzen nicht die des Dienstbenutzers.

Vor dem Start die Konfiguration an Modelle und Hardware anpassen, dann:

```bash
sudo -u eliezer /usr/local/bin/eliezer --config /etc/eliezer/config.toml --check-config
sudo systemctl daemon-reload
sudo systemctl enable --now eliezer.service
systemctl status eliezer.service
journalctl -u eliezer.service
```

Die Unit erzeugt `/var/lib/eliezer` und `/run/eliezer` mit passenden Besitzrechten.
Schreibzugriffe sind auf die vorgesehenen Verzeichnisse begrenzt. Bei einem
Modellverzeichnis außerhalb von `/var/lib/eliezer` dessen Schreibfreigabe mit
`ReadWritePaths` ergänzen. Liegt es unter einem Home-Verzeichnis, muss auch
`ProtectHome` passend angepasst werden.

## Container: Modellverzeichnis als Bind-Mount

Die mitgelieferte [Containerkonfiguration](packaging/config.container.toml)
startet das Gerüst ohne Modelle und ohne GPU. Sie wird schreibgeschützt eingebunden.
Der Port ist auf dem Host standardmäßig nur unter `127.0.0.1:1234` erreichbar.

```bash
mkdir -p models
docker compose build
docker compose up -d
curl --fail http://127.0.0.1:1234/healthz
docker compose logs eliezer
docker compose down
```

Der Container läuft standardmäßig mit UID/GID `10001:10001`. Das Host-Modellverzeichnis
muss für diese Containeridentität lesbar und für Downloads beschreibbar sein.
Die tatsächliche Host-Zuordnung hängt davon ab, ob die Containerengine rootlos läuft.
Ein existierender Modellbestand sollte vorab passend berechtigt werden.

Konfigurierbar sind:

| Variable | Bedeutung | Vorgabe |
|---|---|---|
| `ELIEZER_CONFIG_FILE` | Host-Datei für die Konfiguration | `./packaging/config.container.toml` |
| `ELIEZER_MODELS_DIR` | Host-Verzeichnis für Modelle | `./models` |
| `ELIEZER_PORT` | Auf `127.0.0.1` veröffentlichter Host-Port | `1234` |
| `ELIEZER_UID`, `ELIEZER_GID` | Identität des Containerprozesses | jeweils `10001` |

Bei einer anderen UID/GID müssen auch Volume- und Laufzeitverzeichnisrechte dazu
passen. Für die mitgelieferte rootlose Podman-Variante die Vorgabe `10001:10001`
beibehalten; deren Benutzerzuordnung bildet den aufrufenden Benutzer darauf ab.

Für rootloses Podman die Benutzerzuordnung ergänzen:

```bash
podman-compose -f compose.yaml -f compose.podman.yaml up -d --build
podman-compose -f compose.yaml -f compose.podman.yaml down
```

Diese Variante benötigt für das M2-Gerüst weder GPU-Geräte noch `crun`.
Die Ergänzung deaktiviert beim Podman-Compose-Provider außerdem die automatische
Pod-Erzeugung, damit die Benutzerzuordnung funktioniert.

## Container: persistentes Modell-Volume

Die Ergänzungsdatei ersetzt den Modell-Bind-Mount durch ein benanntes Volume:

```bash
docker compose -f compose.yaml -f compose.volume.yaml up -d --build
docker compose -f compose.yaml -f compose.volume.yaml down
```

Sowohl Modell-Volume als auch Zustands-Volume bleiben bei `down` und bei einer
Neuerstellung des Containers erhalten. `down --volumes` würde diese Daten löschen.
Beim Bind-Mount bleiben die Modelle unabhängig vom Container im Host-Verzeichnis.
Ein neues Modell-Volume enthält zunächst keine Modelle.

## GPU-Zugriff im Container

Die folgenden Ergänzungen reichen Geräte durch. Das M2-Image enthält noch kein
Modellbackend und keine ROCm-/CUDA-Laufzeitbibliotheken; Gerätezugriff allein
aktiviert daher noch keine Inferenz.

Für Docker auf AMD die tatsächlichen Gerätegruppen ermitteln und setzen:

```bash
export ELIEZER_RENDER_GID="$(getent group render | cut -d: -f3)"
export ELIEZER_VIDEO_GID="$(getent group video | cut -d: -f3)"
docker compose -f compose.yaml -f compose.amd.yaml up -d
```

Für GPU-Zugriff unter rootlosem Podman ist die mitgelieferte Ergänzung auf `crun` und dessen
`keep-groups`-Unterstützung ausgelegt. Der Benutzer muss auf dem Host Zugriff
auf `/dev/dri` und für ROCm/HIP auf `/dev/kfd` haben. `crun` muss dafür installiert sein; die Ergänzung wählt es ausdrücklich als Runtime.

```bash
podman-compose -f compose.yaml -f compose.podman.yaml -f compose.amd.podman.yaml up -d --build
# Alternativ zusätzlich mit Modell-Volume:
podman-compose -f compose.yaml -f compose.volume.yaml -f compose.podman.yaml -f compose.amd.podman.yaml up -d --build
```

`compose.amd.yaml` ist die Docker-GPU-Variante, `compose.amd.podman.yaml` die rootlose
Podman-GPU-Variante; sie werden nicht miteinander kombiniert.
Für NVIDIA ist eine passende NVIDIA-Containerintegration erforderlich; eine
NVIDIA-Compose-Variante wurde auf der vorhandenen AMD-Hardware nicht geprüft.

## Token und API-Schlüssel

Geheimnisse gehören weder ins Repository noch in Images, Compose-Dateien oder
Kommandozeilen. Später konfigurierte Token- und API-Schlüsseldateien müssen dem
Dienstbenutzer gehören und Modus `0600` haben. Sie liegen außerhalb der Modellablage
und werden im Container separat schreibgeschützt eingebunden.

Die Authentifizierung und der Token-Bezug sind noch nicht Bestandteil des M2-Gerüsts.
Bis diese Funktionen implementiert sind, die veröffentlichte Adresse auf den
lokalen Rechner begrenzen.

Prüfungen und gemessene Ergebnisse: [tests/README.md](tests/README.md).
