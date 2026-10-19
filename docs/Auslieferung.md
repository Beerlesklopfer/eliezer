# Auslieferung: deploy.sh und deploy.fish

Auftrag an einen Agenten (Arbeitspaket **M2-05** in [Implementierung.md](Implementierung.md)): ein Skript, das Eliezer auf
dem GPU-Rechner **baut, installiert und neu startet**. Es gibt zwei gleichwertige Fassungen mit demselben Verhalten:
`scripts/deploy.sh` (POSIX-`sh`, getestet mit `dash` und `bash`) und `scripts/deploy.fish` (fish ≥ 3.6). Der Nutzer arbeitet mit fish;
die `sh`-Fassung ist für Rechner ohne fish.

## Verhalten (beide Fassungen)

Aufruf: `scripts/deploy.sh [--host <ziel>] [--features <liste>] [--no-restart] [--dry-run] [--help]`.

1. **Vorprüfung:** Rust-Toolchain (`cargo`, `rustc` ≥ `rust-version` aus `Cargo.toml`), bei `--features` mit `mistralrs-cuda` die
   CUDA-Werkzeuge (`nvcc`), bei `llamacpp` zusätzlich `cmake` und einen C++-Compiler; fehlt etwas, klare deutsche Meldung mit dem
   fehlenden Programm und Exit-Code 2, **ohne** etwas zu verändern.
2. **Bauen:** `cargo build --release --locked -p eliezer --features <liste>` (Standard: `mistralrs`). Die Merkmalsliste kommt aus der
   Befehlszeile oder aus `scripts/deploy.env` (nicht im Repository, nur `deploy.env.example`).
3. **Lokal oder entfernt:** Ohne `--host` wird auf diesem Rechner installiert. Mit `--host` wird das Binary mit `rsync` und
   `ssh` zum Ziel kopiert und dort installiert; Ziel-Anmeldung per ssh-Schlüssel, nie mit Passwort im Skript.
4. **Installieren (idempotent):**
   - Systembenutzer `eliezer` (Gruppen `video`, `render`) anlegen, falls er fehlt.
   - Verzeichnisse `/etc/eliezer` (0750, `root:eliezer`), `/var/lib/eliezer` und `/run/eliezer` (`eliezer:eliezer`).
   - Binary nach `/usr/local/bin/eliezer` (0755), vorher das alte nach `/usr/local/bin/eliezer.prev` sichern.
   - `packaging/eliezer.service` nach `/etc/systemd/system/` und `systemctl daemon-reload`.
   - `/etc/eliezer/config.toml` **nur anlegen, wenn sie fehlt** (aus `packaging/config.example.toml`); eine vorhandene Konfiguration
     und die Dateien `keys` und `hf-token` werden **nie überschrieben oder gelesen/ausgegeben**. Fehlt `hf-token`, nur ein Hinweis
     auf Pfad und Rechte (0600).
5. **Prüfen:** `eliezer --check-config /etc/eliezer/config.toml`; schlägt es fehl, wird das alte Binary wiederhergestellt und der
   Dienst nicht neu gestartet.
6. **Neu starten:** `systemctl restart eliezer`, danach bis zu 30 s auf `GET /healthz` warten (Adresse aus der Konfiguration);
   schlägt das fehl, **Rückfall** auf `eliezer.prev`, Neustart, Meldung und Exit-Code 1.
7. **Ausgabe:** Eine Zeile je Schritt (deutsch), am Ende Version (`eliezer --version`) und `systemctl is-active`. Nie Tokens,
   Schlüssel oder Konfigurationsinhalt in der Ausgabe.

## Regeln

- `--dry-run` zeigt nur, was getan würde, und ändert nichts. `--no-restart` installiert, startet aber nicht neu.
- Keine Annahme über `sudo`: Das Skript prüft `id -u`; als Nutzer ohne Root-Rechte ruft es für die Schritte 4–6 `sudo` auf und
  sagt vorher, welche Befehle das sind. Es fragt nie nach Passwörtern und speichert keine.
- Fehler beenden das Skript sofort (`set -eu` bzw. `status`-Prüfung in fish); Exit-Codes: 0 Erfolg, 1 Fehler beim Betrieb/Rückfall,
  2 Voraussetzung fehlt, 64 falsche Befehlszeile.
- Beide Fassungen werden gemeinsam geprüft: `scripts/test-deploy.sh` führt beide mit `--dry-run` aus und vergleicht die Ausgabe
  Zeile für Zeile (bis auf die Fassungsangabe); `shellcheck` für `.sh` und `fish --no-execute` für `.fish` müssen sauber sein.
- Kein Zugriff auf das Netz außer `cargo` (Crates) und `ssh`/`rsync` zum Ziel.

## Abnahme

`scripts/deploy.sh --dry-run` und `scripts/deploy.fish --dry-run` liefern dieselben Schritte; ein Lauf auf einem Rechner ohne
vorhandene Installation legt Benutzer, Verzeichnisse, Unit und Beispielkonfiguration an, `systemctl is-active eliezer` meldet
`active`, ein zweiter Lauf ändert nichts außer dem Binary; ein absichtlich kaputtes Binary löst den Rückfall aus.
