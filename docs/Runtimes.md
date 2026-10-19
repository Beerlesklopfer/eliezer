# Laufzeitvoraussetzungen

Zielhardware: mindestens 8 GB VRAM; größere Modelle benötigen entsprechend mehr Speicher.

## Verbindliche Voraussetzungen (Must-haves)

| Bereich | Voraussetzung |
|---|---|
| Betriebssystem | Linux mit funktionierendem GPU-Treiber |
| Grafikkarte | Eine NVIDIA- oder AMD-GPU mit mindestens 8 GB VRAM; Modell, Quantisierung und Kontext passen in das verfügbare Speicherbudget |
| GPU-Laufzeit | Zum gewählten Backend passende Beschleunigung: CUDA für NVIDIA, nachgewiesene ROCm/HIP- oder Vulkan-Unterstützung für AMD |
| Gerätezugriff | Der Dienstbenutzer kann auf die GPU zugreifen; bei AMD auf die benötigten DRM-Geräte und für ROCm/HIP zusätzlich `/dev/kfd`. Gruppen und ACLs müssen diesen Zugriff erlauben |
| Containerbetrieb | Bei Nutzung von Docker/Podman sind GPU-Geräte und benötigte Treiberbibliotheken im Container verfügbar |
| Modellablage | Eigenes, konfigurierbares Verzeichnis für sämtliche Modellbestandteile; im Container als Bind-Mount oder persistentes Volume |
| Modellbestand | Nur konfigurierte Modelle werden geladen; benötigte Tokenizer, Bildprojektoren, Textencoder und VAE sind vollständig vorhanden |
| Speicherverwaltung | Geladene Modelle lassen sich entladen; die Freigabe des VRAM wird gemessen |
| Geheimnisse | Token und API-Schlüssel kommen aus geschützten Dateien (0600) oder der Umgebung und erscheinen nicht in Logs oder Prozessargumenten |

## Rechteverwaltung

Eliezer läuft als eigener unprivilegierter Dienstbenutzer `eliezer`. Bei einer
Benutzerinstallation gelten die folgenden Anforderungen entsprechend für den
ausführenden Benutzer. Der laufende Dienst benötigt weder `root` noch `sudo`.
Die Einrichtung von Benutzer, Gruppen und Systemverzeichnissen übernimmt die Administration.

| Ressource | Erforderliche Rechte |
|---|---|
| GPU-Geräte | Zugriff über die tatsächlichen Gerätegruppen oder passende ACLs. Auf AMD sind insbesondere `render` und gegebenenfalls `video` relevant; ROCm/HIP benötigt zusätzlich Zugriff auf `/dev/kfd` |
| Konfiguration | Für den Dienst lesbar, nur durch die Administration änderbar; alle übergeordneten Verzeichnisse müssen durchsuchbar sein |
| HF-Token und API-Schlüsseldatei | Eigentümer ist der ausführende Dienstbenutzer, Dateimodus `0600`; andere Benutzer erhalten keinen Zugriff |
| Modellverzeichnis | Lesen und Durchsuchen für die Inferenz; Schreiben zusätzlich für Downloads und Cache-Verwaltung |
| Zustandsverzeichnis `/var/lib/eliezer` | Lesen, Schreiben und Durchsuchen für den Dienst, insbesondere für `pins.json` und `measured.json` |
| Laufzeitverzeichnis `/run/eliezer` | Für den Dienst beschreibbar, insbesondere für PID-Dateien; bei Dienststart bereitstellen |
| Backend-Programme und Bibliotheken | Für den Dienst ausführbar beziehungsweise lesbar; nicht durch HTTP-Anfragen austauschbar |

Gruppenmitgliedschaften gelten für den tatsächlich gestarteten Dienstprozess.
Nach einer Änderung muss dieser mit den neuen Gruppen neu gestartet werden.
Die Rechte des angemeldeten Benutzers allein belegen nicht die Rechte eines separat
eingerichteten Dienstbenutzers. Geräte werden nicht pauschal für alle Benutzer freigegeben.

### API-Zugriff und Modellauswahl

- Ist `server.api_keys_file` gesetzt, erfordern die geschützten Endpunkte einen
  konfigurierten API-Schlüssel; `/healthz` bleibt ohne Anmeldung erreichbar.
- Ohne Schlüsseldatei ist die API ohne Anmeldung erreichbar. Die konfigurierte
  Bind-Adresse und die Netzfreigabe bestimmen, wer sie erreichen kann.
- Nur Modelle aus der Positivliste sind ausführbar. Modellschlüssel aus Anfragen
  werden gegen diese Liste geprüft; sie gewähren keinen beliebigen Dateizugriff.
- Benutzerrollen, individuelle Modellberechtigungen und Abrechnung verwaltet MultiGPT.
  Die Eliezer-API-Schlüssel unterscheiden keine Rollen; wer direkten API-Zugriff erhält,
  erhält auch Zugriff auf die geschützten Verwaltungsendpunkte.

### Containerbetrieb

Der Containerprozess erhält die erforderlichen GPU-Geräte und Gerätegruppen.
UID/GID-Zuordnung, zusätzliche Gruppen und gegebenenfalls ACLs müssen auch bei
rootlosem Docker-/Podman-Betrieb den Zugriff auf Geräte und eingebundene Dateien erlauben.
Die Prüfung erfolgt mit der Identität des Containerprozesses.

Konfiguration und Geheimnisdateien werden schreibgeschützt eingebunden.
Modell- und Zustandsvolumes müssen für die Containeridentität passend berechtigt sein;
die Geheimnisdateien müssen dabei ihre restriktiven Zugriffsrechte behalten.
Der Containerbetrieb soll ohne privilegierten Container auskommen.

### Nativer Dienstbetrieb

Die systemd-Unit verwendet `User=eliezer`, die benötigten Gerätegruppen,
`NoNewPrivileges` und `ProtectSystem=strict`. Schreibzugriff wird gezielt auf
Modell-, Zustands- und Laufzeitverzeichnisse begrenzt. Ein abweichendes
Modellverzeichnis muss dabei ausdrücklich berücksichtigt werden.

## Zusätzlich beim Bauen aus Quellcode

- Rust stable ab 1.88 mit Cargo, Edition 2024; rustfmt und Clippy für die Projektprüfungen.
- Die für das gewählte Backend erforderlichen Entwicklungsbibliotheken und GPU-Werkzeuge.
- CMake und ein C/C++-Compiler beim Bau von stable-diffusion.cpp beziehungsweise llama.cpp.

Konkrete Backend-Versionen und unterstützte GPU-Kombinationen werden erst nach einem
bestätigten Bau- und Modelllauf als unterstützt dokumentiert. Eine installierte Bibliothek
oder erfolgreiche Geräteerkennung allein genügt dafür nicht.
