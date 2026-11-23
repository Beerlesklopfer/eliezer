# Eliezer

[English](README.md) · **Deutsch**

**Eine GPU, viele Modellarten, ein Dienst.** Eliezer ist ein Modellverwalter für Linux ohne grafische Oberfläche,
geschrieben in Rust. Er stellt lokale Sprach-, Bild-, Musik-, Sprachsynthese- und Spracherkennungsmodelle hinter
OpenAI- und Anthropic-kompatiblen HTTP-Schnittstellen bereit. Jedes Modell wird geladen, sobald eine Anfrage es braucht,
im Leerlauf wieder entladen, und alle Modelle teilen sich abwechselnd eine einzige Grafikkarte. Entstanden ist Eliezer als
Ersatz für LM Studio im selbst gehosteten Familien-KI-System [MultiGPT](https://github.com/Beerlesklopfer/multi-gpt); er
arbeitet aber mit jedem Client zusammen, der diese Schnittstellen spricht.

Der Name geht auf Eliezer zurück, den Knecht Abrahams, der dessen Haus verwaltete und seine Aufträge ausführte
([warum dieser Name](docs/wiki/ProjectName.md)).

## Das Problem

Ein Heimserver hat meist eine GPU mit 16 oder 24 GB Speicher. Ein Chatmodell, ein Bildmodell, eine Spracherkennung und ein
Stimmmodell passen nicht gleichzeitig hinein, und eine Desktop-Anwendung mit Fenster taugt schlecht als Dienst im Keller.
Eliezer behandelt den GPU-Speicher als Budget: Er weiß, wie viel Speicher jedes Modell braucht, misst, was die Karte
tatsächlich belegt, verdrängt unbenutzte Modelle nach Priorität, stellt Anfragen, die warten müssen, in eine Warteschlange
und verlagert Sprachmodelle auf die CPU, solange ein Bildmodell die GPU belegt.

## Was er bereitstellt

| Art | Endpunkt | Kommandozeile |
|---|---|---|
| Chat, Completion, Embeddings, Bildeingabe | `/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`, `/v1/messages`, `/v1/messages/count_tokens` | `eliezer run MODELL [PROMPT]` |
| Bilder (Erzeugen und Bearbeiten) | `/v1/images/generations`, `/v1/images/edits` | – |
| Musik (Liedtext und Beschreibung) | `/v1/audio/speech` mit einem `music`-Modell | – |
| Sprachsynthese (Stimmklonen, Streaming) | `/v1/audio/speech`, `/v1/audio/voices` | `eliezer speak` (oder `say`), `eliezer voices` |
| Spracherkennung | `/v1/audio/transcriptions` | `eliezer transcribe` |
| Modelle und Speicher | `/v1/models`, `/api/v1/models`, `/api/v1/models/load`, `/api/v1/models/unload`, `/api/v1/models/estimate`, `/api/v1/status`, `/healthz` | `ls`, `ps`, `load`, `unload`, `status`, `show` |
| Modelle von Hugging Face | `/api/v1/hub/*`, `/api/v1/models/download`, `/api/v1/models/pins` | `search`, `inspect`, `get`, `download`, `pins`, `rm` |

Weil Eliezer neben der OpenAI- auch die Anthropic-Messages-API spricht, können Clients wie Claude Code mit lokalen Modellen
arbeiten. Die Verwaltungs-API (`/api/v1`) hat dieselbe Form wie die von LM Studio 0.4, bestehende Anbindungen funktionieren
also weiter. Die vollständige Schnittstelle beschreibt [docs/wiki/API.md](docs/wiki/API.md) (englisch).

## Wie es funktioniert

```
 Clients (MultiGPT, Claude Code, curl, eliezer-Kommandozeile)
        │  HTTP: OpenAI, Anthropic, LM-Studio-Verwaltungs-API
        ▼
 ┌──────────────────────── eliezer (ein Prozess) ────────────────────────┐
 │  axum-Router ──► Supervisor-Task: einziger Besitzer des Zustands      │
 │                  Planer (reine Funktionen): VRAM-Buchführung,         │
 │                  Warteschlange, Verdrängung nach Priorität,           │
 │                  Entladen im Leerlauf, Platzierung auf der CPU        │
 │                     │                         │                       │
 │      llama.cpp, als Crate eingebettet   Kindprozesse, bei Bedarf      │
 │      (Sprache, Bildeingabe,             gestartet, auf Loopback-Ports:│
 │       Embeddings)                       sd-server (Bilder)            │
 │                                         audiocpp_server (Musik, TTS)  │
 │                                         whisper-server (Erkennung)    │
 └───────────────────────────────────────────────────────────────────────┘
        │  GPU-Speicher gemessen über sysfs, rocm-smi, NVML oder nvidia-smi
        ▼
   eine GPU (AMD ROCm, NVIDIA CUDA oder Vulkan) und die CPU mit RAM-Budget
```

- **Ein Besitzer des Zustands.** Nur ein einziger Supervisor-Task ändert den Zustand; die HTTP-Handler schicken ihm
  Nachrichten und warten auf die Antwort. Alle Entscheidungen (was passt, was verdrängt wird, was wartet) liegen in
  `eliezer-core` als reine Funktionen ohne Ein-/Ausgabe. Sie werden deshalb ohne GPU und ohne echtes Warten getestet.
- **Speicherangaben, auf die man sich verlassen kann.** Der Planer berechnet den Bedarf jedes Modells (Gewichte, KV-Cache
  für den gewählten Kontext und die Parallelität, einschließlich Sliding-Window- und linearer Schichten, Bildprojektor) aus
  Architekturangaben, die `eliezer --import` aus dem GGUF-Header liest. Der nach dem Laden gemessene Verbrauch wird
  gespeichert und hat beim nächsten Mal Vorrang. Eine Anfrage, die nicht passt, erhält ein `409` mit der Einstellung, die
  zu ändern ist, statt dass der Karte der Speicher ausgeht.
- **Die passende Engine für jede Art.** Sprachmodelle laufen über das llama.cpp-Crate im Prozess.
  [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp), [audio.cpp](https://github.com/0xShug0/audio.cpp)
  und [whisper.cpp](https://github.com/ggml-org/whisper.cpp) laufen als Kindprozesse. Jeder wird ohne Shell gestartet, mit
  geleerter Umgebung und Argumenten, die allein aus der Konfiguration stammen, und im Leerlauf oder beim Beenden wieder
  gestoppt.
- **GPU-Vorrang.** Solange ein Bildmodell geladen ist oder wartet, weichen Sprachmodelle mit `device = "auto"` auf die CPU
  aus, statt es zu verdrängen, und kehren zurück, sobald die GPU frei ist.
- **Gestreamtes Audio.** Die Sprachsynthese teilt einen langen Text an Satzgrenzen in Abschnitte und liefert WAV, PCM oder
  MP3, während sie liest. Die Wiedergabe beginnt also, bevor der Text fertig ist:

```bash
echo "Guten Morgen" | eliezer say --file - --voice anna | play -t wav -    # vorlesen, während erzeugt wird
eliezer speak --file kapitel.txt --voice anna --format mp3 -o kapitel.mp3  # MP3, unterwegs kodiert
eliezer voices add --name anna --file anna.mp3 --file anna.txt --start 5:00 --length 30
eliezer transcribe ~/Audio --recursive --out ~/Texte --language de         # txt, srt, vtt und json
```

## Abgrenzung

Viele Einzelfunktionen von Eliezer gibt es inzwischen auch anderswo. Projekte mit ähnlichem Anspruch:

| Projekt | Stärke | Unterschied zu Eliezer |
|---|---|---|
| [Lemonade](https://github.com/lemonade-sdk/lemonade) | insgesamt am nächsten: fast alle Modellarten, Live-Abfrage des VRAM auf AMD, Verdrängung nach Schwellwert, ROCm, Debian-Paket | C++; Anthropic-Anfragen werden übersetzt; keine Schätzung pro Modell und keine Warteschlange dokumentiert |
| [LocalAI](https://github.com/mudler/LocalAI) | die größte Breite an Modellarten, auch Musik und Stimmklonen; VRAM-Budget mit Schätzungen | Go, Container; keine Live-Messung über den Treiber dokumentiert |
| [llama-swap](https://github.com/mostlygeek/llama-swap) | das am feinsten einstellbare Wechseln: Kapazitätspools, Verdrängungskosten, Warteschlange mit Priorität | jede Grenze legt der Betreiber fest; keine Schätzung, keine Messung |
| [Ollama](https://github.com/ollama/ollama) | weit verbreitet; Warteschlange, Entladen im Leerlauf, Aufteilung zwischen CPU und GPU beim Laden | eigenes Modellverzeichnis; Bilderzeugung für Linux nicht dokumentiert |

Soweit die Dokumentation dieser und rund zwanzig weiterer Projekte zeigt (Stand November 2026), vereint keines davon,
was Eliezer bietet:

1. Sprachmodelle wandern automatisch auf die CPU, solange ein Bildmodell die GPU belegt, und danach zurück.
2. Eine zu LM Studio kompatible Verwaltungs-API, sodass bestehende Anbindungen weiter funktionieren.
3. Downloads aus einer Allowlist von Repositorys, festgelegt auf einen Commit und per SHA-256 geprüft.
4. Speicherschätzung pro Modell, Live-Messung über den Treiber, Verdrängung nach Priorität und Warteschlange zusammen.
5. In Rust geschrieben, als Debian-Paket mit Dienst, ROCm-Laufzeit mitgeliefert.

„Nicht dokumentiert“ beweist nicht „nicht vorhanden“: Der Vergleich stützt sich auf die Dokumentation der Projekte, nicht
auf ihren Quellcode.

## Gemessen auf echter Hardware

Alle Werte stammen von einem Rechner mit einer AMD Radeon RX 7900 XTX (24 GB, ROCm) und wurden über die Endpunkte des
Dienstes selbst geprüft. Einzelheiten und die Umstände jeder Messung stehen in [docs/wiki/Status.md](docs/wiki/Status.md)
und [docs/Runtimes.md](docs/Runtimes.md) (englisch).

| Aufgabe | Modell | Ergebnis |
|---|---|---|
| Sprachmodell laden, antworten, entladen (installiertes Paket) | Mistral | 1,53 s, 0,37 s, 0,03 s |
| Eine eingescannte Rechnung lesen (Bildeingabe) | olmOCR 2 7B, Qwen3.8 27B, Gemma 4 26B | alle drei lasen Rechnungsnummer und Summe richtig |
| Sprachmodell auf der CPU, während ein Bildmodell die GPU belegt | Mistral | Antwort in 3 s, Bildmodell blieb geladen |
| Ein Bild zeichnen | Qwen Image 2.1 Q4_K_M, 20 Schritte | 13 s bei 512×512, etwa 80 s bei 1024×1024 |
| Ein Lied komponieren | MiniMax Music 3 Q4_0 | 30 s Musik in 2 min 32 s (Modell bereits geladen) |
| Einen Text mit geklonter Stimme vorlesen | Chatterbox Q8_0 | 5 min 45 s Audio in 94 s, erster Ton nach 0,9 s |
| Eine deutsche Aufnahme transkribieren | Whisper large-v3-turbo (deutsch) | 53 min 48 s in 49 bis 53 s, etwa 60-fache Echtzeit |
| Rundweg: Text → Sprache → Text, 1 504 Wörter | Chatterbox + Whisper | Wortfehlerrate 0,8 % bis 1,9 % |

## Sicherheit und Betrieb

- **Allowlist.** Geladen wird nur, was in der Konfiguration steht. Modellnamen in Anfragen sind Schlüssel aus dieser Liste,
  nie Pfade, und kein Wert aus einer Anfrage gelangt ungeprüft in eine Kommandozeile oder eine Shell.
- **Geheimnisse.** API-Schlüssel und das Hugging-Face-Token liegen in Dateien mit Modus 0600 oder in der Umgebung. Sie
  erscheinen nie im Repository, in Logs, in Fehlermeldungen oder in Prozessargumenten, die `ps` zeigt.
  `eliezer --set-hf-token` liest das Token von der Standardeingabe oder einer verdeckten Abfrage, nie aus einem Argument.
- **Kontrollierte Downloads.** Heruntergeladen wird nur aus erlaubten Repositorys, festgelegt auf einen Commit und geprüft
  gegen die SHA-256-Summen, die Hugging Face angibt. Abgebrochene Downloads werden fortgesetzt.
- **Tests ohne Hardware.** Knapp 400 Tests laufen ohne Netz und ohne GPU gegen Ersatz-Backends, die llama.cpp, sd-server,
  audio.cpp und whisper-server nachbilden, einschließlich ihrer Fehler (Absturz beim Laden, nie bereit, Tod nach der
  ersten Antwort). Hardware-Tests sind gekennzeichnet und laufen nur auf Anforderung.

## Ausprobieren

Eliezer gibt es als Debian-Paket (ROCm-Laufzeit und Hilfsprogramme enthalten; etwa 1,05 GB für eine GPU-Architektur,
keine Modelle), als Container-Image mit Compose-Dateien für Docker und Podman oder zum Selbstbauen mit Cargo. Die
Modell-Laufzeit wird beim Bauen gewählt: `rocm`, `cuda`, `vulkan` oder `cpu`. Installation, systemd-Dienst, Container,
die Übernahme von Modellen aus LM Studio und das Einrichten von Schlüsseln beschreibt [INSTALL.MD](INSTALL.MD) (englisch).

```bash
sudo eliezer --config /etc/eliezer/config.toml --import ~/.lmstudio/models --dry-run   # was übernommen würde
sudo eliezer --new-key --name multigpt --save      # ein API-Schlüssel für einen Client, wird einmal angezeigt
eliezer search "qwen3 coder" --task llm --details  # Kandidaten auf Hugging Face mit Passbewertung; lädt nichts
eliezer load qwen/qwen3-coder-30b --context 32768 --parallel 2
eliezer run qwen/qwen3-coder-30b "Erkläre Mutexe in zwei Sätzen" --stats
eliezer ps                                         # geladene Modelle, Gerät, Speicher, Leerlaufzeit
```

`eliezer help` listet alle Befehle; `eliezer BEFEHL --help` zeigt einen einzelnen.

## Stand und Grenzen

**Mittester gesucht:** Eliezer ist bisher nur auf einer AMD-Karte gemessen. Wer eine NVIDIA-Karte, eine andere AMD-Generation
hat oder den Vulkan- bzw. CPU-Build ausprobieren möchte: [TESTING.md](TESTING.md) beschreibt einen halbstündigen Test und wie
man das Ergebnis meldet (englisch).

Version 0.6.0, entwickelt seit Oktober 2026. Alles in den Tabellen oben ist umgesetzt und wurde live geprüft.

- **NVIDIA:** CUDA-Builds und die NVML-Messung sind vorhanden, liefen aber noch nie auf einer echten NVIDIA-Karte. Die
  geprüfte Plattform ist AMD.
- Die Sprachsynthese liest Ziffern (Jahreszahlen, Versangaben) weniger zuverlässig als Wörter; ein Schritt, der Zahlen
  vorher in Wörter umschreibt, ist noch nicht gebaut.
- Live-Spracheingabe als Stream (`/v1/audio/transcriptions/stream`) ist spezifiziert, aber noch nicht gebaut.
- Keine Ziele: Cluster, Benutzerverwaltung, Abrechnung oder eine Weboberfläche. Rechte, Ranking und Abrechnung gehören in
  den Client (MultiGPT); Eliezer führt aus und meldet Fakten.

## Projekt

- Lizenz: [GPL-3.0](LICENSE). Mitgelieferte Fremdkomponenten und ihre Lizenzen: [docs/wiki/Dependencies.md](docs/wiki/Dependencies.md).
- Spezifikation: [docs/Specification.md](docs/Specification.md), Fahrplan mit Meilensteinen: [docs/Implementation.md](docs/Implementation.md),
  Erkenntnisse zu den Fremdprogrammen: [docs/Runtimes.md](docs/Runtimes.md) (alle englisch).
- Eliezer wird mit KI-Programmieragenten entwickelt, die nach schriftlichen Regeln arbeiten ([AGENTS.md](AGENTS.md)):
  kleine Commits, jede Schnittstelle eines Fremdprogramms vor der Verwendung am echten Programm geprüft, Messungen im
  Statusprotokoll festgehalten.
