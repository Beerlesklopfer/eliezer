# Spezifikation Eliezer

Stand 2026-10-19. Ziel dieses Dokuments: Ein Agent (Codex) kann daraus Eliezer in Rust bauen, ohne
MultiGPT zu kennen. Alles, was MultiGPT von Eliezer erwartet, steht in Abschnitt 5.

## 1. Zweck

Eliezer läuft als Dienst auf einem Rechner mit **einer** GPU (Ziel: 24 GB VRAM, Linux, **NVIDIA (CUDA) oder AMD (ROCm/HIP, ersatzweise Vulkan)**) und
bedient lokale Modelle:

- Chat- und Vision-Modelle (Sprachmodelle) über mistral.rs,
- Embedding-Modelle (später),
- Bildmodelle über ein austauschbares Hintergrundprogramm.

Er lädt Modelle bei Bedarf, entlädt sie im Leerlauf, hält ein VRAM-Buch, verdrängt nach Priorität
und stellt Anfragen in eine Warteschlange, wenn der Speicher nicht reicht. Nach außen ist er
**kompatibel zu LM Studio** (Abschnitt 5), damit MultiGPT ohne Änderung umschalten kann.

## 2. Begriffe

| Begriff | Bedeutung |
|---|---|
| Modell | Eintrag der Positivliste (`[[model]]` in der Konfiguration), Schlüssel `key` |
| Instanz | geladenes Modell, `instance_id` (hier gleich `key`, es gibt höchstens eine Instanz je Modell) |
| Backend | Hintergrundprogramm, das Modelle bedient (`mistralrs`, `sdcpp`) |
| Budget | nutzbarer VRAM = `vram_budget_gb − reserve_gb` |
| Lease | Reservierung von VRAM für eine Instanz oder einen Ladevorgang |
| Priorität | Zahl je Modell, höher verdrängt niedriger nicht-genutzte Modelle |

## 3. Aufbau

Cargo-Workspace:

```
eliezer/
├─ Cargo.toml                 (workspace)
├─ crates/
│  ├─ eliezer-core/           Typen, Konfiguration, VRAM-Rechnung, Zustandsautomat, Planer (ohne I/O)
│  ├─ eliezer-backend/        Backend-Trait, mistral.rs-Steuerung, sd.cpp-Steuerung, fake-backend
│  ├─ eliezer-api/            axum-Router: /v1/*, /api/v1/*, Authentifizierung, Fehlerformat
│  ├─ eliezer-gpu/            VRAM-Messung (NVML oder nvidia-smi), Trait mit Prüfstand
│  └─ eliezer/                Programm: Argumente, Konfiguration laden, Dienst starten
├─ docs/  (diese Dokumente)
└─ tests/ (Integrationstests gegen fake-backend)
```

Prinzip: `eliezer-core` enthält die gesamte Entscheidungslogik als **reine Funktionen/Zustandsautomaten**
(Eingabe: Zustand + Ereignis, Ausgabe: neuer Zustand + Aktionen). Alles mit Seiteneffekten
(Prozesse, HTTP, Uhr, GPU) steckt hinter Traits und wird in Tests ersetzt. Die Uhr ist ein Trait
(`Clock`), damit Leerlaufzeiten ohne Warten testbar sind.

Laufzeitmodell: ein Prozess `eliezer`, darin ein Verwalter-Task (`Supervisor`), der **allein**
Zustand ändert (Nachrichten über `tokio::sync::mpsc`). HTTP-Handler schicken Befehle und warten auf
Antworten (`oneshot`). Je Backend-Instanz ein Kindprozess.

## 4. Konfiguration

Datei `/etc/eliezer/config.toml` (Pfad per `--config` oder `ELIEZER_CONFIG`). Keine Geheimnisse
darin. Beispiel (vollständig, alle Felder erklärt):

```toml
[server]
listen = "127.0.0.1:1234"          # LM Studio nutzt 1234; im Heimnetz an die LAN-Adresse binden
api_keys_file = "/etc/eliezer/keys" # optional; eine Zeile je Schlüssel, 0600. Fehlt sie: keine Anmeldung
request_timeout_secs = 600

[gpu]
vendor = "nvidia"                   # "nvidia" | "amd" | "none" (nur Buchführung, ohne Messung)
index = 0
vram_budget_gb = 24.0               # Gesamtspeicher der Karte (Dezimal-GB, 10^9 Byte)
reserve_gb = 2.0                    # für Anzeige/System
measure = "nvml"                    # nvidia: "nvml" | "nvidia-smi"; amd: "sysfs" | "rocm-smi"; sonst "none"
# amd + sysfs: liest /sys/class/drm/card<index>/device/mem_info_vram_total und mem_info_vram_used (Byte, amdgpu-Kerneltreiber)

[scheduler]
idle_unload_secs = 1800             # Leerlauf-Entladen (30 min)
queue_max = 64
queue_wait_secs = 120               # danach Antwort 503 mit Hinweis

[huggingface]
token_file = "/etc/eliezer/hf-token"   # 0600; oder Umgebungsvariable HF_TOKEN
cache_dir = "/var/lib/eliezer/hf"      # wird als HF_HOME an Backends gegeben
cache_limit_gb = 200.0
downloads = "on_request"            # "on_request" | "never"

[backend.mistralrs]
program = "/usr/local/bin/mistralrs"   # Aufruf wird in M1 geprüft (docs/Laufzeiten.md)
mode = "process"                       # "process" (ein Prozess je Modell); Bibliothek ist später möglich
port_range = [24500, 24599]            # nur 127.0.0.1

[backend.llamacpp]                     # optional: Ersatz-Laufzeit für Sprachmodelle, falls mistral.rs die GPU des Rechners nicht unterstützt
program = "/usr/local/bin/llama-server"   # Router-Modus oder ein Prozess je Modell; wird in M1 geprüft
port_range = [24550, 24559]

[backend.sdcpp]
program = "/usr/local/bin/sd-server"
port_range = [24600, 24609]

[[model]]
key = "qwen/qwen3-coder-30b"           # Name, den MultiGPT verwendet
backend = "mistralrs"
source = { hf = "Qwen/Qwen3-Coder-30B-A3B-Instruct", revision = "main" }  # oder { file = "/pfad/modell.gguf" }
kind = "llm"                           # "llm" | "embedding" | "image"
priority = 10
max_context = 131072
default_context = 32768
parallel = 1                           # Vorgabe; je Ladeanforderung überschreibbar
vision = false
reasoning = { options = ["none", "low", "medium", "high"], default = "medium" }
arch = { layers = 48, kv_heads = 4, head_dim = 128 }   # für die VRAM-Rechnung, Abschnitt 6
size_gb = 17.3                         # Gewichte auf der GPU (Dateigröße); sonst aus der Datei gelesen

[[model]]
key = "qwen-image"
backend = "sdcpp"
source = { file = "/srv/models/qwen-image-2.1-UC-Q4_K_M.gguf" }
kind = "image"
priority = 5
vram_gb = 14.0                         # gemessener Wert hat immer Vorrang vor der Rechnung
extra = { text_encoder = "/srv/models/qwen2.5-vl.gguf", vae = "/srv/models/vae.safetensors" }
```

Regeln: Unbekannte Felder sind ein Fehler (`#[serde(deny_unknown_fields)]`). `key` ist eindeutig.
Fehler beim Start werden mit Zeile und Feld gemeldet (deutsch) und beenden den Dienst mit Code 2.
Nach `SIGHUP` lädt der Dienst die Datei neu; geladene Modelle bleiben, entfernte werden entladen.

## 5. Schnittstelle (kompatibel zu LM Studio)

Alle Pfade mit `Authorization: Bearer <Schlüssel>`, falls `api_keys_file` gesetzt ist; sonst offen.
Fehlerformat bei jedem Fehler: `{"error": {"message": "<deutscher Text>", "type": "<code>"}}` mit
passendem HTTP-Status. Meldungen enthalten nie Schlüssel, Tokens oder Pfade außerhalb der Konfiguration.

### 5.1 Chat (OpenAI-kompatibel)

- `POST /v1/chat/completions`, `POST /v1/completions`, `GET /v1/models`, `POST /v1/embeddings`.
- Das Feld `model` wählt das Modell (`key`). Ist es nicht geladen, startet Eliezer es (Abschnitt 7),
  die Anfrage wartet in der Warteschlange und läuft dann **unverändert durchgereicht** (Streaming per SSE
  muss Byte für Byte weitergeleitet werden, ohne Puffern).
- `reasoning_effort`: erlaubt sind `none`, `minimal`, `low`, `medium`, `high`, `xhigh`. Ist ein Wert für
  das Modell nicht in `reasoning.options`, antwortet Eliezer mit 400. Fehlt das Feld, gilt
  `reasoning.default` des Modells.
- Antworten mit Streaming müssen `usage` (Prompt- und Ausgabe-Tokens) im letzten Block liefern, wenn
  `stream_options.include_usage` gesetzt ist (MultiGPT berechnet daraus Token/s).
- Werkzeugaufrufe (`tools`, `tool_calls`) werden unverändert weitergegeben; welches Modell sie kann,
  bestimmt das Backend.

### 5.2 Verwaltung (`/api/v1`, Form wie LM Studio 0.4)

`GET /api/v1/models` → 200:

```json
{"models": [{
  "type": "llm",                       // "llm" | "embedding" | "image"
  "key": "qwen/qwen3-coder-30b",
  "display_name": "Qwen3 Coder 30B",
  "architecture": "qwen3moe",
  "params_string": "30B-A3B",
  "size_bytes": 17300000000,
  "max_context_length": 131072,
  "quantization": {"name": "Q4_K_M", "bits_per_weight": 4.8},
  "capabilities": {"vision": false,
                    "reasoning": {"allowed_options": ["none","low","medium","high"], "default": "medium"}},
  "loaded_instances": [{"id": "qwen/qwen3-coder-30b",
                         "config": {"context_length": 32768, "parallel": 1}}]
}]}
```

MultiGPT liest genau diese Felder (`type`, `key`, `display_name`, `architecture`, `params_string`,
`size_bytes`, `max_context_length`, `quantization.bits_per_weight`, `capabilities.vision`,
`capabilities.reasoning.allowed_options|default`, `loaded_instances[].id|config.context_length|config.parallel`).
Fehlende Felder sind erlaubt; falsche Typen nicht.

`POST /api/v1/models/load`, Körper `{"model": "<key>", "context_length": 32768, "parallel": 1,
"echo_load_config": true}` (nur `model` ist Pflicht) → 200 `{"type":"llm","instance_id":"<key>",
"load_time_seconds":12.3,"status":"loaded","load_config":{"context_length":32768,"parallel":1}}`.
Reicht der Speicher nach erlaubtem Verdrängen nicht, antwortet 409 mit `type = "insufficient_memory"`
und einer Meldung „nicht genug Speicher: benötigt X GB, frei Y GB“ (MultiGPT zeigt sie dem Nutzer).
Der Aufruf kehrt erst zurück, wenn das Modell antwortbereit ist (oder scheitert).
Lädt ein Aufruf mit anderer Kontextlänge ein bereits geladenes Modell, wird es neu geladen.
**Kein TTL im Körper:** Das Leerlauf-Entladen macht Eliezer selbst (`idle_unload_secs`), MultiGPT sendet keines.

`POST /api/v1/models/unload`, Körper `{"instance_id": "<key>"}` → 200 `{"instance_id": "<key>"}`;
läuft gerade eine Anfrage, wird sie abgewartet (bis `request_timeout_secs`), dann 409 `busy`.

### 5.3 Erweiterungen (neu, nicht Teil von LM Studio)

- `GET /api/v1/status` → `{"gpu": {"total_bytes", "used_bytes", "free_bytes", "budget_bytes"},
  "models": [{"key", "state": "unloaded|loading|loaded|unloading|failed", "vram_bytes", "last_used_ms_ago",
  "in_flight", "queue_len"}], "queue": [{"model", "waiting_secs"}]}`.
- `GET /healthz` → 200 `ok` (ohne Anmeldung).
- `POST /v1/images/generations` (OpenAI-Form: `prompt`, `n`, `size`, `response_format = "b64_json"`)
  für Modelle mit `kind = "image"`; Antwort `{"created": <unix>, "data": [{"b64_json": "…"}]}`.
- Streaming-Statistik: im letzten SSE-Block der Chat-Antwort zusätzlich `"eliezer": {"ttft_ms", "gen_ms",
  "tokens"}` (MultiGPT misst selbst und ignoriert es; Eliezer dient der Gegenprobe).

## 6. VRAM-Rechnung

Alle Größen in Byte, 1 GB = 10⁹ Byte. Belegung einer Instanz (bewusst konservativ):

```
gesamt = Gewichte + KV-Cache + Zuschlag
KV     = 2 × Schichten × KV-Köpfe × Kopfdimension × 2 Byte (f16) × Kontext
```

- **Gewichte:** `size_gb` bzw. Dateigröße; bei Vision-Modellen inklusive Bildprojektor.
- **Parallelität** vervielfacht den Cache nicht: das eine Kontextfenster wird auf die gleichzeitigen
  Anfragen aufgeteilt.
- **Sliding-Window-Schichten** (`swa_layers`, `window`): nur `window × parallel + 2048` Token, höchstens
  der Kontext. **Gemischte Köpfe** (`full_kv_heads`, `full_head_dim` für volle Schichten). **Lineare
  Schichten** (`linear_layers`): kein wachsender Cache, stattdessen fester `state_bytes` je Sequenz.
  Optionale `arch`-Felder: `swa_layers`, `window`, `full_kv_heads`, `full_head_dim`, `linear_layers`, `state_bytes`.
- **Zuschlag:** 0,5 GB je Sprachmodell; +0,3 GB bei Bild-Eingabe; 0,25 GB bei Embedding-Modellen.
- Ohne `arch`: Näherung nach Parameterzahl `8 KV-Köpfe × 128 Dimensionen`, viele Schichten (z. B. 80).
- **Gemessener Wert hat Vorrang:** `vram_gb` im Modell, später ein im Betrieb gemessener Wert
  (NVML-Differenz vor/nach dem Laden), der gespeichert wird (`/var/lib/eliezer/measured.json`).
- **Bildmodelle:** `vram_gb` ist Pflicht (kein sinnvolles Modell der Architektur).

Beispiel zum Testen: 48 Schichten, 4 KV-Köpfe, Kopfdimension 128, Kontext 32 768 → KV = 2·48·4·128·2·32768
= 3 221 225 472 Byte (≈ 3,22 GB). Mit Gewichten 17,3 GB und Zuschlag 0,5 GB ≈ 21,02 GB.

## 7. Zustandsautomat und Planer

Zustände je Modell: `unloaded → loading → loaded → unloading → unloaded`, Fehlerzustand `failed`
(zurück zu `unloaded` nach `failed_retry_secs`, Standard 30 s). Übergänge nur im Verwalter-Task.

**Ereignisse:** `Request(model, ctx, parallel)`, `LoadDone`, `LoadFailed`, `RequestDone`, `Tick` (jede Sekunde),
`UnloadCmd(model)`, `ConfigReload`.

**Planer (`plan_load`)** – reine Funktion: Eingabe Zustand, gemessener freier Speicher, Anforderung;
Ausgabe `Load`, `LoadAfterUnload(Vec<Key>)`, `Wait` oder `Reject(insufficient_memory)`:

1. Passt das Modell ins freie Budget (`budget − Summe der Leases`) → laden.
2. Sonst Kandidaten zum Entladen: Instanzen mit `in_flight == 0`, die **nicht angeheftet** sind
   (`pinned = true` im Modell, optional) und deren Priorität **kleiner oder gleich** der des Antragstellers ist.
   Reihenfolge: niedrigste Priorität zuerst, dann am längsten ungenutzt. Entladen, bis es passt.
3. Reicht auch das nicht: **nichts entladen, nichts laden**. Hat der Antragsteller einen
   Verdränger, der gerade arbeitet (`in_flight > 0`), dann `Wait` (bis `queue_wait_secs`), sonst `Reject`.
4. Bildmodelle (`kind = "image"`) laufen **exklusiv**: Sie verdrängen alle Sprachmodelle ohne laufende Anfrage;
   danach wird das zuletzt genutzte Sprachmodell nicht automatisch neu geladen (es lädt bei der nächsten Anfrage).

**Warteschlange:** FIFO je Modell, fair über Modelle (Round-Robin), höchstens `queue_max`. Aufträge
**bündeln**: Solange Anfragen für das geladene Modell warten, wird nicht gewechselt (kein Ping-Pong zwischen
Chat und Bild). Eine wartende Anfrage wird abgebrochen, wenn der Client trennt.

**Leerlauf:** `Tick` entlädt Instanzen ohne laufende Anfragen, deren letzte Nutzung länger als `idle_unload_secs`
zurückliegt. Das gilt auch für Modelle, die über `/api/v1/models/load` geladen wurden (kein TTL von außen).

**Wiederherstellung:** Stirbt ein Backend-Prozess, wird die Instanz `failed`; laufende Anfragen enden
mit 502, die nächste Anfrage lädt neu. Beim Start des Dienstes ist nichts geladen; verwaiste Kindprozesse
einer früheren Instanz werden beendet (PID-Datei in `/run/eliezer`).

## 8. Backends

Trait (vereinfacht):

```rust
#[async_trait]
trait Backend: Send + Sync {
    async fn start(&self, spec: &LoadSpec) -> Result<Running, BackendError>; // wartet bis bereit
    async fn stop(&self, running: Running) -> Result<(), BackendError>;      // sanft, dann hart nach 10 s
    fn endpoint(&self, running: &Running) -> Endpoint;                        // 127.0.0.1:Port
    async fn health(&self, running: &Running) -> bool;
}
```

- **mistralrs:** Kindprozess `mistralrs` je Modell, Port aus `port_range`, nur an 127.0.0.1. Der genaue
  Aufruf (Modellart, Quantisierung, Kontext, Parallelität, Chatvorlage) wird in **M1** am installierten
  Programm festgestellt und in `docs/Laufzeiten.md` festgehalten. Eliezer reicht Chat-Anfragen an den
  lokalen Port weiter. Denkoptionen (`reasoning_effort`) setzt Eliezer, soweit das Backend sie nicht selbst
  versteht, in die Chatvorlage bzw. in Parameter um (M4).
- **llamacpp (optional, vor allem für AMD):** Kindprozess `llama-server` je Modell (GGUF), Port aus `port_range`.
  Nur zu bauen, wenn M1 zeigt, dass mistral.rs auf der vorhandenen GPU nicht läuft (Herstellerunterstützung
  von mistral.rs ist **nicht geprüft**: CUDA ist belegt, ROCm/Vulkan nicht). Gleiche Schnittstelle nach außen.
- **sdcpp:** Kindprozess für ein Bildmodell; Anfragen `/v1/images/generations` werden in dessen Schnittstelle
  übersetzt (in M1 am Programm festgestellt). Das GGUF enthält nur den Transformer: Textencoder und VAE
  kommen aus `extra`. Exklusiv (Abschnitt 7, Punkt 4).
- **fake:** Prüfstand: konfigurierbare Ladezeit, VRAM-Bedarf, Antworten, Abstürze; Grundlage aller Tests.
- Umgebung der Kindprozesse: nur `HF_HOME`, `HF_TOKEN` (aus Datei gelesen, nicht aus der Eltern-Umgebung
  durchgereicht), `PATH` und die Gerätewahl des Herstellers (`CUDA_VISIBLE_DEVICES` bzw. `HIP_VISIBLE_DEVICES`/`ROCR_VISIBLE_DEVICES`). Nie Tokens in Argumenten.

## 9. Modellbezug (Hugging Face)

- Bezug nur für Modelle der Liste. `downloads = "on_request"`: Herunterladen beginnt erst, wenn ein
  Verwalter `POST /api/v1/models/download` mit `{"model": "<key>"}` aufruft (Antwort 202; Fortschritt in
  `GET /api/v1/status` als `download: {bytes, total}`), nie automatisch bei einer Chat-Anfrage.
- Revision wird als Commit-Hash festgehalten (`/var/lib/eliezer/pins.json`); Abweichung beim Start wird gemeldet.
- `cache_limit_gb` wird vor dem Herunterladen geprüft (Größe aus der Dateiliste); sonst 507.
- Lizenzpflichtige Modelle: Meldet Hugging Face 401/403, antwortet Eliezer 502 mit dem Hinweis, die
  Lizenz im Konto des Tokens zu bestätigen; der Token erscheint nie in der Meldung.

## 10. Betrieb

- `systemd`-Unit `eliezer.service` (Beispiel in `packaging/`), eigener Benutzer `eliezer`, Gruppen `video`/`render`,
  `NoNewPrivileges`, `ProtectSystem=strict`, `ReadWritePaths=/var/lib/eliezer /run/eliezer`.
- Logs über `tracing` nach stdout (journald). Felder: Zeit, Ebene, Modell, Ereignis, Dauer. **Nie** Inhalte
  von Anfragen oder Antworten, nie Tokens.
- Beenden (`SIGTERM`): laufende Anfragen beenden (bis 30 s), dann Kindprozesse stoppen.
- Metriken (optional, M3): `GET /metrics` im Prometheus-Format (geladene Modelle, VRAM, Wartezeiten).

## 11. Tests

- **Einheit (`eliezer-core`):** VRAM-Rechnung (Beispiel in Abschnitt 6 und je Architekturtyp), Planer
  (alle vier Fälle in Abschnitt 7, Prioritäten, Anheften, Bündeln), Zustandsautomat (jeder Übergang,
  jede Verbotene Kante), Konfigurationsprüfung (unbekannte Felder, doppelte Schlüssel, Zeilenangabe).
- **Integration (`tests/`, mit fake-backend, ohne Netz):** `GET /api/v1/models` entspricht dem Beispiel in
  5.2 (JSON-Schema-Prüfung); Laden/Entladen; Leerlauf mit gestellter Uhr; Anfrage an entladenes Modell
  lädt es und beantwortet sie; zwei Modelle, die nicht zusammen passen → Verdrängung; Wartezeit-Überlauf → 503;
  Trennen des Clients bricht die wartende Anfrage ab; Absturz des Backends → 502, nächste Anfrage lädt neu;
  SSE wird ohne Puffern weitergegeben (Zeitstempel der Teile); `reasoning_effort` ungültig → 400.
- **Vertragstest gegen MultiGPT:** `tests/contract/lmstudio_models.json` enthält eine echte Antwort von LM Studio 0.4
  und eine von Eliezer; beide müssen durch dieselben Prüfungen gehen (Felder und Typen aus 5.2).
- **GPU-Messung (`eliezer-gpu`):** NVIDIA-Anbindung gegen den Prüfstand; AMD-Anbindung gegen ein **nachgebautes
  sysfs-Verzeichnis** in einem temporären Ordner (Dateien `mem_info_vram_total`/`mem_info_vram_used` mit Byte-Zahlen),
  ohne echte Karte.
- **GPU-Tests (`#[ignore]`):** echte Messung (Abschnitt 6, Gegenprobe der Rechnung), Wechsel Chat ↔ Bild.

## 12. Abnahme (gesamt)

1. Mit nur geändertem `base_url` arbeitet MultiGPT mit Eliezer wie mit LM Studio: Modellliste,
   Laden/Entladen mit Kontext und Parallelität, Chat mit Streaming, Denkoptionen, Statistik.
2. Ein Chat-Modell und ein Bildauftrag wechseln sich auf der 24-GB-Karte ab; das Chat-Modell lädt bei der
   nächsten Anfrage von selbst.
3. Nach `idle_unload_secs` ist der Speicher frei (gemessen mit `nvidia-smi`).
4. Kein Token in Logs, Prozessliste oder Fehlermeldungen (Prüfskript `scripts/check-secrets.sh`).
