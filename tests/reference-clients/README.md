# Kommandozeilen von Ollama und LM Studio

Die installierten Clients dienen als Vorlage für Start-, Lade- und
Entladeparameter. Ihre Einstellungen werden auf die Eliezer-Konfiguration und
die eingebetteten Backends abgebildet. Shell-Befehle aus HTTP-Anfragen werden
nicht ausgeführt.

## Lokal geprüfte Schnittstellen

Am 19.10.2026 ohne Start eines Servers und ohne Laden eines Modells ausgeführt:

```bash
ollama --help
ollama serve --help
ollama run --help
ollama show --help
lms --help
lms load --help
lms unload --help
lms server start --help
```

Ollama-Client: 0.18.2. `ollama --version` meldet außerdem, dass kein laufender
Ollama-Dienst erreichbar war. Die Hilfe von `lms` stammt aus der vorhandenen
Benutzerinstallation unter `~/.lmstudio/bin/lms`.

## Übernahme der Einstellungen

| Aufgabe | LM Studio | Ollama | Eliezer |
|---|---|---|---|
| Modell laden | `lms load <model-key>` | `ollama run <model>` | Modellschlüssel gegen Positivliste prüfen, dann eingebettet laden |
| Kontext | `--context-length <length>` | `OLLAMA_CONTEXT_LENGTH` | `default_context` beziehungsweise angeforderte `context_length` |
| Parallelität | `--parallel <count>` | `OLLAMA_NUM_PARALLEL` | Modellvorgabe `parallel` beziehungsweise Ladeanforderung |
| GPU-Auslagerung | `--gpu off`, `--gpu max` oder Anteil | Backend-/Modellkonfiguration; kein entsprechendes Flag in `ollama run --help` | Backend-Einstellung; im ROCm-Probeprogramm vollständige Auslagerung mit `with_n_gpu_layers(999)` geprüft |
| Leerlauf-Entladen | `--ttl <seconds>` | `OLLAMA_KEEP_ALIVE`, `run --keepalive` | Zentrale Einstellung `idle_unload_secs`; kein TTL im LM-Studio-kompatiblen Ladekörper |
| Entladen | `lms unload <identifier>` | `ollama stop <model>` | `/api/v1/models/unload` mit `instance_id` |
| HTTP-Adresse | `lms server start --bind <address> --port <port>` | `OLLAMA_HOST` | `server.listen` |
| Modellablage | Separat konfigurierte Modellablage | `OLLAMA_MODELS` | `storage.models_dir` |
| Warteschlange | Kein entsprechendes Flag in geprüfter Ladehilfe | `OLLAMA_MAX_QUEUE` | `scheduler.queue_max` |
| VRAM-Reserve | Kein entsprechendes Flag in geprüfter Ladehilfe | `OLLAMA_GPU_OVERHEAD` in Byte | `gpu.reserve_gb` mit Umrechnung in Byte |

Die Tabelle beschreibt die fachliche Zuordnung. Abweichende Semantik, etwa
Parallelität und Kontextaufteilung, muss im jeweiligen Probeprogramm geprüft
werden; gleiche Bezeichnungen garantieren kein gleiches Verhalten.

Modellabhängige Chatvorlagen und Parameter können über `ollama show <model>
--template` beziehungsweise `--parameters` geprüft werden. Diese modellbezogene
Prüfung wurde noch nicht durchgeführt. Persönliche Systemprompts werden nicht
ungeprüft in Testdaten oder Standardkonfigurationen übernommen.
