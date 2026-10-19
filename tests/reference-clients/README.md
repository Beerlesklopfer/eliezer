# Ollama and LM Studio command lines

The installed clients serve as a template for start, load and
unload parameters. Their settings are mapped onto the Eliezer configuration and
the embedded backends. Shell commands from HTTP requests are
not executed.

## Locally verified interfaces

Run on 19.10.2026 without starting a server and without loading a model:

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

Ollama client: 0.18.2. `ollama --version` also reports that no running
Ollama service was reachable. The help of `lms` comes from the existing
user installation under `~/.lmstudio/bin/lms`.

## Mapping of the settings

| Task | LM Studio | Ollama | Eliezer |
|---|---|---|---|
| Load model | `lms load <model-key>` | `ollama run <model>` | Check the model key against the allowlist, then load embedded |
| Context | `--context-length <length>` | `OLLAMA_CONTEXT_LENGTH` | `default_context` or the requested `context_length` |
| Parallelism | `--parallel <count>` | `OLLAMA_NUM_PARALLEL` | Model default `parallel` or the load request |
| GPU offload | `--gpu off`, `--gpu max` or a fraction | Backend/model configuration; no corresponding flag in `ollama run --help` | Backend setting; full offload with `with_n_gpu_layers(999)` verified in the ROCm probe program |
| Idle unload | `--ttl <seconds>` | `OLLAMA_KEEP_ALIVE`, `run --keepalive` | Central setting `idle_unload_secs`; no TTL in the LM-Studio-compatible load body |
| Unload | `lms unload <identifier>` | `ollama stop <model>` | `/api/v1/models/unload` with `instance_id` |
| HTTP address | `lms server start --bind <address> --port <port>` | `OLLAMA_HOST` | `server.listen` |
| Model store | Separately configured model store | `OLLAMA_MODELS` | `storage.models_dir` |
| Queue | No corresponding flag in the verified load help | `OLLAMA_MAX_QUEUE` | `scheduler.queue_max` |
| VRAM reserve | No corresponding flag in the verified load help | `OLLAMA_GPU_OVERHEAD` in bytes | `gpu.reserve_gb`, converted to bytes |

The table describes the functional mapping. Differing semantics, such as
parallelism and context splitting, must be verified in the respective probe
program; identical names do not guarantee identical behavior.

Model-dependent chat templates and parameters can be checked with `ollama show <model>
--template` or `--parameters`. This model-specific check
has not been carried out yet. Personal system prompts are not adopted unchecked
into test data or default configurations.
