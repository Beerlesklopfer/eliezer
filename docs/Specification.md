# Eliezer Specification

As of 2026-10-19. Purpose of this document: an agent (Codex) can build Eliezer in Rust from it without
knowing MultiGPT. Everything MultiGPT expects from Eliezer is in section 5.

## 1. Purpose

Eliezer runs as a service on a machine with **one** GPU (at least 8 GB VRAM; reference for large models: 24 GB, Linux, **NVIDIA (CUDA) or AMD (ROCm/HIP, alternatively Vulkan)**) and
serves local models:

- chat and vision models (language models) via mistral.rs,
- embedding models (later),
- image models via a replaceable backend program.

It loads models on demand, unloads them when idle, keeps a VRAM ledger, evicts by priority
and queues requests when memory is insufficient. Externally it is
**compatible with LM Studio** (section 5), so that MultiGPT can switch without changes, and besides the
OpenAI interface it also speaks the **Anthropic interface (Messages API)**, so that clients such as Claude Code can work against local models.

## 2. Terms

| Term | Meaning |
|---|---|
| Model | Entry of the allowlist (`[[model]]` in the configuration), identified by `key` |
| Instance | loaded model, `instance_id` (here equal to `key`; there is at most one instance per model) |
| Backend | Runtime that serves models: embedded (`mistralrs`, `llamacpp`) or as a child process (`sdcpp`) |
| Budget | usable VRAM = `vram_budget_gb − reserve_gb` |
| Lease | Reservation of VRAM for an instance or a load operation |
| Priority | Number per model; a higher one evicts lower, unused models |

## 3. Structure

Cargo workspace:

```
eliezer/
├─ Cargo.toml                 (workspace)
├─ crates/
│  ├─ eliezer-core/           types, configuration, VRAM calculation, state machine, planner (no I/O)
│  ├─ eliezer-backend/        backend trait; mistral.rs embedded (feature `mistralrs`), llama.cpp embedded (feature `llamacpp`), sd.cpp process control, fake-backend
│  ├─ eliezer-api/            axum router: /v1/*, /api/v1/*, authentication, error format
│  ├─ eliezer-gpu/            VRAM measurement (NVML or nvidia-smi), trait with test bench
│  └─ eliezer/                program: arguments, load configuration, start service
├─ docs/  (these documents)
└─ tests/ (integration tests against fake-backend)
```

Principle: `eliezer-core` contains all decision logic as **pure functions/state machines**
(input: state + event, output: new state + actions). Everything with side effects
(processes, HTTP, clock, GPU) sits behind traits and is replaced in tests. The clock is a trait
(`Clock`), so that idle times can be tested without waiting.

Runtime model: one process `eliezer`, containing a supervisor task (`Supervisor`) that **alone**
changes state (messages via `tokio::sync::mpsc`). HTTP handlers send commands and wait for
replies (`oneshot`). Per instance: for `mistralrs` and `llamacpp` a handle in the process, for `sdcpp` a child process.

## 4. Configuration

File `/etc/eliezer/config.toml` (path via `--config` or `ELIEZER_CONFIG`). No secrets
in it. Example (complete, all fields explained):

```toml
[server]
listen = "127.0.0.1:1234"          # LM Studio uses 1234; on a home network, bind to the LAN address
api_keys_file = "/etc/eliezer/keys" # optional; one line per key, 0600. If missing: no authentication
run_user = "eliezer"              # optional; account the service runs as; `--import` hands files to it
run_group = "eliezer"             # optional; default: the primary group of run_user
request_timeout_secs = 600

[storage]
models_dir = "/var/lib/eliezer/models" # shared store for all model components

[gpu]
vendor = "auto"                     # "auto" | "nvidia" | "amd" | "none"
index = 0                           # card number (vendor numbering, see section 6a)
vram_budget_gb = 24.0               # total memory of the card (decimal GB, 10^9 bytes); with measurement only an upper limit
reserve_gb = 2.0                    # for display/system
measure = "auto"                    # "auto" | "nvml" | "nvidia-smi" | "sysfs" | "rocm-smi" | "none"

[huggingface]
search = true                        # allow the search of Hugging Face (public metadata, no token)
allow = []                           # repositories the service may download from, e.g. ["leejet/*", "Org/Repo"]

[cpu]                                # optional; language models may run in system memory (section 7a)
ram_budget_gb = 0.0                  # memory for models on the CPU (decimal GB); 0 = CPU placement is off
reserve_gb = 4.0                     # kept free for the system

[scheduler]
idle_unload_secs = 1800             # unload when idle (30 min)
queue_max = 64
queue_wait_secs = 120               # afterwards answer 503 with a hint

[huggingface]
token_file = "/etc/eliezer/hf-token"   # 0600; or environment variable HF_TOKEN
cache_dir = "/var/lib/eliezer/models/hf"      # passed to backends as HF_HOME
cache_limit_gb = 200.0
downloads = "on_request"            # "on_request" | "never"

[backend.mistralrs]                    # embedded as a Rust crate, no child process and no port of its own
enabled = true                         # not available without the Cargo feature `mistralrs` at build time (see section 8)
max_loaded = 4                         # upper limit of simultaneously loaded models of this backend (in addition to the VRAM ledger)

[backend.llamacpp]                     # optional, embedded via a binding crate (fallback runtime, mainly for AMD)
enabled = false                        # not available without the Cargo feature `llamacpp` at build time
max_loaded = 4

[backend.sdcpp]
program = "/usr/libexec/x86_64-linux-gnu/eliezer/sd-server"
port_range = [24600, 24609]
library_path = ["/usr/lib/x86_64-linux-gnu/eliezer", "/opt/rocm/lib/llvm/lib"]   # optional: libraries of the program (it has no run path)

[backend.audiocpp]                     # optional: music and speech synthesis models through audio.cpp (a child process per loaded model)
program = "/usr/libexec/x86_64-linux-gnu/eliezer/audiocpp_server"
port_range = [24700, 24709]            # must not overlap port_range of sdcpp or the service port
compute = "hip"                        # "hip" (default) | "cuda" | "vulkan" | "cpu": the compute backend of the program
# library_path = [...]                 # as for sdcpp
# Speech synthesis (kind = "tts"); every value is a setting, a model can set its own for the first five (extra.*):
# voices_dir = "/var/lib/eliezer/models/voices"   # the voice samples; default: "voices" inside storage.models_dir
# max_input_chars = 100000             # the longest text of one request (1 to 400000) -> 413 above
# max_passage_chars = 400              # a text is cut into passages of at most this many characters (40 to 4000), at sentence boundaries
# pause_ms = 400                       # silence between two passages of a paragraph (0 to 10000)
# paragraph_pause_ms = 800             # silence after the last passage of a paragraph
# voice_max_file_mb = 20               # upload limits of a voice sample: size, length (seconds), name length, the stored sample rate, count
# voice_max_seconds = 120
# voice_name_max_chars = 64
# voice_sample_rate = 24000
# max_voices = 100

[backend.whispercpp]                   # optional: speech recognition through whisper.cpp (a child process per loaded model)
program = "/usr/libexec/x86_64-linux-gnu/eliezer/whisper-server"
port_range = [24800, 24809]            # must not overlap the port ranges of sdcpp or audiocpp, nor the service port
chunk_secs = 600                       # a long recording is cut into pieces of this length (30 to 3600) ...
overlap_secs = 5                       # ... that overlap by this much (0 to 60, less than half a piece)
max_file_mb = 100                      # upload limit; a model can set its own (extra.max_file_mb)
max_audio_minutes = 120                # length limit of a recording; a model can set its own (extra.max_audio_minutes)
split_on_word = true                   # segments end between words; a model can set its own (extra.split_on_word)
max_segment_chars = 0                  # 0 = the program's own wrapping, else 20 to 400 (max_len); extra.max_segment_chars per model
load_timeout_secs = 300                # how long a model may take to load (1 to 3600)
request_timeout_secs = 3600            # how long one piece of a recording may take (1 to 86400)
stop_grace_secs = 15                   # how long the program gets to end after the stop, before it is killed (1 to 300)
# flash_attn = true                    # optional: --flash-attn / --no-flash-attn; without it the program's default applies
# gpu_device = 0                       # optional: --device
# library_path = [...]                 # as for sdcpp
# Live speech input (POST /v1/audio/transcriptions/stream, section 5.3; specified, not built yet). Every value is a setting:
# [backend.whispercpp.live]
# max_streams = 2                      # streams at the same time (1 to 64); more are refused with 429
# max_stream_minutes = 60              # the longest stream (1 to 1440)
# max_pending_secs = 60                # audio that may wait for recognition; above it the stream ends with the error "overloaded" (5 to 600)
# idle_timeout_secs = 30               # no incoming bytes for this long ends the stream with "done" (1 to 3600)
# min_speech_ms = 250                  # speech shorter than this is no phrase (50 to 5000)
# min_silence_ms = 600                 # a pause of this length ends a phrase (100 to 5000); a request may set its own within the range
# preroll_ms = 200                     # audio kept before the start of a phrase (0 to 2000)
# max_segment_secs = 25                # a phrase is cut at the quietest point after this long (5 to 60)
# energy_threshold_db = "auto"         # "auto" follows the noise floor of the stream, or a fixed level (-80 to -10)
# carry_context_chars = 200            # the end of the recognized text goes to the next phrase as its prompt (0 to 1000)
# a model may name a VAD model file of the model store in extra.vad_model (a Silero ggml file); each phrase request then carries vad=true

[[model]]
key = "qwen/qwen3-coder-30b"           # name used by MultiGPT
backend = "mistralrs"
source = { hf = "Qwen/Qwen3-Coder-30B-A3B-Instruct", revision = "main" }  # or { file = "/path/model.gguf" }
kind = "llm"                           # "llm" | "embedding" | "image" | "music" | "stt" | "tts"
priority = 10
max_context = 131072
default_context = 32768
parallel = 1                           # default; can be overridden per load request
vision = false
ocr = false                            # a vision model for text recognition: capabilities.ocr = true in the model list
reasoning = { options = ["none", "low", "medium", "high"], default = "medium" }
arch = { layers = 48, kv_heads = 4, head_dim = 128 }   # for the VRAM calculation, section 6
size_gb = 17.3                         # weights on the GPU (file size); otherwise read from the file

[[model]]
key = "qwen-image"
backend = "sdcpp"
source = { file = "/var/lib/eliezer/models/qwen-image-2.1-UC-Q4_K_M.gguf" }
kind = "image"
priority = 5
vram_gb = 14.0                         # a measured value always takes precedence over the calculation
extra = { text_encoder = "/var/lib/eliezer/models/qwen3vl_8b_int8_convrot.safetensors", vae = "/var/lib/eliezer/models/qwen_image_2.1_vae_bf16.safetensors", steps = 20, cfg_scale = 4.0, sampling_method = "euler", offload_to_cpu = true }
```

A music model (`kind = "music"`, `backend = "audiocpp"`) names the package directory of audio.cpp in `source.file`
(the component files, their configs and the tokenizer; `config.json` must be there) and needs `vram_gb` and the GPU:

```toml
[[model]]
key = "music3"
backend = "audiocpp"
kind = "music"
source = { file = "/var/lib/eliezer/models/audio-cpp/MiniMax-Music3-GGUF" }
vram_gb = 10.0
extra = { duration_sec = 30, steps = 30, cfg_scale = 1.7 }
```

`extra` of a music model: `language_model`, `rvq_depth_decoder` and `flow_transformer` (file names of `.gguf` files inside the
package; default `language_model_q4_0.gguf`, `rvq_depth_decoder_q8_0.gguf`, `transformer_q4_0.gguf`), `duration_sec` (1 to 300,
the length drawn when a request names none; default 30), `steps` (flow matching steps) and `cfg_scale` (flow guidance scale).
`size_bytes` of the model list is the sum of the five component files that are in use.

A speech synthesis model (`kind = "tts"`, `backend = "audiocpp"`) names the GGUF file of an audio.cpp family in `source.file` (download
it with `eliezer download audio-cpp/audio.cpp-gguf Chatterbox-GGUF/chatterbox-q8_0.gguf`, the repository must be on `huggingface.allow`)
and needs `vram_gb` and the GPU. It clones the voice of a sample from the voice directory:

```toml
[[model]]
key = "reader-de"
backend = "audiocpp"
kind = "tts"
source = { file = "/var/lib/eliezer/models/audio-cpp/audio.cpp-gguf/Chatterbox-GGUF/chatterbox-q8_0.gguf" }
vram_gb = 4.0
extra = { family = "chatterbox", language = "de", requires_voice = true, default_voice = "mama", tts_options = { exaggeration = 0.5, temperature = 0.8 } }
```

`extra` of a speech synthesis model: `family` (`chatterbox`, the default, or `cosyvoice3`), `language` (a language code such as `de`,
not `auto`; used when a request names none; without it the program's own default, English, applies), `speed` (0.25 to 4, used when a
request names none; the family must support it), `default_voice` (a voice name used when a request names none), `requires_voice` (bool, default false: the model has no speaker of its own, so a request that names no voice and has no `default_voice` is refused with `400 invalid_request` before anything is read, plain or streamed: `The model "KEY" needs a voice: pass "voice" (see GET /v1/audio/voices) or set extra.default_voice in the model's configuration.`, with a hint to add a voice when the library is empty; precedence: request voice, then `default_voice`, then none), `max_input_chars`,
`max_passage_chars`, `pause_ms` and `paragraph_pause_ms` (like the settings of the section, for this model) and `tts_options`
(numbers for the request options of the family, from a list per family: Chatterbox `exaggeration`, `guidance_scale`, `temperature`,
`repetition_penalty`, `min_p`, `top_p`, `flow_guidance_scale`, `max_tokens`, `text_chunk_size`; CosyVoice3 `text_chunk_size`,
`max_tokens`, `min_tokens`, `top_k`, `num_inference_steps`). The families are the ones checked against the source of audio.cpp
(docs/Runtimes.md); a family is added to the list only after its spec was read.

**Voices.** A voice is the file `<voices_dir>/<name>.wav` (mono, 16 bit) and optionally `<name>.txt` (the words spoken in the sample,
which CosyVoice3 needs). `name` is a file name stem: letters and digits of ASCII, `-` and `_`, at most `voice_name_max_chars`. The
directory has mode 0700 and belongs to the service user. The reference recordings are private: no endpoint serves a sample or a
path, nothing of a sample is logged, and the program is given the path of a file of this directory only after the name of a request
was found in the directory's listing (a request never carries a path). The administrator puts the samples there, by hand or through
`POST /v1/audio/voices` (a modifying endpoint, protected by the API keys like the others). **Consent, profiles, labelling of the output and
deleting audiobooks that were made with a voice are not Eliezer's business**: they live in the application that calls it (David); Eliezer only
holds the files that were put there and executes.

A speech model takes the GPU like a picture or a song (one job at a time).

A speech model (`kind = "stt"`, `backend = "whispercpp"`) names the ggml file of whisper.cpp in `source.file` (a Hugging Face
source is not accepted: download the file with `eliezer download ORG/REPO FILE`, the repository must be on the `huggingface.allow`
list, and point `source.file` at it) and needs `vram_gb`. Unlike pictures and music it may run on the CPU:

```toml
[[model]]
key = "whisper-de"
backend = "whispercpp"
kind = "stt"
source = { file = "/var/lib/eliezer/models/cstr/whisper-large-v3-turbo-german-ggml/ggml-model-q5_0.bin" }
vram_gb = 2.0
device = "auto"                        # "auto" and "gpu": the GPU slot, one job at a time; "cpu": no GPU, RAM budget
extra = { language = "de", threads = 8, max_file_mb = 200, max_audio_minutes = 180 }
```

`extra` of a speech model: `language` (a lowercase language code such as `de`, or `auto` to detect it; the language used when a
request names none; default `auto`), `threads` (1 to 256; the program's default without), `max_file_mb` and `max_audio_minutes`
(the limits of an upload and of a recording; default: the values of `[backend.whispercpp]`, which default to 100 and 120),
`split_on_word` (default `true`: segments end between words; with `false` the program may cut a word in two at its 60 character
wrap) and `max_segment_chars` (0 = the program's own wrapping, else 20 to 400, sent as `max_len`); both default to the values of the
section. The limits, the length of a piece and the overlap are always settings, never constants. Recommended models (the choice is made by a
listening test with real recordings, counting word errors, names and speed): first
`cstr/whisper-large-v3-turbo-german-ggml`, file `ggml-model-q5_0.bin` (Apache-2.0), for German speech such as sermons; alternatives
`primeline/whisper-large-v3-turbo-german` (convert to ggml, or use a ggml build of it) and `openai/whisper-large-v3-turbo`
(multilingual, also available as ggml in `ggerganov/whisper.cpp`).

`extra` of an image model: `text_encoder`, `vae` and `tokenizer` (paths; `tokenizer` is the external `tokenizer.json` that MOSS-Image needs, passed as `--tokenizer`), `steps` (1 to 150), `cfg_scale` (0 to 30),
`sampling_method` (one of the methods of stable-diffusion.cpp), `offload_to_cpu` (default true) and `prompt_helper` (the
key of a configured language model, see below); all optional, the program's own defaults apply to what is left out.

**Prompt helper.** Short wishes ("a sunflower") give image models uneven results. With `extra.prompt_helper = "<key>"` the
service lets that language model write one detailed English description of at most 80 words (subject, setting, lighting,
mood, style) and draws from it. The helper runs on the CPU when `[cpu] ram_budget_gb` is set, so the image model keeps the
GPU; it is an aid only: if it fails, answers nothing, or answers with a refusal, a question or a talk instead of a description
(the stock models refuse some motives, in the language of the wish), the original prompt is used. `prompt_helper` in the answer
is `used`, `off`, `failed` or `declined`. Which model is the helper is the administrator's choice. A request can switch it off with
`"enhance_prompt": false` (`true` without a configured helper is 400). The answer carries `prompt_enhanced` and, per image,
`revised_prompt`. A language model such as Qwen3-VL-8B-Instruct (the family of the Qwen Image text encoder) works well.

Rules: unknown fields are an error (`#[serde(deny_unknown_fields)]`). `key` is unique.
Errors at startup are reported with line and field (in English) and terminate the service with code 2.
On `SIGHUP` the service reloads the file; loaded models stay, removed ones are unloaded.

## 5. Interface (compatible with LM Studio)

All paths except `/healthz` need `Authorization: Bearer <key>` if `api_keys_file` is set (the Anthropic
dialect may use `x-api-key` instead); otherwise the API is open. A key is an OpenAI-style token
(`sk-` and 64 hex digits); several keys can exist. A missing or wrong key gets `401` with
`{"error": {"message": ..., "type": "authentication_error", "code": "invalid_api_key"}}` and
`WWW-Authenticate: Bearer`; the text never repeats the key that was sent, and the comparison takes the
same time whichever byte differs. Keys are made with `eliezer --new-key`, listed (masked) with
`--list-keys` and removed with `--revoke-key NAME`; the Hugging Face token is stored with
`--set-hf-token` (never an argument; source in this order: standard input, `~/.config/eliezer/hf_key`, the default file of the Hugging Face tools `~/.cache/huggingface/token` (`HF_TOKEN_PATH`, `HF_HOME` honored), a prompt without echo). Files: `server.api_keys_file` and
`huggingface.token_file` (mode 0600, owned by `server.run_user`); the user's own copies are
`~/.config/eliezer/api_key` (read by the command line client) and `~/.config/eliezer/hf_key`.
Error format for every error: `{"error": {"message": "<English text>", "type": "<code>"}}` with
the matching HTTP status. Messages never contain keys, tokens or paths outside the configuration.

### 5.1 Chat (OpenAI-compatible)

- `POST /v1/chat/completions`, `POST /v1/completions`, `GET /v1/models`, `POST /v1/embeddings`.
- The field `model` selects the model (`key`). If it is not loaded, Eliezer starts it (section 7),
  the request waits in the queue and is then **passed through unchanged** (streaming via SSE
  must be forwarded byte for byte, without buffering).
- `reasoning_effort`: allowed values are `none`, `minimal`, `low`, `medium`, `high`, `xhigh`. If a value is not in
  the model's `reasoning.options`, Eliezer answers with 400. If the field is missing, the model's
  `reasoning.default` applies.
- Streaming responses must deliver `usage` (prompt and output tokens) in the last chunk if
  `stream_options.include_usage` is set (MultiGPT computes tokens/s from it).
- Tool calls (`tools`, `tool_calls`) are passed through unchanged; which model supports them
  is determined by the backend.
- **Image input** (models with `capabilities.vision`, that is a llamacpp model with `source.projector`):
  content parts `{"type": "image_url", "image_url": {"url": "data:image/png;base64,…"}}` (also `input_image` and the
  Anthropic `image` block). Only base64 `data:` URIs are accepted; a link is refused with 400 and never fetched.
  At most 8 images per request, 20 MB each and 40 MB together. A model without a projector answers 400
  "Model does not support image input". An image is encoded to at most 2000 tokens (less when the model's context
  can only be created with a smaller micro-batch). While the images are encoded, the other requests of the same model wait.

### 5.1a Chat (Anthropic-compatible, Messages API)

In addition to 5.1, same models, same planning (section 7), same queue. Eliezer **translates** the request into
the backend's OpenAI form and the response back (module `eliezer-api::anthropic`, pure functions, easy to test).

- Endpoints: `POST /v1/messages`, `POST /v1/messages/count_tokens`. Authentication with `x-api-key: <key>` **or**
  `Authorization: Bearer`; the header `anthropic-version` is accepted and ignored (without it: no error).
- Request: `model` (= `key`), `messages` (roles `user`/`assistant`, content as text or block list with `text`, `image`
  (base64), `tool_use`, `tool_result`), `system` (text or block list), `max_tokens` (required), `temperature`, `top_p`,
  `stop_sequences`, `stream`, `tools` (`name`, `description`, `input_schema`), `tool_choice` (`auto`, `any`, `tool`, `none`),
  `thinking` (`{"type":"enabled","budget_tokens":N}` or `{"type":"disabled"}`).
- Mapping to OpenAI: `system` → system message; `tool_use` → `tool_calls`; `tool_result` → message with role `tool`;
  `input_schema` → `parameters`; `stop_sequences` → `stop`; `thinking` → the model's `reasoning_effort` (`disabled` → `none`;
  `enabled` → closest matching level from `reasoning.options` by `budget_tokens`: up to 1 024 `low`, up to 8 192 `medium`, above that `high`).
  Unknown fields are ignored, not rejected (clients send new things).
- Response without streaming: `{"id":"msg_…","type":"message","role":"assistant","model":"<key>","content":[{"type":"text","text":"…"},
  {"type":"tool_use","id":"toolu_…","name":"…","input":{…}}],"stop_reason":"end_turn|max_tokens|stop_sequence|tool_use",
  "stop_sequence":null,"usage":{"input_tokens":N,"output_tokens":M}}`. Thinking content as a block `{"type":"thinking","thinking":"…"}`
  before the text, if the backend delivers it. The backend's `finish_reason`: `stop`→`end_turn`, `length`→`max_tokens`,
  `tool_calls`→`tool_use`.
- Streaming (SSE) with the events `message_start`, `content_block_start`, `content_block_delta` (`text_delta`,
  `thinking_delta`, `input_json_delta`), `content_block_stop`, `message_delta` (with `stop_reason` and `usage`), `message_stop`;
  `ping` in between. Parts are forwarded **without buffering** (as in 5.1). Tool arguments arrive as partial JSON.
- `count_tokens`: response `{"input_tokens": N}`; counted with the backend's tokenizer if available, otherwise an approximation
  (characters ÷ 3.5, rounded up) and flagged in the header `x-eliezer-estimate: true`.
- Errors in Anthropic format: `{"type":"error","error":{"type":"invalid_request_error|authentication_error|not_found_error|rate_limit_error|api_error|overloaded_error","message":"<English text>"}}`
  with the matching status; queue overload → 529 `overloaded_error`, wait-time overflow → 529 with a hint.
- Not supported (clear 400 message instead of silently ignoring): document blocks (`document`), citations, web search and other
  server-side tools, batches. The model list `GET /v1/models` answers depending on the header (`anthropic-version` present →
  Anthropic form `{"data":[{"type":"model","id":"<key>","display_name":"…","created_at":"…"}],"has_more":false}`, otherwise OpenAI form).

### 5.2 Management (`/api/v1`, form as in LM Studio 0.4)

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

MultiGPT reads exactly these fields (`type`, `key`, `display_name`, `architecture`, `params_string`,
`size_bytes`, `max_context_length`, `quantization.bits_per_weight`, `capabilities.vision`,
`capabilities.reasoning.allowed_options|default`, `loaded_instances[].id|config.context_length|config.parallel`).
Missing fields are allowed; wrong types are not.

`POST /api/v1/models/load`, body `{"model": "<key>", "context_length": 32768, "parallel": 1,
"echo_load_config": true}` (only `model` is required) → 200 `{"type":"llm","instance_id":"<key>",
"load_time_seconds":12.3,"status":"loaded","load_config":{"context_length":32768,"parallel":1}}`.
If memory is insufficient even after permitted eviction, it answers 409 with `type = "insufficient_memory"`
and a message "not enough memory: needs X GB, free Y GB" (MultiGPT shows it to the user).
The call returns only when the model is ready to answer (or fails).
If a call with a different context length loads a model that is already loaded, it is reloaded.
**No TTL in the body:** Eliezer handles idle unloading itself (`idle_unload_secs`); MultiGPT sends none.

`POST /api/v1/models/unload`, body `{"instance_id": "<key>"}` → 200 `{"instance_id": "<key>"}`;
if a request is currently running, it is awaited (up to `request_timeout_secs`), then 409 `busy`.

### 5.3 Extensions (new, not part of LM Studio)

- `GET /api/v1/status` → `{"gpu": {"total_bytes", "used_bytes", "free_bytes", "budget_bytes"},
  "models": [{"key", "state": "unloaded|loading|loaded|unloading|failed", "vram_bytes", "last_used_ms_ago",
  "in_flight", "queue_len"}], "queue": [{"model", "waiting_secs"}]}`.
- `POST /api/v1/models/estimate`, body as for load (`model`, optional `context_length`, `parallel`,
  `device`) → the memory Eliezer calculates: `{"model", "context_length", "parallel", "device",
  "estimate": {"total_bytes", "basis": "calculated|configured|measured", "weights_bytes",  // the whole figure for configured|measured
  "kv_cache_bytes", "overhead_bytes", "vision_bytes"}, "pools": {"gpu": {"budget_bytes", "free_bytes",
  "free_after_eviction_bytes", "fits_budget", "fits_now", "fits_after_eviction", "max_context_length"},
  "cpu": {…}}}`. The `cpu` pool appears only with a RAM budget. `max_context_length` is the largest
  context that fits the pool's budget (null when none does). Clients such as MultiGPT use this instead of
  estimating from the file size; the calculation uses the `arch` of the model (section 6).
- `GET /api/v1/hub/search`, `GET /api/v1/hub/models/{org}/{repo}`: search of Hugging Face for model candidates (section 9.1).
- `GET /healthz` → 200 `ok` (without authentication).
- `POST /v1/images/generations` (OpenAI form: `prompt`, `n`, `size`, `response_format = "b64_json"`)
  for models with `kind = "image"`; response `{"created": <unix>, "data": [{"b64_json": "…"}]}`.
- `POST /v1/images/edits` (OpenAI form, `multipart/form-data`: `model`, `prompt`, `image[]` or `image` with one to four
  PNG, JPEG or WebP files of at most 20 MB, `n`, `size`, `seed`) for image models with `extra.image_edit = true`
  (`capabilities.image_edit` in the model list); same response. The files are checked by their first bytes and size and
  passed to `sd-server` as `POST /v1/images/edits` (reference pictures). `quality`, `background` and `output_format` are
  ignored, a `mask` is refused (400), a picture over 20 MB gives 413.
- `POST /v1/audio/speech` (the shape of the speech API of the model's serving framework: JSON with `model`, `input` = the
  lyrics, `instructions` = the description of the music (genre, mood, vocals, instruments), `duration_sec` (1 to 300),
  `seed`, `response_format = "wav"`) for models with `kind = "music"` (`capabilities.music_generation`); the answer is the WAV
  file (`audio/wav`, 16 bit, stereo; 44.1 kHz as measured with audio.cpp). Lyrics may carry section tags such as `[verse]` and `[chorus]`. `model` can be
  left out when exactly one music or speech synthesis model is configured. The length is a budget: the song can end earlier.
  The same endpoint chooses by the kind of the model: for a model with `kind = "tts"` (`type: "tts"`, `capabilities.speech_synthesis`
  and `voice_cloning`) the JSON is `model`, `input` (the text, at most `max_input_chars`, else `413`), `voice` (a name from
  `GET /v1/audio/voices`; empty or missing is `extra.default_voice`, else the model's own speaker; an unknown or invalid name is `400`; a model with `requires_voice` and no voice is `400` too),
  `response_format` (`wav`, the default, or `pcm`; no encoder for mp3 or flac is in the service, sox or ffmpeg do that), `stream`
  (true or false, default false), and optionally `speed`, `language` (a code) and `seed` (0 to 4294967295). The service cuts the text into passages at sentence boundaries (blank lines separate paragraphs; a
  number or a single letter before a full stop and a lower-case word after a sentence mark do not end a sentence; longer sentences are
  cut at commas, then between words), reads them one after the other with the loaded model, and joins the audio with `pause_ms` (and
  `paragraph_pause_ms` after a paragraph), so a chapter works. The answer is one WAV file. A passage that fails fails the request.
  - `pcm` is raw signed 16 bit little-endian samples (`audio/pcm`, as the `pcm` of OpenAI) without any header; the format is in the
    response headers `X-Audio-Sample-Rate` (the rate of the first passage's WAV from audio.cpp, not an assumed value),
    `X-Audio-Channels` and `X-Audio-Bits` (`16`). Without `stream` the whole body comes in one piece with a `Content-Length`.
  - `stream = true` sends the answer while it is made, with chunked transfer encoding (`Transfer-Encoding: chunked`, no
    `Content-Length`): the first passage is read before the status is sent (so an error in it is an ordinary error answer, for
    example `502`), then its audio goes out at once and the later passages follow as they are read. The pauses (`pause_ms`,
    `paragraph_pause_ms`) are written as silence after the passage they follow. At most two finished chunks wait for a slow reader, so
    the synthesis does not run far ahead of a player. `wav` is then a WAV file of open length: RIFF and data sizes are `0xFFFFFFFF`,
    which sox, ffmpeg and mpv read until the end of the stream. `input` limits, passage size and pauses are the same settings as
    without a stream; the non-streamed answer is unchanged.
  - The sample rate and the channel count are those of the first passage. A later passage with another rate or channel count ends the
    stream with an error instead of being resampled: one model has one output rate, so a change means that the speech program
    misbehaves, and converting would hide that (the non-streamed answer refuses such a mix with `502` as well).
  - A client that closes the connection stops the synthesis, also in the middle of a passage (the request to audio.cpp is dropped), so
    the passages that follow are not read and the GPU is free again. An error in passage N after the first bytes were sent cannot be
    reported (the status is out, a chunked body has no place for an error): it is logged and the service aborts the connection
    without the closing chunk, so the client sees a failed transfer (curl: error 18, `eliezer speak`: exit 1) and not a file that
    looks complete. A client must treat a stream that ends without the closing chunk as failed.
- `GET /v1/audio/voices`: `{"voices": ["mama", "papa"]}`, the names only. `POST /v1/audio/voices` (multipart: `name`, `file` in any
  format the service decodes, optional `transcript`, optional `replace=true`): the file is decoded and stored as a mono 16 bit WAV at
  `voice_sample_rate`; `201` (`200` when replaced), `400` for a bad name or a file that is not audio, `409` when the name exists
  without `replace=true` or `max_voices` is reached, `413` for a file or recording over the limits. `DELETE /v1/audio/voices/{name}`
  removes the sample and its transcript (`404` for an unknown name). Without `[backend.audiocpp]` these are `404`.
- `POST /v1/audio/transcriptions` (speech to text, OpenAI shape, `multipart/form-data`) for models with `kind = "stt"`
  (`type: "stt"` and `capabilities.transcription` in the model list):
  - Fields: `file` (the audio, required), `model` (can be left out when exactly one speech model is configured), `language`
    (`auto` or a lowercase code; default: `extra.language` of the model, else `auto`), `prompt` (a hint with names and terms),
    `response_format` (`json` default, `text`, `verbose_json`, `srt`, `vtt`), `temperature` (0 to 1) and
    `timestamp_granularities[]` (only `segment`; `word` is refused). Other fields (`stream` and the like) are ignored.
  - Answers: `json` is `{"text": "…"}`; `text` is the plain text (`text/plain`); `verbose_json` is `{"task": "transcribe",
    "language", "duration" (seconds of the whole file), "text", "segments": [{"id", "start", "end", "text"}]}`; `srt` and `vtt`
    are subtitle files (`application/x-subrip`, `text/vtt`). All times are seconds from the start of the whole file.
  - Decoding is the service's job and happens in Rust: MP3, M4A/AAC, Ogg Vorbis, FLAC and WAV (any rate, any channel count, 8 to 32
    bit) are decoded to 16 kHz mono 16 bit. ffmpeg is not used. A file that is not audio (or that no decoder of the build reads)
    gives `400`.
  - Limits (settings, section 4): a file over `max_file_mb` or a recording over `max_audio_minutes` gives `413`; the file is
    read in pieces and refused as soon as it passes the limit, and decoding stops at the length limit, so a hostile file cannot
    fill the memory. `404` for an unknown model, and for a request without `model` when no speech model is configured; `400` for a
    model that does not transcribe, for several speech models without `model`, for a missing `file` and for a wrong field value.
  - Long files: the decoded audio is cut into pieces of `chunk_secs` that overlap by `overlap_secs` (the last piece takes a tail
    that would only be overlap). The pieces go to the program one after the other, so only one piece is in flight. The cut lies in the
    middle of the overlap: a segment of the earlier piece is kept when its middle is before the cut, a segment of the later piece
    when its middle is at or after it; so the speech heard twice appears once. Segments without text are dropped; the times get
    the offset of their piece; the `id`s are numbered again.
  - The request is answered when the whole file is done (it can take minutes: the request timeout of the service must allow it).
    There is no job form (`POST` returning a job id) in this version; a client that wants to poll cuts the file itself.
  - Scheduling: the job takes the GPU slot like a music job (one picture, song or transcription at a time; language models then
    run on the CPU, section 7a), unless the model is set to `device = "cpu"`: then it is placed in RAM like a language model on the
    CPU (`[cpu] ram_budget_gb`), starts with `--no-gpu` and does not wait for the GPU.
- `POST /v1/audio/transcriptions/stream` (**live speech input**; **specified on 2026-10-31, not built yet**) for models with `kind = "stt"`: the
  client sends raw PCM while it speaks and receives the text of each phrase as soon as it is recognized. It is the counterpart of
  `POST /v1/audio/speech` with `"stream": true` and `pcm`, so that a small device (a Raspberry Pi with a microphone and a speaker) can talk to
  Eliezer in both directions with raw samples.
  - Request: `Content-Type: audio/pcm` with the parameters `rate` (8000 to 48000, default 16000), `channels` (1 or 2, default 1) and `format`
    (`s16le`, the only one in the first version), either as media type parameters (`audio/pcm; rate=16000; channels=1; format=s16le`) or as query
    parameters of the same names (the query wins). Further query parameters: `model` (can be left out when exactly one `stt` model exists),
    `language`, `prompt`, `min_silence_ms`. The body is a chunked request body (HTTP/1.1 `Transfer-Encoding: chunked`) that ends when the
    client closes it. Another content type is `415`, a wrong parameter `400`, an unknown model `404`, too many streams `429`
    (`max_streams`), a model that does not transcribe `400`; these are ordinary statuses before the first byte.
  - Response: `200` with `Content-Type: application/x-ndjson`, chunked, sent **while the request body is still arriving** (full duplex on one
    HTTP/1.1 connection: the client reads and writes at the same time, and a proxy that buffers bodies breaks it). One JSON object per line:
    `{"type":"segment","id":0,"start":1.2,"end":4.8,"text":"..."}` (seconds since the start of the stream, as in `verbose_json`),
    `{"type":"error","code":"overloaded|too_long|backend","message":"..."}` (the stream ends after it), and at the end
    `{"type":"done","duration":123.4,"segments":17}`. The final `done` line is also sent when the client closed the body or the idle time ran out
    (`idle_timeout_secs`), after the audio that was still waiting has been recognized.
  - Cutting into phrases (no new dependency): an energy detector with hysteresis on 16 kHz mono (stereo is mixed down, other rates are resampled
    with `rubato`, as for uploads). The level that counts as speech follows the noise floor measured in the stream (`energy_threshold_db = "auto"`)
    or is fixed. A phrase starts after `min_speech_ms` of speech, contains `preroll_ms` of the audio before it, and ends after `min_silence_ms` of
    silence, or at `max_segment_secs` (then cut at the quietest point of the last second, so a word is not split in the middle of a vowel).
  - Recognition: each phrase goes to the program as one request (the path of uploads: 16 kHz mono 16 bit WAV, `verbose_json`, `split_on_word`),
    with the last `carry_context_chars` characters of the text recognized so far as its `prompt` (continuity of names and spelling). When the model
    names `extra.vad_model`, each request carries `vad=true`, so the program's own detector drops phrases without speech; this is the guard against
    the hallucinations on silent or noisy stretches that the round trip of 2026-10-31 showed (an insertion of 11 words in one read-back). A phrase
    without text is not sent. The times of a segment are the offset of its phrase plus the time in the program's answer.
  - Scheduling: **a stream does not hold the GPU.** Each phrase takes the GPU slot (section 7a) like a short transcription and gives it back, so other
    jobs run between two phrases; waiting phrases are recognized in order. With `device = "cpu"` the stream runs in RAM and never waits for the GPU.
  - Limits (all settings of `[backend.whispercpp.live]`): `max_streams`, `max_stream_minutes` (the stream ends with `too_long`), `max_pending_secs`
    (when the recognition is slower than the input, audio piles up; above the limit the stream ends with `overloaded`, it never drops audio
    silently), `idle_timeout_secs`, `max_segment_secs`. Permissions are those of the other endpoints (API keys).
  - Privacy: the audio is never written to disk and never logged; the log carries the model, the duration, the number of segments and the end reason.
  - Latency: a phrase appears `min_silence_ms` after the speaker stopped plus the recognition time. **Measured** on the Radeon RX 7900 XTX with the
    German large-v3-turbo q5 model: 4 min 11 s of speech were recognized in 4.1 s and 53 min 48 s in 49 s (about 60 times real time), so a
    phrase of 5 to 10 s needs well under half a second: about 1 s after the end of a sentence. On a CPU with a small model it is several times the
    length of a phrase (**not measured**).
  - **Not verified yet** (to probe when it is built): that the server (axum 0.8 on hyper) really answers while it is still reading the request body
    (if not, WebSocket is the alternative, with the `ws` feature of axum and one more dependency); the energy detector on real recordings (noise,
    music, several speakers, very quiet speech); the memory of a long stream; the behaviour of common reverse proxies.
  - Tests (the bench, no network and no GPU): the fake `whisper-server` of `tests/m8`; synthetic PCM (silence, tones, noise bursts, speech-like
    bursts) for the phrase boundaries (`min_speech_ms`, `min_silence_ms`, `preroll_ms`, the cut at `max_segment_secs`), the adaptive threshold, the carried
    context in the `prompt`, `vad=true` only when `extra.vad_model` is set, back-pressure with a slow fake (`overloaded`, nothing dropped before it),
    `idle_timeout_secs`, `max_stream_minutes`, `429` above `max_streams`, a disconnect that drops the pending phrases, the release of the GPU slot between
    two phrases (another job proceeds in between), the parsing of `audio/pcm` (rates, channels, a wrong format, the query wins), the mix-down and
    the resampling, an error line followed by the end of the stream, and that no audio reaches the log.
- Streaming statistics: in the last SSE chunk of the chat response additionally `"eliezer": {"ttft_ms", "gen_ms",
  "tokens"}` (MultiGPT measures on its own and ignores it; Eliezer serves as a cross-check).

The minimum requirement is **8 GB VRAM**. The concrete model selection, quantization and
context length must fit the available budget; the 24 GB examples are not a
minimum requirement. The model search also takes the 8 GB target class into account.

### 5.4 Command line client

The `eliezer` binary is also a client of the running service (synchronous HTTP/1.1 over
`std::net`, dispatched before the async runtime starts; the flags of the service keep working).

| Command | API | Output |
|---|---|---|
| `ls` | `GET /api/v1/models`, `GET /api/v1/status` | key, type, capabilities (`vision`, `reasoning`), size, max context, state |
| `ps` | `GET /api/v1/status`, `GET /api/v1/models` | loaded models: VRAM, context, parallel, idle, in-flight, queued |
| `status` | `GET /healthz`, `GET /api/v1/status` | reachability, health, GPU memory with measurement method, models, queue |
| `show MODEL` | `GET /api/v1/models`, `GET /api/v1/status` | details of one model and the endpoints serving it |
| `load MODEL [--context N] [--parallel N]` | `POST /api/v1/models/load` | load time and applied configuration |
| `unload [MODEL \| --all]` | `POST /api/v1/models/unload` | unloaded models; without argument the only loaded model |
| `run MODEL [PROMPT]` | `POST /v1/chat/completions` (SSE) | tokens as they arrive; reasoning dimmed (stderr when piped); `--stats` |
| `search [TEXT] [--task T] [--format F] [--license L] [--limit N] [--details]` | `GET /api/v1/hub/search` | candidate repositories on Hugging Face; nothing is downloaded |
| `inspect ORG/REPO` | `GET /api/v1/hub/models/{org}/{repo}` | files, quantizations, backend support and fit of one repository |
| `speak` (alias `say`) `[--model KEY] [--voice NAME] [--language CODE] [--speed F] [--seed N] [--format wav\|pcm\|mp3] [--text TEXT \| --file FILE] [--out PATH\|-] [--replace]` | `GET /api/v1/models`, `POST /v1/audio/speech` (`stream = true`) | the audio, while it is made |
| `voices` (alias `--list-voices`), `voices add [--name NAME] [NAME] [FILE] [--file AUDIO] [--file TEXTFILE] [--start TIME] [--length TIME] [--replace] [--transcript TEXT]`, `voices rm NAME` | `GET`/`POST /v1/audio/voices`, `DELETE /v1/audio/voices/{name}` | the voice names, one per line |
| `transcribe PATH... --out DIR [--model KEY] [--language CODE] [--format txt,srt,vtt,json] [--prompt TEXT] [--prompt-file FILE] [--recursive] [--extensions LIST] [--replace] [--dry-run] [--limit N] [--timeout-secs N]` | `GET /api/v1/models` (only without `--model`), `POST /v1/audio/transcriptions` per file | `<name>.txt/.srt/.vtt/.json` below `DIR`; progress and summary on stderr |
| `listen [--model KEY] [--language CODE] [--prompt TEXT \| --prompt-file FILE] [--rate N] [--channels N] [--min-silence-ms N] [--source -\|FILE] [--json]` | `POST /v1/audio/transcriptions/stream` (specified, not built) |
| `help` | – | overview |

`speak` reads the text from `--text`, `--file` or standard input (the limit `max_input_chars` is the service's: it answers `413`) and
uses `--model` or the only `tts` model of the service (none or several: exit 2). The audio goes to standard output (also with
`--out -`; chunk by chunk as it arrives, nothing else on stdout, messages on stderr) so that `sox`, `ffmpeg` or `play` can take it
over: `eliezer speak --file kapitel.txt | sox -t wav - -t mp3 kapitel.mp3`. Without `--out` and with a terminal as stdout it
refuses (exit 2); `--out -` writes anyway. `--out PATH` writes a regular file as `PATH.part` and renames it when the stream is
complete (a WAV gets its real sizes then), refuses an existing file without `--replace` (exit 2) and removes the `.part` file when the
stream fails or on Ctrl-C; a named pipe (found by its file type) is opened for writing when the service has accepted the request and
is written block by block, without `.part` and rename; a missing path is created as a regular file. A broken pipe (the reader quit,
for example `head -c`) ends quietly with exit 0 after the connection is closed, which stops the synthesis. A stream that ends
without the closing chunk is exit 1. `voices add` uploads the recording as the endpoint expects it (the name is `--name` or the first word; `--file` may be given twice, once for the recording and once for a `.txt` file with the words spoken in it, which becomes the transcript; `--start` and `--length` (seconds, `MM:SS` or `HH:MM:SS`) cut a part out of a long recording in the command, with the decoder of the service (no ffmpeg), and upload it as a mono WAV at 24 kHz, so the upload limits of the service apply to the part; a service that refuses the upload early, for example with `413`, has its message shown); the library is the
service's (names only, no paths). Eliezer writes `wav`, `pcm` and `mp3`: the service sends wav, and the command encodes `--format mp3` from the stream (LAME through the crate `mp3lame-encoder`, constant 96 kbit/s, no tag, no seek back, so it works on a pipe); flac and other formats are made by sox or ffmpeg.

`transcribe` works through a collection of audio files, one request at a time (the service serves one speech job at
a time). A `PATH` is a file (taken whatever its extension) or a directory, which is searched for the extensions
of `--extensions` (default `mp3,m4a,aac,ogg,oga,opus,flac,wav`; hidden files and directories are ignored, sub-directories
only with `--recursive`, which keeps the relative directory structure below `--out`). `--out` is required; nothing
is written anywhere else. For each file `<stem>` the outputs `<out>/<stem>.<format>` are written, by default all
four. The client asks for `verbose_json` once per file and builds `txt` (one segment per line), `srt`, `vtt` (with
the `WEBVTT` header) and `json` (the answer as received) from it, so a file is transcribed once however many formats
are wanted. The upload is `multipart/form-data` with the file name quoted as browsers do (a comma, a space or an
umlaut is no problem). The model is `--model`, else the only `stt` model of `GET /api/v1/models`; none or several is
a usage error (exit 2). The hint words come from `--prompt` or `--prompt-file`. An existing output is never
overwritten unless `--replace`; a file whose outputs for all requested formats exist is skipped, so a run can be
repeated after an interruption (nothing is requested at all when everything exists). Each output is written as
`<name>.part` and renamed. `--dry-run` lists what would be done and sends nothing; `--limit N` stops after `N`
transcribed files (skipped files do not count). One progress line per file goes to stderr (index/total, name, audio
length, seconds taken, speed factor, outcome), then a summary with the counts, the total audio and wall time and
the failed files; with `--json` a JSON summary goes to stdout. A failing file does not stop the run (exit 1 at the
end); a vanished service (connection refused) stops it after the current file with exit 3. One request waits at
most 6 hours (`--timeout-secs` changes it). Ctrl-C removes the `.part` file, prints the summary so far and exits
with 130.

`listen` (specified on 2026-10-31, not built yet) reads raw PCM (`s16le`, `--rate` and `--channels` as the input has them, default 16000 and 1) from standard
input or from `--source FILE` and sends it while it arrives; each phrase is printed on its own line on standard output as soon as it is recognized
(`--json`: the lines of the service as received), messages go to standard error. The command ends when the input ends (the audio still waiting is
recognized first) with exit code 0; Ctrl-C ends the stream and prints what is pending; the codes are those of the other commands (3 service not
reachable, 1 error line of the service). `arecord -t raw -f S16_LE -r 16000 -c 1 | eliezer listen` listens on a Raspberry Pi; together with
`eliezer speak --format pcm` and a language model it is the loop of a voice assistant:
`arecord ... | eliezer listen | <assistant> | eliezer speak --format pcm | aplay -t raw -f S16_LE -r 24000 -c 1`. Stopping the speech when the
user starts to talk (barge-in) belongs to the assistant, not to Eliezer.

Global options: `--host URL` (else `ELIEZER_HOST`, else `server.listen` of the configuration with
`0.0.0.0` read as `127.0.0.1`, else `http://127.0.0.1:1234`), `--key-file PATH` (else
`ELIEZER_API_KEY_FILE`, `ELIEZER_API_KEY`, else the first key of `server.api_keys_file`), `--json`,
`--config PATH`; the environment variable `ELIEZER_TIMEOUT_SECS` sets how long to wait for an
answer. The key is never a command-line argument and never printed. The client refuses a server
that does not send `Server: eliezer/...`. Exit codes: 0 ok,
1 request failed or server error (the service's message is printed), 2 usage error, 3 service not
reachable. Sizes are decimal GB with one decimal (MB below 0.1 GB).

## 6. VRAM calculation

All sizes in bytes, 1 GB = 10⁹ bytes. Usage of an instance (deliberately conservative):

```
total = weights + KV cache + overhead
KV    = 2 × layers × KV heads × head dimension × 2 bytes (f16) × context
```

- **Weights:** `size_gb` or file size; for vision models including the image projector.
- **Parallelism** does not multiply the cache: the single context window is divided among the concurrent
  requests.
- **Sliding-window layers** (`swa_layers`, `window`): only `window × parallel + 2048` tokens, at most
  the context. **Mixed heads** (`full_kv_heads`, `full_head_dim` for full layers). **Linear
  layers** (`linear_layers`): no growing cache, instead a fixed `state_bytes` per sequence.
  Optional `arch` fields: `swa_layers`, `window`, `full_kv_heads`, `full_head_dim`, `linear_layers`, `state_bytes`.
- **Overhead:** 0.5 GB per language model; 0.25 GB for embedding models. Image input (a llamacpp model with
  `source.projector`, the `mmproj` GGUF file, and `vision = true`) adds `projector_gb`, else 0.3 GB.
  `eliezer --import` and `--suggest` attach the only `mmproj*.gguf` file of a model's directory.
- Without `arch`: approximation by parameter count `8 KV heads × 128 dimensions`, many layers (e.g. 80).
- **A measured value takes precedence:** `vram_gb` in the model, later a value measured in operation
  (difference of the measurement (section 6a) before and after loading), which is stored (`/var/lib/eliezer/measured.json`).
- **Image models:** `vram_gb` is required (no meaningful model of the architecture).

Example for testing: 48 layers, 4 KV heads, head dimension 128, context 32 768 → KV = 2·48·4·128·2·32768
= 3 221 225 472 bytes (≈ 3.22 GB). With weights 17.3 GB and overhead 0.5 GB ≈ 21.02 GB.

## 6a. GPU measurement (uniform for NVIDIA and AMD)

The rest of Eliezer knows **no vendor**. It sees only the trait `GpuProbe` in `eliezer-gpu`:

```rust
trait GpuProbe: Send + Sync {
    fn memory(&self) -> Result<GpuMemory, ProbeError>;   // one measurement, without blocking > 2 s
    fn name(&self) -> &str;                              // e.g. "nvml", "sysfs"
}
struct GpuMemory { total_bytes: u64, used_bytes: u64 }   // free = total − used (computed, never measured)
```

**Uniform meaning:** `used_bytes` is the **total occupied** VRAM of the card, including foreign processes (display,
other programs). Nothing is attributed per process. The unit is always bytes; conversion to GB happens only on
output. A measurement that yields `used > total` is an error (`ProbeError::Inconsistent`).

**Method per vendor** (`measure` makes the choice; `auto` takes the first available row of the table for the vendor):

| Vendor | Order with `auto` | Source |
|---|---|---|
| NVIDIA | 1. `nvml` (library, feature `nvml`), 2. `nvidia-smi` | `nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits --id=<index>` (MiB × 1 048 576) |
| AMD | 1. `sysfs` (amdgpu kernel driver), 2. `rocm-smi` | `/sys/class/drm/card<index>/device/mem_info_vram_total` and `…/mem_info_vram_used` (bytes; kernel documentation "Misc AMDGPU driver information") |
| `none` | – | `total = vram_budget_gb`, `used` = sum of the leases (bookkeeping only) |

**Vendor detection (`vendor = "auto"`):** first read `/sys/class/drm/card<index>/device/vendor` (`0x10de` = NVIDIA,
`0x1002` = AMD); if it is not readable, `none` with a warning in the log. An explicitly set `vendor` takes precedence.

**Common rules:** Each measurement has a time limit of 2 s; `Tick` measures at most once per second and anew on every
load decision. If the chosen source fails, Eliezer falls back **once** to the next row of the table and
reports this (in English, once per change); if all are exhausted, `none` applies and the service stays usable. `GET /api/v1/status`
names the method in use (`gpu.probe`) and `gpu.name`. **The planner always calculates with
`min(budget, total) − max(sum of leases, used − own instances)`** – this accounts for foreign usage without
the method playing a role.

## 7. State machine and planner

States per model: `unloaded → loading → loaded → unloading → unloaded`, error state `failed`
(back to `unloaded` after `failed_retry_secs`, default 30 s). Transitions only in the supervisor task.

**Events:** `Request(model, ctx, parallel)`, `LoadDone`, `LoadFailed`, `RequestDone`, `Tick` (every second),
`UnloadCmd(model)`, `ConfigReload`.

**Planner (`plan_load`)** – pure function: input state, measured free memory, request;
output `Load`, `LoadAfterUnload(Vec<Key>)`, `Wait` or `Reject(insufficient_memory)`:

1. If the model fits into the free budget (`budget − sum of leases`) → load.
2. Otherwise candidates for unloading: instances with `in_flight == 0` that are **not pinned**
   (`pinned = true` in the model, optional) and whose priority is **less than or equal to** that of the requester.
   Order: lowest priority first, then longest unused. Unload until it fits.
3. If even that is not enough: **unload nothing, load nothing**. If the requester has an
   evictor that is currently working (`in_flight > 0`), then `Wait` (up to `queue_wait_secs`), otherwise `Reject`.
4. Image models (`kind = "image"`) run **exclusively**: they evict all language models without a running request;
   afterwards the most recently used language model is not automatically reloaded (it loads on the next request).

**Queue:** FIFO per model, fair across models (round-robin), at most `queue_max`. Jobs are
**batched**: as long as requests for the loaded model are waiting, no switch happens (no ping-pong between
chat and image). A waiting request is aborted if the client disconnects.

**Idle:** `Tick` unloads instances without running requests whose last use is longer than `idle_unload_secs`
ago. This also applies to models loaded via `/api/v1/models/load` (no TTL from outside).

**Recovery:** If a backend process dies, the instance becomes `failed`; running requests end
with 502, the next request reloads. At service start nothing is loaded; orphaned child processes
of an earlier instance are terminated (PID file in `/run/eliezer`).

## 7a. GPU priority and CPU placement

Language models run acceptably on the CPU, diffusion models do not. Therefore the GPU belongs to diffusion
models first. There are two pools of memory: the VRAM budget (`[gpu]`) and, if `[cpu] ram_budget_gb` is above
0, a RAM budget. Each model has `device = "auto" | "gpu" | "cpu"` (default `auto`), and a load request may
override it with the field `device`.

- Image models always take the GPU; `device = "cpu"` for them is a configuration error.
- `gpu` and `cpu` are kept as asked.
- `auto` takes the GPU, **except** while a diffusion model has priority there (one is loading or loaded, or a
  request for one is waiting) and a RAM budget exists: then the language model is placed on the CPU instead of
  evicting the image model or waiting for it. It also falls back to the CPU when it does not fit on the GPU
  even after eviction.
- The planner runs per pool: only the instances of one pool compete for its budget. A model is not moved
  between pools while loaded; it changes pool on the next load.
- Without a RAM budget nothing is placed on the CPU and the behavior is as before.
- A rejection on the CPU reports `409 insufficient_memory` with a hint to raise `[cpu] ram_budget_gb`.
- `GET /api/v1/status` reports `device` per model and `cpu.budget_bytes` / `cpu.used_bytes`;
  `eliezer ps` shows a DEVICE column.

## 8. Backends

Decision from the runtime check: on the AMD card examined, the embedded
backend `llamacpp` is implemented; mistral.rs v0.9.5 and its Candle dependency offer
no ROCm/Vulkan backend. The in-house build is continued. The image service uses
stable-diffusion.cpp as a child process. Checks, measurements and limits are in the
[M1 result](../tests/m1/README.md). The restart path for residual usage remains mandatory.
The minimum Rust version 1.94 required by mistral.rs v0.9.5 must be decided against the project requirement 1.88
before its later adoption.

Trait (simplified):

```rust
#[async_trait]
trait Backend: Send + Sync {
    async fn start(&self, spec: &LoadSpec) -> Result<Running, BackendError>; // waits until ready
    async fn stop(&self, running: Running) -> Result<(), BackendError>;      // gracefully, then forcefully after 10 s
    fn endpoint(&self, running: &Running) -> Endpoint;                        // 127.0.0.1:port
    async fn health(&self, running: &Running) -> bool;
}
```

- **mistralrs (embedded as a crate):** The crate `mistralrs` (Rust SDK of mistral.rs) is **linked into the process**,
  behind the Cargo feature `mistralrs` (on by default) in `eliezer-backend`; version fixed in `Cargo.lock`, dependency in the
  workspace with justification in `docs/Dependencies.md`. There is no child process, no port and no HTTP hop to the model.
  - **Loading:** `LoadSpec` → the crate's model builder (text or vision depending on `kind`/`vision`, quantization, context,
    parallelism, chat template); `start()` waits until the model is ready to answer and returns a handle (`Arc<Model>`).
  - **Unloading:** The handle is dropped; the VRAM must then be **measurably free** (measurement per section 6a,
    wait time at most 10 s). If that does not succeed (crate does not release memory), this is an error `unload_leak`: Eliezer
    reports it and marks the backend as `needs_restart`; the supervisor then terminates the process in an orderly way (systemd restarts it).
    Whether and how the crate unloads/reloads models at runtime is determined in **M1** (`docs/Runtimes.md`).
  - **Requests:** Eliezer builds the request using the crate's types, streams the response piece by piece and converts it into the
    OpenAI form (5.1) or Anthropic form (5.1a). Whether the crate's ready-made OpenAI routes (`mistralrs-server-core`)
    can be embedded instead of building them ourselves is checked in M1; prerequisite: streaming without buffering, `usage`, tool calls.
  - **Isolation:** The computation does not run on the Tokio workers but on dedicated threads (`spawn_blocking` or the
    crate's runtime thread). A **panic** in the backend is caught (`catch_unwind` at the boundary, task join error) and
    makes the instance `failed`; a hard crash (CUDA error, process abort) terminates Eliezer – systemd restarts it,
    running requests end with 502. This is the known price of embedding; the backend `sdcpp`, in contrast, runs as a child process.
  - **Hardware:** Which GPUs the crate supports is determined by its Cargo features at build time (`cuda`, `flash-attn`, `cudnn`,
    `metal`, …). The Eliezer feature `mistralrs-cuda` passes them through. **AMD (ROCm/Vulkan) is not verified** – on AMD
    therefore plan for `llamacpp` first and decide only after M1.
  - **Reasoning options** (`reasoning_effort`) are translated by Eliezer into the chat template or into
    parameters, insofar as the crate does not understand them itself (M4).
- **candle (embedded, fallback stage before llama.cpp):** The original architecture plan names "Hugging Face Candle or
  mistral.rs" as the embedded runtime. mistral.rs builds on Candle; where the mistral.rs crate does not support a card or a model,
  **Candle directly** (pure Rust, no C++) is the next check in M1, **before** `llamacpp` is built. Candle, too,
  is low-level (wire up models, tokenizer, sampler yourself); the gaps count as with `llamacpp`. Whether Candle supports AMD is
  **not verified** and is checked in M1.
- **llamacpp (embedded, optional, mainly for AMD):** llama.cpp is linked via a **binding crate** (candidates:
  `llama-cpp-2`, `llama-cpp-rs`; choice and version to be fixed in M1 and justified in `docs/Dependencies.md`), behind the
  Cargo feature `llamacpp`. There is no `llama-server`, no router, no port. To be built only if M1 shows that the
  mistral.rs crate does not support the available GPU.
  - **Note the effort:** The binding is low-level (load model, create context, tokens, decoding, sampler). The layer
    above it is built by Eliezer itself: apply the chat template (llama.cpp ships `chat template` functions), streaming piece by
    piece, abort, parallelism (several sequences in one context), recognize tool calls in the model output,
    `usage`, vision (image projector, as far as the binding supports it). What the binding cannot do is recorded in `docs/Runtimes.md`
    as a gap and decided on (build it ourselves or leave it out), not silently omitted.
  - **Build prerequisites:** The crate compiles llama.cpp (C++) at build time (CMake, compiler); acceleration is selected by
    Cargo features (`cuda`, `vulkan`, `metal`, …). **Whether ROCm/HIP is available through a feature is not verified** and is
    built and measured in M1; Vulkan is the more likely route for AMD. "All Rust" applies here to Eliezer's code, not to
    the linked llama.cpp.
  - **Loading/unloading, isolation, errors:** as with `mistralrs` (drop the handle, "VRAM free" measurement, `unload_leak`,
    panic catching, dedicated threads). A hard crash in C++ code terminates the process (systemd restarts it).
- **sdcpp (M6):** Child process `sd-server` (stable-diffusion.cpp, `[backend.sdcpp] program`) for each loaded image
  model on a free loopback port of `port_range`. Started without a shell with `--diffusion-model` (the GGUF, which holds
  only the transformer), `--llm` (`extra.text_encoder`), `--vae` (`extra.vae`), `--tokenizer` (`extra.tokenizer`), `--listen-ip 127.0.0.1`, `--listen-port`,
  `--offload-to-cpu` (default; weights stay in RAM and go to the GPU when needed), `--steps`, `--cfg-scale`,
  `--sampling-method` (from `extra`), `--seed -1` (random; the program's default is the fixed seed 42, which made every
  request with the same prompt draw the same picture) and `--log-level warn`. It counts as loaded when `GET /v1/models` answers 200 (the
  load time out is 300 s). A request becomes `POST /v1/images/generations` with `prompt`, `n` (1 to 4), `size` and
  `output_format = "png"` and, when the request has a `seed`, the block `<sd_cpp_extra_args>{"seed":N}</sd_cpp_extra_args>` written by
  Eliezer (the only native setting a request can set); the answer is `data[].b64_json`. The prompt can not carry native settings: the program reads
  `<sd_cpp_extra_args>` blocks inside the prompt, so the name `sd_cpp_extra_args` is removed from every prompt. Unloading
  kills the child. A child that dies marks the instance failed (the next request reloads), requests it cannot answer are
  502. Exclusive on the GPU (section 7, item 4), and language models then run on the CPU (section 7a).
- **audiocpp (M7):** Child process `audiocpp_server` (audio.cpp, `[backend.audiocpp] program`) for each loaded music model on a free
  loopback port of `port_range`. The service writes the program's configuration file into a private directory (mode 0700, file
  0600, removed when the model is unloaded) and starts it without a shell with `--config FILE --no-ui --log`. The file holds the
  loopback address, the port, the compute backend (`compute`), `lazy_load = false` (the weights go to the GPU at start),
  `busy_timeout_ms = 0` (waiting is the service's job) and one model: family `minimax_music3`, task `gen`, the package
  directory, the three component files as session options and `mem_saver = true` (large stages are loaded only while needed).
  The model counts as loaded when `GET /v1/models` lists it with `loaded = true` (load time out 600 s). A request becomes
  `POST /v1/audio/speech` with the description as `input` and the lyrics, `duration_sec`, `seed` and `num_inference_steps` as
  `options`; the answer must be a WAV file. Request text is only ever a JSON string. Exclusive on the GPU like an image model
  (section 7, item 4; language models then run on the CPU, section 7a). A request that fails does not take the model down while
  its program still answers (the same holds for image models): `400` for what the program refuses, `502` for a song it could not
  compose; only a program that no longer answers marks the model failed.
- **audiocpp, speech synthesis:** the same program and the same way of starting it, with another model entry: `id`, the family of
  `extra.family`, `path` = the GGUF file, `task = "tts"`, `mode = "offline"`, `busy_timeout_ms = 0` and the numbers of `extra.tts_options` as
  `default_request_options`. A passage becomes `POST /v1/audio/speech` with `input`, `response_format = "wav"`, `language`, `speed` and
  `seed` when set, and the voice as `voice_ref = {"type": "path", "path": FILE}` (the file of the voice directory) plus `reference_text`
  (the content of `<name>.txt`) when there is one. Text goes into JSON strings only. The answer must be a 16 bit PCM WAV file; all
  passages of a request must have one sample rate and channel count. Interface read from the source, not yet run: docs/Runtimes.md.
- **whispercpp (M8):** Child process `whisper-server` (whisper.cpp, `[backend.whispercpp] program`) for each loaded speech model on a free
  loopback port of `port_range`. The service starts it without a shell, with a cleared environment (the same short list of
  inherited variables as audio.cpp) and a private run directory (mode 0700, working directory of the child and the program's
  `--public` folder, removed when the model is unloaded), with the fixed arguments `--model FILE --host 127.0.0.1 --port N
  --language L --public DIR`, plus `--threads N` (`extra.threads`), `--no-gpu` for a model placed on the CPU, or else `--device N`
  and `--flash-attn`/`--no-flash-attn` (the settings of the section). `--convert` (which starts ffmpeg) is never used. The model counts
  as loaded when `GET /health` answers 200 (the program opens its port only after the model is loaded; load time out `load_timeout_secs`, default 300 s; one piece may take `request_timeout_secs`, default 3600 s; a stopped child gets `stop_grace_secs`, default 15 s, before it is killed). A
  piece becomes `POST /inference`, a `multipart/form-data` body with the 16 kHz mono WAV as `file` and the fields
  `response_format = verbose_json` (the service builds every other format itself from the segments), `language`, `prompt`,
  `temperature`, `no_language_probabilities`, `split_on_word` (setting, default true) and `max_len` (only when `max_segment_chars` is above 0). Request text is only ever a form field. The answer must hold `segments` with
  `start`, `end` (seconds) and `text`. `400` of the program is a request it refused (`400`); other failures and answers
  without segments are `502`; a program that no longer answers marks the model failed. The program serializes requests itself.
- **fake:** Test bench: configurable load time, VRAM demand, responses, crashes; basis of all tests.
- Environment of child processes: only `HF_HOME`, `HF_TOKEN` (read from the file, not passed through from the
  parent environment), `PATH` and the vendor's device selection (`CUDA_VISIBLE_DEVICES` or `HIP_VISIBLE_DEVICES`/`ROCR_VISIBLE_DEVICES`). Never tokens in arguments.

## 9. Model search, model store and model retrieval (Hugging Face)

### 9.1 Targeted model search

Eliezer shall also enable the targeted search for specialized models on Hugging Face.
Search criteria are in particular intended use (e.g. programming, vision, embeddings or
image generation), model format, quantization, license and suitability for the available runtime
and the VRAM budget. Search results are initially candidates: they are not automatically
downloaded, added to the allowlist or executed. Before adoption, the
model card, required additional files and the actual backend support are checked.

**Interface (M5-00).** Read only; nothing is downloaded and the allowlist does not change. The service
reads the public metadata of Hugging Face without a token (`[huggingface] search = false` switches it off).

- `GET /api/v1/hub/search?q=&task=&format=&license=&limit=&details=` → `{"object": "list", "data": [{"repo",
  "downloads", "likes", "task", "license", "gguf", "detail"}], "budgets": {"gpu_bytes", "ram_bytes"}}`.
  `task` is `any|llm|vision|embedding|image` (the `pipeline_tag` of Hugging Face), `format` is `gguf` (default)
  or `safetensors`, `license` a license tag, `limit` 1 to 20 (default 10). `q` or `task` is required, `q` has at
  most 100 characters. With `details=true` the first 10 hits carry `detail` (as below), else it is `null`.
- `GET /api/v1/hub/models/{org}/{repo}` → `{"repo", "sha", "task", "license", "gated", "downloads",
  "architecture", "support": {"level": "verified|expected|unsupported|unknown", "backend"}, "variants":
  [{"file", "quantization", "bytes", "parts", "estimate_bytes", "fit": "gpu|cpu|too_large"}], "projector":
  {"file", "bytes"} | null, "required_files": [{"file", "bytes"}]}`.
  `sha` is the commit of the listing; a later download pins it. `variants` are the weight files (all parts of a
  split file count as one). `estimate_bytes` is the file size plus 20 %, plus the vision projector (language
  model) or the smallest text encoder and VAE (image model): a rough figure, the exact one comes from the GGUF
  header (`eliezer --suggest`). `fit` compares it with the GPU budget, then the RAM budget.
  `support.level` is `verified` for architectures served and checked here, `expected` for architectures a runtime
  lists but that were not checked here, `unsupported` for unknown ones, `unknown` when Hugging Face reports none.
- Errors: 400 `invalid_request` (bad parameters), 403 `hub_disabled`, 404 `not_found`, 502 `hub_unreachable` or
  `hub_bad_response`, 503 `overloaded` (Hugging Face limits requests). A repository name is validated
  (`Organisation/Repository`, letters, digits, `.`, `_`, `-`) before it becomes part of a URL.
- Command line: `eliezer search [TEXT] [--task] [--format] [--license] [--limit] [--details]` and
  `eliezer inspect ORG/REPO`.
- Before adopting a candidate the model card, the required additional files and the actual backend support are
  checked by the administrator; downloading (M5-02) is a separate step.

### 9.2 Shared, persistent model store

All model files reside under a dedicated, configurable model directory
(`[storage].models_dir`, default `/var/lib/eliezer/models`). This includes locally provided
weights, Hugging Face downloads, tokenizers, image projectors, text encoders and VAE.
Separate subdirectories are used per model and pinned revision;
file names and subdirectories are derived from validated configuration.
Model paths must lie within this directory after resolving symbolic links.
`huggingface.cache_dir` also lies within it (default: `/var/lib/eliezer/models/hf`).
Secrets live outside the model store.

In the container, the same path is to be mounted either as a bind mount of a dedicated host directory
or alternatively as a named, persistent Docker/Podman volume.
Replacing the container must not require new model downloads. The storage of
`pins.json` and `measured.json` likewise remains persistent under `/var/lib/eliezer`.

### 9.2a Importing existing models (`eliezer --import`)

Models that another program already downloaded (for example LM Studio under `~/.lmstudio/models`)
are moved into the model store with `eliezer --import DIR`. The command is run by an administrator
(usually with `sudo`) and does the following:

- It collects the regular `*.gguf` files below `DIR` (symbolic links are never followed) and keeps
  the directory structure below `DIR` in `models_dir`. Existing files are never overwritten;
  `DIR` and `models_dir` must not contain each other.
- **Move** is the default: on one file system a rename (instant, no extra space), otherwise a
  verified copy (written as `.partial`, size checked, then renamed) followed by deleting the
  original. `--copy` keeps the originals; `--dry-run` only prints the plan.
- The files and new directories are handed to `server.run_user` / `server.run_group` (mode 0644 and
  0755). Accounts are looked up with `getent`, so local, LDAP and domain accounts work. A user
  alone gives the files its primary group; a group alone changes only the group; with neither set
  the ownership stays as it is (a note is printed).
- **The configuration is updated automatically.** For every language and embedding model it
  appends a `[[model]]` block (key derived from the file name and made unique; `backend = "llamacpp"`;
  `max_context` and `arch` from the GGUF header, which is read without touching the tensor data; a
  `vram_gb` estimate of file size + 20 % when the header has no attention shape; at least 0.1 GB) and
  adds `[backend.llamacpp] enabled = true` when it is missing. Existing text and comments stay as they
  are, files that are already configured (same `source.file`) are skipped, and vision projector
  files (`mmproj-*`) are imported but get no entry. Sliding-window and recurrent models get a hint that
  the memory estimate needs `swa_layers`/`window` or `linear_layers`/`state_bytes`.
- **Image models** (a GGUF with no tokenizer and no transformer layers, such as `qwen_image21`) are
  moved too, but only a *commented* block is printed, because their runtime (stable-diffusion.cpp)
  is not connected yet. An existing entry that points at such a file without being an image model is
  reported as a warning.
- **Safety:** the new text is validated exactly like at service start (syntax, values and paths)
  before anything is written. If it would be invalid, the file stays untouched and the reason is
  printed. The write is atomic (temporary file in the same directory, same mode and owner) and the
  old file is kept as `<name>.bak`. A `[backend.llamacpp]` that exists but is disabled is never
  changed. A broken `[[model]]` list does not block the import: the files are moved either way.
- `--no-config` prints the blocks instead of editing the file; `--dry-run` shows the files and the
  entries that would be added.

Exit code 2 for unusable arguments, configuration or accounts, 1 if a file failed, otherwise 0.

### 9.3 Controlled retrieval

Downloading is an administrative act and never happens on a chat request.

- **Allowlist.** `[huggingface] allow = ["Organisation/Repository", "Organisation/*", "*/Qwen3-VL*"]` names the repositories
  the service may fetch from; in each of the two parts `*` stands for any run of characters. `*/*` (everything) is refused. A
  wildcard in the organisation lets anybody's repository of that name through: write it only where that is meant.
  Empty means nothing can be fetched. `downloads = "never"` switches downloading off
  altogether. A repository on the list is not yet a model: adding the model to the configuration stays a
  separate step (`eliezer --suggest FILE` prints the block).
- `POST /api/v1/models/download`, body `{"repo": "Organisation/Repository", "files": ["a.gguf", "vae/v.safetensors"],
  "revision": "<40 hex digits>"}` (`revision` is optional; without it the current commit is taken and pinned). Answer
  202 with the job `{"id", "repo", "revision", "directory", "state", "total_bytes", "done_bytes", "files": [{"file",
  "total_bytes", "done_bytes", "state", "error"}], "error"}`. `state` is `queued`, `running`, `done` or `failed`; a
  file is `queued`, `running`, `done`, `present` (it was there and matched) or `failed`. One job runs at a time.
- `GET /api/v1/models/download` lists the jobs (newest first), `GET /api/v1/models/download/{id}` shows one,
  and `GET /api/v1/status` carries the latest five as `downloads`.
- **Checks before anything is written:** the repository name and every file name (relative, at most three levels,
  letters, digits and `. _ - +`, nothing hidden, no `..`), the allowlist, that every file exists in the listing of
  that commit, at most 16 files, and that no job of the same repository is waiting (409 `busy`).
  Files go to `<models_dir>/<Organisation>/<Repository>/<file>`; the path is checked to lie inside the store after
  resolving symbolic links, and a link as the target is refused.
- **Integrity.** A file is written as `<file>.part`, continued after an interruption (HTTP range, the hash is
  taken over the whole file), compared with the SHA-256 Hugging Face lists and the listed size, and only then
  renamed. A mismatch discards the file (job `failed`, no file under the final name).
- `cache_limit_gb` counts the pinned files in the store; a download that would exceed it answers 507
  `insufficient_storage` before it starts.
- **Gated repositories:** without a token the answer is 403 `license_required` ("accept the license of the
  repository on huggingface.co and provide a token"); a refusal during the download ends the job the same way.
- **Pins (M5-01).** Every repository directory holds `.eliezer-pin.json` with the commit, size, SHA-256 and time of
  each fetched file. `GET /api/v1/models/pins[?verify=true]` reports `ok`, `missing`, `size_mismatch` or (with
  `verify`, which hashes the files) `modified` for each file and the number of deviations. At start the sizes are
  checked and each deviation is logged (`pin_deviation`); nothing is repaired automatically.
  `DELETE /api/v1/models/pins/{org}/{repo}` (`eliezer rm ORG/REPO`) removes a fetched repository; it refuses
  while a download is active (`409 download_active`) or a configured model uses its files (`409 in_use`).
- **Token (M5-03).** From `huggingface.token_file` (mode 0600; a file others can read is not used and a warning
  says so) or else the environment variable `HF_TOKEN`; `eliezer --set-hf-token` writes the file. It is sent only
  to `huggingface.co`, never forwarded to another host, never an argument of a process (a child process gets it as
  `HF_TOKEN` in its environment) and never part of a message or log line. `scripts/check-secrets.sh` searches
  logs, the journal and the process list for tokens and keys without printing them.
- Command line: `eliezer download ORG/REPO FILE... [--revision COMMIT]` follows the job; `eliezer get ORG/REPO
  [--quant Q4_K_M] [--with FILE,FILE]` looks at the repository, picks the weight file in the best quantization that
  fits (or the named one) with its vision projector, and fetches it at the commit it looked at; `eliezer pins
  [--verify]`.
- A model store under `/home` must be writable for the service: the systemd drop-in with `ReadWritePaths` (see
  `packaging/eliezer-models-in-home.conf.example`); otherwise the job fails with a message saying so.

## 10. Operation

- `systemd` unit `eliezer.service` (example in `packaging/`), dedicated user `eliezer`, groups `video`/`render`,
  `NoNewPrivileges`, `ProtectSystem=strict`, `ReadWritePaths=/var/lib/eliezer /run/eliezer`.
- Logs via `tracing` to stdout (journald). Fields: time, level, model, event, duration. **Never** contents
  of requests or responses, never tokens.
- Shutdown (`SIGTERM`): finish running requests (up to 30 s), then stop child processes.
- Metrics (optional, M3): `GET /metrics` in Prometheus format (loaded models, VRAM, wait times).

### 10.1 Container operation with Compose

In addition to the systemd unit, a Compose configuration is part of the deliverables:
`compose.yaml` for operation with a dedicated model directory as a bind mount and
`compose.volume.yaml` as an alternative with a named model volume. Configuration is mounted
read-only; token and API keys are mounted as separate files with
restrictive permissions, never stored in the image or Compose file.
State data stays persistent, runtime data under `/run/eliezer` is volatile.

Container build instructions, GPU access and start/stop instructions are set up for the available hardware
after the results of M1. Docker Compose and a suitable
Podman Compose setup are checked; differences and prerequisites are documented.
Acceptance covers both storage variants as well as the retention of the models after recreating
the container. The container must make the configured HTTP address reachable and
handle SIGTERM in an orderly way.

## 11. Tests

- **Unit (`eliezer-core`):** VRAM calculation (example in section 6 and per architecture type), planner
  (all four cases in section 7, priorities, pinning, batching), state machine (every transition,
  every forbidden edge), configuration validation (unknown fields, duplicate keys, line information).
- **Integration (`tests/`, with fake-backend, no network):** `GET /api/v1/models` matches the example in
  5.2 (JSON schema check); load/unload; idle with a fake clock; request to an unloaded model
  loads it and answers it; two models that do not fit together → eviction; wait-time overflow → 503;
  client disconnect aborts the waiting request; backend crash → 502, next request reloads;
  SSE is passed through without buffering (timestamps of the parts); invalid `reasoning_effort` → 400.
- **Anthropic translation (unit, no network):** one test per mapping rule from 5.1a (system text, tool call there and back,
  `tool_result`, images, `thinking` → level, `stop_reason`), streaming event order against a fixed expectation file,
  partial JSON of the tool arguments, error format, `count_tokens` with approximation. Integration: the same request via `/v1/chat/completions`
  and `/v1/messages` yields the same content against the fake-backend.
- **Contract test against MultiGPT:** `tests/contract/lmstudio_models.json` contains a real response from LM Studio 0.4
  and one from Eliezer; both must pass the same checks (fields and types from 5.2).
- **GPU measurement (`eliezer-gpu`):** **one** set of tests against the trait `GpuProbe` with all methods (section 6a), without a card:
  `nvidia-smi` and `rocm-smi` against a script in the temporary directory that delivers fixed outputs; `sysfs` against a replica
  directory (`mem_info_vram_total`/`mem_info_vram_used`, `device/vendor`); NVML only as an `#[ignore]` test. Checked are: same
  meaning (bytes, `used ≤ total`), time limit, fallback to the next source, vendor detection, error case `none`.
- **GPU tests (`#[ignore]`):** real measurement (section 6, cross-check of the calculation), switching chat ↔ image.

## 12. Acceptance (overall)

1. With only a changed `base_url`, MultiGPT works with Eliezer as with LM Studio: model list,
   load/unload with context and parallelism, chat with streaming, reasoning options, statistics.
2. A chat model and an image job alternate on the 24 GB card; the chat model loads by itself on the
   next request.
3. After `idle_unload_secs` the memory is free (measured with `nvidia-smi`).
4. An Anthropic client (e.g. Claude Code with `ANTHROPIC_BASE_URL` pointing at Eliezer) conducts a conversation with tool call and streaming.
5. No token in logs, process list or error messages (check script `scripts/check-secrets.sh`).
