# Runtime Requirements

Target hardware: at least 8 GB VRAM; larger models need correspondingly more memory.

## Mandatory requirements (must-haves)

| Area | Requirement |
|---|---|
| Operating system | Linux with a working GPU driver |
| Graphics card | An NVIDIA or AMD GPU with at least 8 GB VRAM; model, quantization and context fit into the available memory budget |
| GPU runtime | Acceleration matching the chosen backend: CUDA for NVIDIA, verified ROCm/HIP or Vulkan support for AMD |
| Device access | The service user can access the GPU; on AMD the required DRM devices and, for ROCm/HIP, additionally `/dev/kfd`. Groups and ACLs must allow this access |
| Container operation | When using Docker/Podman, GPU devices and required driver libraries are available in the container |
| Model store | Own, configurable directory for all model components; in the container as a bind mount or persistent volume |
| Model inventory | Only configured models are loaded; required tokenizers, image projectors, text encoders and VAE are fully present |
| Memory management | Loaded models can be unloaded; the release of VRAM is measured |
| Secrets | Tokens and API keys come from protected files (0600) or the environment and do not appear in logs or process arguments |

## Permission management

Eliezer runs as its own unprivileged service user `eliezer`. For a
user installation the following requirements apply accordingly to the
executing user. The running service needs neither `root` nor `sudo`.
Setting up the user, groups and system directories is up to the administration.

| Resource | Required permissions |
|---|---|
| GPU devices | Access via the actual device groups or suitable ACLs. On AMD, `render` and possibly `video` are particularly relevant; ROCm/HIP additionally needs access to `/dev/kfd` |
| Configuration | Readable by the service, changeable only by the administration; all parent directories must be searchable |
| HF token and API key file | Owned by the executing service user, file mode `0600`; other users get no access |
| Model directory | Read and search for inference; additionally write for downloads and cache management |
| State directory `/var/lib/eliezer` | Read, write and search for the service, in particular for `pins.json` and `measured.json` |
| Runtime directory `/run/eliezer` | Writable by the service, in particular for PID files; provide at service start |
| Backend programs and libraries | Executable or readable by the service; not replaceable through HTTP requests |

Group memberships apply to the service process as actually started.
After a change, it must be restarted with the new groups.
The permissions of the logged-in user alone do not prove the permissions of a separately
set-up service user. Devices are not released blanket-wise to all users.

### API access and model selection

- If `server.api_keys_file` is set, the protected endpoints require a
  configured API key; `/healthz` remains reachable without authentication.
- Without a key file, the API is reachable without authentication. The configured
  bind address and network exposure determine who can reach it.
- Only models on the allowlist are executable. Model keys from requests
  are checked against this list; they do not grant arbitrary file access.
- User roles, individual model permissions and billing are managed by MultiGPT.
  The Eliezer API keys do not distinguish roles; whoever gets direct API access
  also gets access to the protected management endpoints.

### Container operation

The container process receives the required GPU devices and device groups.
UID/GID mapping, supplementary groups and, where applicable, ACLs must allow access to devices and mounted files even with
rootless Docker/Podman operation.
The check is made with the identity of the container process.

Configuration and secret files are mounted read-only.
Model and state volumes must have suitable permissions for the container identity;
the secret files must keep their restrictive access permissions.
Container operation should work without a privileged container.

### Native service operation

The systemd unit uses `User=eliezer`, the required device groups,
`NoNewPrivileges` and `ProtectSystem=strict`. Write access is limited specifically to
the model, state and runtime directories. A differing
model directory must be explicitly taken into account.

## Additionally when building from source

- Rust stable 1.88 or later with Cargo, edition 2024; rustfmt and Clippy for the project checks.
- The development libraries and GPU tools required for the chosen backend.
- CMake and a C/C++ compiler when building stable-diffusion.cpp or llama.cpp.

Concrete backend versions and supported GPU combinations are documented as supported only after a
confirmed build and model run. An installed library
or successful device detection alone is not sufficient for that.

## Hugging Face metadata API (probed 2026-10-20)

Checked against the live service with `curl`, without a token; used by the search (`eliezer-hub`).

- `GET https://huggingface.co/api/models?search=TEXT&filter=gguf&filter=license:LICENSE&pipeline_tag=TAG&sort=downloads&direction=-1&limit=N`
  answers a JSON array. Each item has `id` (`Organisation/Repository`), `downloads`, `likes`, `tags`
  (including `gguf` and `license:<tag>`), often `pipeline_tag` (`text-generation`, `image-text-to-text`,
  `text-to-image`, `feature-extraction`), `createdAt`. It lists no files.
- `GET https://huggingface.co/api/models/{org}/{repo}?blobs=true` answers an object with `sha` (the commit),
  `pipeline_tag`, `tags`, `gated` (`false` or a string), `downloads`, `gguf` (`architecture`, `total`,
  `totalFileSize`) and `siblings[]` with `rfilename`, `size` and `lfs.sha256`. `gguf.architecture` is
  missing for some repositories (the support is then reported as `unknown`).
- Seen: `lmstudio-community/olmOCR-2-7B-1025-GGUF` (architecture and license in the tags);
  `abenzerps/Qwen-Image-2.1-Uncensored-Quantized` (sha `7959e658915f52274897fb573ffeda49de8dc16b`,
  `gguf.architecture` = `qwen_image21`, files `text_encoders/…` and `vae/…`).
- Status codes handled: 401, 403 and 404 as not found, 429 as busy, others as an unexpected answer.

## stable-diffusion.cpp `sd-server` (checked 2026-10-20, commit b13a482)

Built with HIP for gfx1100 as in `tests/runtime/sdcpp/README.md` (`-DSD_HIPBLAS=ON`), run through Eliezer's own adapter.

- Start: `--diffusion-model FILE`, `--llm FILE` (text encoder; for Qwen Image the Qwen3-VL file), `--vae FILE`,
  `--listen-ip`, `--listen-port`, `--offload-to-cpu`, `--steps N`, `--cfg-scale F`, `--sampling-method NAME`,
  `--log-level warn`. The methods are those of `sd-server --help`.
- Ready: `GET /v1/models` answers 200 (3 s after the start with Qwen Image 2.1 and the weights in RAM).
- Draw: `POST /v1/images/generations` with `prompt`, `n`, `size` (`WIDTHxHEIGHT`) and `output_format` (`png`), answer
  `created`, `output_format` and `data[].b64_json`.
- Settings inside a prompt: `<sd_cpp_extra_args>{...}</sd_cpp_extra_args>` (documented in `examples/server/api.md`)
  sets native options per request; Eliezer removes the name from user prompts.
- Measured with Qwen Image 2.1 Q4_K_M, the int8 encoder and the bf16 VAE, 20 steps, cfg 4: 512x512 in 13 s,
  768x512 twice in 50 s, 1024x1024 in 80 s. The weights stay in RAM (idle VRAM +0.6 GB); the VRAM rose by 6.4 GB at most
  during 1024x1024. A child exits when it is killed and leaves no process behind after unloading or a stop of the service.
- FLUX.2 klein 4B (checked 2026-10-20, as `docs/flux2.md` of the commit describes): `--diffusion-model` the Q8_0 file of
  `leejet/FLUX.2-klein-4B-GGUF`, `--llm` `Qwen3-4B-Q8_0.gguf` of `unsloth/Qwen3-4B-GGUF`, `--vae`
  `split_files/vae/flux2-vae.safetensors` of `Comfy-Org/flux2-dev` (same size as the VAE of the gated
  `black-forest-labs/FLUX.2-dev`, but not gated), `--steps 4 --cfg-scale 1.0`. A fox at 1024x1024 took 12 s (2.4 s per step)
  on the RX 7900 XTX; the picture is sharp and matches the prompt. Fetched with `eliezer download` (hash verified).
- Editing (`POST /v1/images/edits`, multipart: `prompt`, `n`, `size`, `output_format`, `image[]`; checked 2026-10-20): both models
  edit with their normal start arguments, no extra option needed. FLUX.2 klein 4B (4 steps): a 1024x1024 fox made blue in
  27 s including the load. Qwen Image 2.1 UC Q4_K_M with the int8 encoder (20 steps, cfg 4): a 512x512 fox made blue in 35 s
  including the load; the safetensors encoder works for editing (the docs ask for `--llm_vision` only with a GGUF encoder).
- MOSS-Image-SFT (checked 2026-10-20, `docs/moss_image.md` of the commit): `--diffusion-model` `MOSS-Image-SFT-Q8_0.gguf` of
  `leejet/MOSS-Image-SFT-GGUF`, `--llm` `Qwen3.5-9B-Q4_K_M.gguf` of `unsloth/Qwen3.5-9B-GGUF`, `--vae` the FLUX.2 VAE (as
  for FLUX.2 klein), `--tokenizer` the `tokenizer/tokenizer.json` of `OpenMOSS-Team/MOSS-Image-SFT` (the external file is
  required: Qwen3.5 has another vocabulary than the embedded one; `extra.tokenizer`), `--cfg-scale 3.0`, default steps (20).
  A cat with a wooden sign "moss-image" at 1024x1024: the text is rendered correctly; 4 min 17 s including the load, 768x768 in
  2 min 21 s (warm), with the weights in RAM (`offload_to_cpu` default) and 10.6 GB VRAM in use. Two requests at once: the second
  waits inside the program and is answered after the first. Slower than Qwen Image 2.1 (80 s at 1024x1024): MOSS runs two
  passes per step (CFG) and a 9B text encoder.
- The server also offers an asynchronous native API (`/sdcpp/v1/img_gen`, `/jobs/{id}`, cancel); Eliezer does not use it yet.

## audio.cpp `audiocpp_server` (checked 2026-10-20, commit 18bba55)

Apache-2.0, ggml based, built with HIP for gfx1100 by `scripts/build-audiocpp.sh` (`--backend hip --model-set custom --models
minimax_music3 --target audiocpp_server`; the server is 223 MB). The model package is `audio-cpp/MiniMax-Music3-GGUF`
(MiniMax Music 3 community license; Q4_0 language model 6.0 GB, Q4_0 flow transformer 1.4 GB, Q8_0 RVQ depth decoder 0.7 GB,
condition encoder and vocoder 0.3 GB; 8.4 GB in all).

- Start: `audiocpp_server --config server.json --no-ui --log`. The JSON holds `host`, `port`, `backend` (`hip`), `device`,
  `lazy_load`, `max_loaded_models`, `ui`, `busy_timeout_ms` and `models` (`id`, `family = minimax_music3`, `path` = the package
  directory, `task = gen`, `mode = offline`, `session_options` with `minimax_music3.language_model_gguf`,
  `minimax_music3.rvq_depth_decoder_gguf`, `minimax_music3.flow_transformer_gguf` and `minimax_music3.mem_saver`,
  `default_request_options` with `num_inference_steps` and `guidance_scale`).
- Ready: `GET /v1/models` lists the model with `"loaded": true` (with `lazy_load = false`); `GET /health` answers at once.
- Compose: `POST /v1/audio/speech` with `model` (the entry's `id`), `input` (the music description; the program calls it the
  text), `response_format = "wav"` and `options` (`lyrics`, `duration_sec`, `seed`, `num_inference_steps`); the answer is the WAV
  file. `duration_sec` is a frame budget, not a hard length. Errors: `500` with `{"error": {"message", "type"}}` (for example a missing component file).
- The program runs one request per model at a time; a second request waits (`busy_timeout_ms = 0` switches the 5 minute limit of the program off).
- Measured on the Radeon RX 7900 XTX through the service: a 20 s song in 1 min 20 s including the load (3 s), the output is 44.1 kHz,
  16 bit, stereo; a warm 30 s song in 2 min 32 s (about 5 times real time, much slower than the 1.35 times of the CUDA numbers
  in the package's README); the weights sit in VRAM only while a stage needs them (`mem_saver`). The output is not silent and varies in
  loudness (checked by the sample values; not listened to). Stopping the service leaves no `audiocpp_server` process and no directory behind.


## whisper.cpp `whisper-server` (read and run 2026-10-30/31, tag v1.9.5)

MIT licensed, ggml based. The program is not installed on this machine and was not built, so nothing below was run against
the real binary. What was **verified** is what the source of `examples/server/server.cpp` and its README on the `master` branch of
`ggml-org/whisper.cpp` say on 2026-10-30 (read over HTTPS; the commit was not recorded, so pin one when the program is built).
What could **not be verified** is listed separately; the live check stays to do, as for audio.cpp (`cargo test -- --ignored` has no
such test yet).

Verified against the source:

- Routes: `POST /inference` (the path prefix can be changed with `--request-path`/`--inference-path`; the defaults are used),
  `POST /load` (swap the model; not used), `GET /health` (503 with a loading status while a model is being loaded, otherwise ready),
  `GET /` (a built-in HTML form; static files of the `--public` folder).
- `POST /inference` is `multipart/form-data`. `file` is required (a missing one is `400`); optional text fields include
  `response_format` (`json` default, `text`, `srt`, `vtt`, `verbose_json`), `language` (a code or `auto`; the default is `en`
  unless the program was started with `--language`), `prompt`, `temperature` (default 0) and `temperature_inc` (0.2),
  `no_language_probabilities`, `translate`, `detect_language`, `no_timestamps`, `offset_t`, `offset_n`, `duration`, `best_of`,
  `beam_size`, `vad` and more. Booleans are true for `true`, `1`, `yes`, `y`. The service sends only `response_format`,
  `no_language_probabilities`, `language`, `prompt` and `temperature`.
- `verbose_json` holds `task`, `language`, `duration`, `text` and `segments`, each with `id`, `text`, `tokens`, `words`,
  `temperature`, `avg_logprob`, `no_speech_prob`, and `start` and `end` in **seconds** (the program's centiseconds times 0.01).
  `json` holds only `text`; `text` is one line per segment; `srt` and `vtt` are subtitle files. The service asks for `verbose_json`
  every time and builds its own `json`, `text`, `srt` and `vtt`, because it needs the segment times to join the pieces of a long
  file; only `segments[].start/end/text` and `language` are read.
- Errors are JSON with an `error` key: `400` for a missing `file` or audio that cannot be read, `500` for a failed inference or a
  failed ffmpeg conversion, `499` when the client left. There is no "busy" answer: `/inference` and `/load` take one mutex, so
  requests run one at a time and the others wait. `/health` does not take the mutex.
- The HTTP port is opened only after the model is loaded (so connection refused until ready; the service polls `/health`).
- Arguments used: `-m/--model`, `--host` (a value ending in `.sock` selects a Unix socket; the service always passes
  `127.0.0.1`), `--port`, `-l/--language`, `-t/--threads`, `-ng/--no-gpu`, `-fa/--flash-attn` and `-nfa/--no-flash-attn`
  (flash attention is on by default), `-dev/--device` (GPU index, default 0; also the environment variable `WHISPER_ARG_DEVICE`),
  `--public`. Long forms are used. `--convert` makes the program start ffmpeg for uploads: **not used**; the README warns to
  run the server sandboxed and to sanitize uploads, which is why the service decodes the audio itself and sends only a WAV file of
  its own making.
- The program's request parser is not hardened against hostile input beyond that, so it listens on loopback only, runs with a
  cleared environment in a private directory, and gets nothing but the files the service wrote.

**Not verified (needs the binary):**

- That the program's WAV reader takes the service's file (16 kHz, mono, 16 bit PCM, 44-byte header, exact `data` size) without
  `--convert`. The README names 16 kHz WAV as the requirement; the reader is a separate module that was not read. If it refuses
  the file, `400` appears in the log and the fix is in `eliezer_speech::wav`.
- The body of `GET /health` when ready (the service only checks for a success status), and that `/health` exists in the
  revision that gets built.
- That an empty `language` falls back to the `--language` of the start; the service therefore always sends `language` itself
  (the request's, else the model's `extra.language`, else `auto`).
- The exact `language` text in `verbose_json` (a name such as `german` or a code); the service shows its own value when the request
  or the model named a language and the program's value only for `auto`.
- Speed, memory (`vram_gb = 2.0` of the example is an estimate for a large-v3-turbo q5_0 model, not a measurement) and the quality of
  the recommended German models. The choice between `cstr/whisper-large-v3-turbo-german-ggml` (`ggml-model-q5_0.bin`,
  Apache-2.0; first choice for German sermons), `primeline/whisper-large-v3-turbo-german` and `openai/whisper-large-v3-turbo`
  is made by a listening test with a real recording (word errors, names, real-time factor on the GPU and the CPU).

Build (from the README of whisper.cpp and the CMake options of ggml; not run here): `cmake -B build -DGGML_VULKAN=1` for Vulkan
(needs the Vulkan headers and `glslc`), `cmake -B build -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1100` for AMD with HIP (older revisions
call the option `GGML_HIPBLAS`; ROCm development files are needed, the same as for audio.cpp), `-DGGML_CUDA=1` for NVIDIA, none
of them for the CPU. Then `cmake --build build -j --config Release --target whisper-server`; the binary is `build/bin/whisper-server`.
`scripts/build-whispercpp.sh` builds the tag v1.9.5 (`d2a764c`) with HIP for the installed card into `target/whispercpp` (the same way as
`scripts/build-audiocpp.sh`; nothing is installed). Bundling the program in the ROCm package is not written yet.

**Live check, 2026-10-30** (workstation with a Radeon RX 7900 XTX, gfx1100, ROCm; model `cstr/whisper-large-v3-turbo-german-ggml`,
`ggml-model-q5_0.bin`, fetched by `eliezer download`): the program takes the 16 kHz mono 16 bit WAV of the service and finds the card
(`ggml_cuda_init: found 1 ROCm devices`). A German sermon of 53 min 48 s (MP3, 48 kHz mono, 42.8 MB) was transcribed through
`POST /v1/audio/transcriptions` in **49 s** (about 66 times real time) in 6 pieces of 10 minutes with a 5 s overlap: 1058 segments,
8133 words, times monotonic, no gap over 25 s, no repeated segments, the joins at 600/1200/1800 s without a lost or doubled sentence,
the Swiss spelling of the speaker kept ("Füsse", "Grösse"). Not measured: the memory of the program, and the word error rate (no
reference text).

**Finding:** with `response_format=verbose_json` the program turns the token times on and then wraps every segment after about 60
characters on a token boundary, which cut words in two ("Füs" / "se."; with `json` the text is whole). The service joins the segments with a
space, so the words reached the text as "Füs se", "bed ienen", "harmon isch". The service therefore sends `split_on_word` from the setting
`backend.whispercpp.split_on_word` (default `true`, per model `extra.split_on_word`; `inference_fields`, tested in
`tests/m8/whispercpp.rs`), so the segments end between words; `max_segment_chars` (default 0) additionally sends `max_len` for a
shorter or longer wrap.

What the tests cover instead: `tests/m8/fake_whisper_server.rs` is a stand-in that speaks the interface as described above
(arguments, `/health`, the multipart form, `verbose_json`); `tests/m8/whispercpp.rs` checks the arguments, the environment, the
private directory, the form, error mapping and crashes against it.

## audio.cpp `audiocpp_server`: speech synthesis with voice cloning (read 2026-10-31, commit 18bba55)

Nothing below was run: the program was built with the music model only and no speech model was downloaded. What was **verified** is the
source of audio.cpp at the pinned commit `18bba55495c8991bbbfe7625234d4d8dbc6e82f4` (fetched over HTTPS and read: `app/server/README.md`,
`app/server/api.md`, `app/server/server_protocol.cpp`, `app/server/runtime.cpp`, `examples/server/README.md`, `docs/tts.md`,
`docs/models/cosyvoice3.md`, `model_specs/chatterbox.json`, `model_specs/cosyvoice3.json`, `src/models/chatterbox/loader.cpp`,
`CMakeLists.txt`, `scripts/build_linux.sh`) and the file lists of the Hugging Face repositories (read with `hf` and the Hub API; nothing
downloaded).

Verified against the source:

- The route is the same as for music: `POST /v1/audio/speech`, JSON. `model` (the entry's `id`), `input` (the text), `language` (a
  top-level string, becomes the language of the text), `speed`/`speaking_rate` (a positive number; an error unless the model's spec
  declares the option, except that 1 is accepted as "not set"), `seed` (and `temperature`, `top_k`, `top_p`, `max_tokens`, ...; also
  inside `options`), `response_format` (`wav` default; `json`/`b64_json` gives base64 WAV in JSON; `mp3` only in a build with the optional
  `mp3_encode` frontend module; `pcm` for streaming), `options` (model specific). `instructions` maps to the model option `instruction`.
- The reference voice for cloning: `voice_ref` is **either a path string or an object** `{"type": "path", "path": ...}` (a path on the
  server's machine; relative paths resolve against the config file's directory) **or** `{"type": "base64", "data": ...}` (WAV bytes,
  also a `data:audio/wav;base64,` URI; decoded limit 5 MiB, so about 110 s of 24 kHz mono 16 bit). `reference_text` is the transcript
  of the reference (a model option; models that do not take it ignore it or refuse it). The service sends `voice_ref` as a `path`
  object built from the file in its own voice directory and never sends `voice`, so the server's own `voice_dir`/`prompt_text` library and
  its `voice_presets` are not used (a name that matches neither is passed to the model as a "cached voice id", which is why the service
  does not send names).
- Response: the WAV file (`audio/wav`, 16 bit PCM from `encode_pcm16_wav`, the sample rate of the model, 24 kHz for Chatterbox as far
  as its S3Gen decoder is concerned; not measured). Errors are JSON with `error.message`; a bad request is 4xx, a failed run 5xx.
- Streaming exists (`stream_format = "sse"` or `"audio"`, base64 PCM chunks) but only for models configured with `"mode": "streaming"`; the
  service does not use it (it joins whole passages, and Chatterbox is `offline` only).
- Model entry of the server config (`models[]`): `id`, `family`, `path` (a `.gguf` **file** for single-file packages such as
  Chatterbox and CosyVoice3, a directory for packages with several files), `task` (`tts`; `clon`/`clone` is the same path for Chatterbox:
  "`tts` is an alias to the same voice-cloning synthesis path"), `mode = "offline"`, `default_request_options` (request options as numbers or
  strings), `session_options` (`chatterbox.weight_type`, `chatterbox.mem_saver`, `chatterbox.multilingual_t3 = v2|v3`,
  `chatterbox.conditionals_cache_slots` default 1, ...; not used by the service, they stay at the program's defaults), `busy_timeout_ms`.
  The service writes `id`, `family`, `path`, `task = "tts"`, `mode = "offline"`, `busy_timeout_ms = 0` and the numbers of `extra.tts_options` as
  `default_request_options`.
- `chatterbox` (docs/tts.md, `model_specs/chatterbox.json`): tasks `tts`, `clon`, `vc`; offline; languages `ar da de el en es fi fr hi it ko
  ms nl no pl pt sv sw tr`; a reference WAV is required ("Built-in voices: not exposed"); the default language is `en`, so the service
  sends the model's `extra.language`. Request options: `exaggeration` 0.5, `guidance_scale` 0.5, `temperature` 0.8, `repetition_penalty`
  1.2, `min_p` 0.05, `top_p` 1, `flow_guidance_scale` 0.7, `max_tokens` 384, `do_sample`, `stop_on_eos`, `text_chunk_size` 128 (the
  program cuts long text itself; the service cuts earlier, at sentence boundaries), `seed`. License of the model: MIT
  (`docs/model_licenses.md`). Speech model with `speed` support: not declared by the spec, so a `speed` other than 1 is expected to be refused.
- `cosyvoice3` (`docs/models/cosyvoice3.md`): tasks `tts`, `clon`; languages `zh en ja ko de es fr it ru yue`; reference WAV required;
  `template_name` default `zero_shot` (voice cloning with prompt audio **and its transcript**: this is what `<name>.txt` is for),
  other templates `cross_lingual`/`instruct` are not used. Options: `text_chunk_size` 600, `max_tokens`, `min_tokens`, `top_k` 25,
  `num_inference_steps` 10, `seed` 1986. License Apache-2.0.
- Build: `scripts/build_linux.sh --model-set custom --models LIST` passes `LIST` to CMake as `AUDIOCPP_MODELS`; CMake accepts commas
  or semicolons, strips blanks, and takes every name as a target or alias of `audiocpp_add_model(...)` (`minimax_music3`,
  `chatterbox`, `cosyvoice3` exist at the commit; an unknown name stops with "Unknown AUDIOCPP_MODELS entry"). A comma separated list
  therefore builds several families into one `audiocpp_server`. `scripts/build-audiocpp.sh` takes the list from `AUDIOCPP_MODELS`
  (default `minimax_music3`) and keeps one cache stamp per list.
- GGUF packages (file lists of `audio-cpp/audio.cpp-gguf` at its revision `874fcb41`, 2026-10-27, license field `other`, per-model
  licenses in `docs/model_licenses.md`): `Chatterbox-GGUF/chatterbox-q8_0.gguf` (2 088 399 204 bytes) and `chatterbox-f16.gguf`
  (3 744 365 922 bytes), `Chatterbox-Turbo-GGUF/chatterbox-turbo-q8_0.gguf` (English only), `CosyVoice3-GGUF/cosyvoice3-q8_0.gguf`
  (2 257 658 336 bytes) and `cosyvoice3-f32.gguf`. Fetch with `eliezer download audio-cpp/audio.cpp-gguf Chatterbox-GGUF/chatterbox-q8_0.gguf`
  (the repository must be on `huggingface.allow`).

Not verified (needs a model and the binary):

- That `audiocpp_server` accepts a `.gguf` file as `path` for Chatterbox. The Kokoro/Pocket examples and `docs/gguf.md` say GGUF files are
  single-file packages, and `audio-cpp/audio.cpp-gguf` publishes Chatterbox that way, but the loader (`src/models/chatterbox/assets.cpp`)
  was not read to the end.
- The sample rate and channels of the answer, and that all passages of one request have the same format (the service refuses to join
  parts that differ and answers 502; a stream ends with an aborted connection). The service does not assume 24 kHz: it reads the rate from
  the first passage's WAV and reports it in `X-Audio-Sample-Rate`. Also not verified: that dropping the HTTP request to `audiocpp_server`
  (a client that went away) stops the program's work on that passage.
- That a changing `seed` or the same `seed` for different passages and voices is accepted: the spec says the cloning settings, including
  `seed`, are "captured during clone preparation; changing it on the same prepared session is rejected". The service sends the same
  `seed` for every passage of a request and nothing otherwise; a request with another seed for the same voice may be refused by the program
  (it would show as a 400). To be tested live.
- Memory (`vram_gb = 4.0` of the example is an estimate for the q8_0 file of 2.1 GB plus the decoders), speed, and the quality of
  German speech with a cloned voice (a listening test with a real parent's recording decides between Chatterbox and CosyVoice3).
- **Ming-3.0-TTS and XTTS v2 are not available at the pinned commit.** `audio-cpp/Ming-3.0-TTS-0.5B-GGUF` (MIT, `ming-3.0-tts-0.5b-q8_0.gguf`
  and `-bf16.gguf`) was published on 2026-10-21, two days after the commit (2026-10-19), and there is no model target or spec for it in the
  source; the commit has to be moved before it can be tried. XTTS v2 does not appear anywhere in audio.cpp (no target, no spec, no GGUF
  repository under `audio-cpp`); it is not served by this backend.
- mp3 and flac are not offered: the program writes mp3 only in a build with an optional frontend module, and the service has no encoder
  (symphonia only decodes). `response_format` is `wav`.

What the tests cover instead: `tests/m7/fake_audio_server.rs` answers the request as described above (the model entry, `voice_ref` and
the options are recorded); `tests/m7/audiocpp.rs` checks the configuration file, the request body and the error mapping;
`tests/m9/tts_api.rs` checks the endpoints with the fake backend.

**Live check, 2026-10-31** (real `audiocpp_server` with `chatterbox-q8_0.gguf`): the `.gguf` file is accepted as the model `path`, the
model loads, and `GET /v1/models` reports the family with `task: tts`. **Chatterbox needs a reference voice**: a request without
`voice_ref` fails with HTTP 500 `{"error":{"message":"Chatterbox prepare requires speaker reference audio","type":"server_error"}}`.
The service therefore has `extra.requires_voice` (set for Chatterbox in the example configuration) and answers such a request with a 400
before it reads anything. The message of any error answer of the program is written to the log at WARN (`error.message`, control
characters removed, cut to 300 characters, never the request text). It is not put into the API answer: the error types of the service are
unit values without a detail field, and a program message could leak internals.
