# Eliezer

**English** · [Deutsch](README.de.md)

**One GPU, many kinds of models, one service.** Eliezer is a headless model manager for Linux, written in Rust. It serves
local language, vision, image, music, speech synthesis and speech recognition models behind OpenAI- and Anthropic-compatible
HTTP interfaces. It loads each model when a request needs it, unloads it when idle and makes the models take turns on a
single graphics card. It was built to replace LM Studio as the model server of the self-hosted family AI system
[MultiGPT](https://github.com/Beerlesklopfer/multi-gpt), and it works with any client that speaks those interfaces.

The name comes from Eliezer, the servant of Abraham who managed the household and carried out his errands
([why this name](docs/wiki/ProjectName.md)).

## The problem

A home server usually has one GPU with 16 or 24 GB of memory. A chat model, an image model, a speech recogniser and a
voice model will not all fit at once, and a desktop application with a window is a poor fit for a service in the basement.
Eliezer treats GPU memory as a budget. It knows how much memory each model needs, measures what the card really uses,
evicts idle models by priority, queues requests that have to wait, and moves language models to the CPU while an image
model is using the GPU.

## What it serves

| Kind | Endpoint | Command line |
|---|---|---|
| Chat, completion, embeddings, vision | `/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`, `/v1/messages`, `/v1/messages/count_tokens` | `eliezer run MODEL [PROMPT]` |
| Images (generation and editing) | `/v1/images/generations`, `/v1/images/edits` | – |
| Music (lyrics and a description) | `/v1/audio/speech` with a `music` model | – |
| Speech synthesis (voice cloning, streaming) | `/v1/audio/speech`, `/v1/audio/voices` | `eliezer speak` (or `say`), `eliezer voices` |
| Speech recognition | `/v1/audio/transcriptions` | `eliezer transcribe` |
| Models and memory | `/v1/models`, `/api/v1/models`, `/api/v1/models/load`, `/api/v1/models/unload`, `/api/v1/models/estimate`, `/api/v1/status`, `/healthz` | `ls`, `ps`, `load`, `unload`, `status`, `show` |
| Models from Hugging Face | `/api/v1/hub/*`, `/api/v1/models/download`, `/api/v1/models/pins` | `search`, `inspect`, `get`, `download`, `pins`, `rm` |

Because it speaks the Anthropic Messages API as well as the OpenAI API, clients such as Claude Code can work against local
models. The management API (`/api/v1`) has the same shape as that of LM Studio 0.4, so existing integrations keep working.
The full interface is in [docs/wiki/API.md](docs/wiki/API.md).

## How it works

```
 clients (MultiGPT, Claude Code, curl, eliezer CLI)
        │  HTTP: OpenAI, Anthropic, LM Studio management API
        ▼
 ┌──────────────────────── eliezer (one process) ────────────────────────┐
 │  axum router ──► supervisor task: the only owner of the state         │
 │                  planner (pure functions): VRAM ledger, queue,        │
 │                  eviction by priority, idle unloading, CPU placement  │
 │                     │                         │                       │
 │      llama.cpp, embedded as a crate     child processes, started on   │
 │      (language, vision, embeddings)     demand on loopback ports:     │
 │                                         sd-server (images)            │
 │                                         audiocpp_server (music, TTS)  │
 │                                         whisper-server (recognition)  │
 └───────────────────────────────────────────────────────────────────────┘
        │  GPU memory measured through sysfs, rocm-smi, NVML or nvidia-smi
        ▼
   one GPU (AMD ROCm, NVIDIA CUDA or Vulkan) and the CPU with a RAM budget
```

- **One state owner.** A single supervisor task changes the state. HTTP handlers send it messages and wait for the answer.
  All decisions (what fits, what to evict, what waits) live in `eliezer-core` as pure functions without I/O, so they are
  tested without a GPU and without waiting for real time.
- **Memory you can trust.** The planner computes each model's needs (weights, KV cache for the chosen context and
  parallelism, including sliding-window and linear layers, image projector) from architecture figures that
  `eliezer --import` reads from the GGUF header. The usage measured after a load is stored and takes precedence next time.
  A request that cannot fit gets a `409` that names the setting to change, instead of the card running out of memory.
- **The right engine for each kind.** Language models run in the process through the llama.cpp crate.
  [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp), [audio.cpp](https://github.com/0xShug0/audio.cpp)
  and [whisper.cpp](https://github.com/ggml-org/whisper.cpp) run as child processes. Each is started without a shell, with a
  cleared environment and arguments built only from the configuration, and stopped again when idle or at shutdown.
- **GPU priority.** While an image model is loaded or waiting, language models set to `device = "auto"` move to the CPU
  instead of evicting it, and come back when the GPU is free.
- **Streaming audio.** Speech synthesis splits a long text into passages at sentence boundaries and streams WAV, PCM or MP3
  while it reads, so playback starts before the text is finished:

```bash
echo "Good morning" | eliezer say --file - --voice anna | play -t wav -    # read aloud while it is made
eliezer speak --file chapter.txt --voice anna --format mp3 -o chapter.mp3  # MP3, encoded on the fly
eliezer voices add --name anna --file anna.mp3 --file anna.txt --start 5:00 --length 30
eliezer transcribe ~/Audio --recursive --out ~/Texts --language de         # txt, srt, vtt and json
```

## How it differs

Many of Eliezer's single features are available elsewhere by now. Projects with a similar scope:

| Project | Strength | Difference from Eliezer |
|---|---|---|
| [Lemonade](https://github.com/lemonade-sdk/lemonade) | the closest overall: nearly all model kinds, live VRAM polling on AMD, eviction by threshold, ROCm, Debian package | C++; Anthropic requests are translated; no per-model estimate or queue documented |
| [LocalAI](https://github.com/mudler/LocalAI) | the broadest range of model kinds, including music and voice cloning; VRAM budget with estimates | Go, containers; no live measurement from the driver documented |
| [llama-swap](https://github.com/mostlygeek/llama-swap) | the most configurable swapping: capacity pools, eviction costs, priority queue | the operator declares every limit; no estimate or measurement |
| [Ollama](https://github.com/ollama/ollama) | wide adoption; queue, idle unloading, CPU/GPU split at load time | own model registry; image generation not documented for Linux |

As far as the documentation of these and about twenty other projects shows (checked November 2026), none of them
combines what Eliezer does:

1. Language models move to the CPU automatically while an image model holds the GPU, and move back afterwards.
2. A management API compatible with LM Studio, so existing integrations keep working.
3. Downloads pinned to a commit and checked against SHA-256, from an allowlist of repositories.
4. A per-model memory estimate, live measurement from the driver, eviction by priority and a queue, all together.
5. Written in Rust, packaged for Debian as a service, with the ROCm runtime bundled.

"Not documented" does not prove "not present": the comparison is based on the projects' documentation, not their source code.

## Measured on real hardware

All figures are from one machine with an AMD Radeon RX 7900 XTX (24 GB, ROCm), checked through the service's own endpoints.
Details and the circumstances of each measurement are in [docs/wiki/Status.md](docs/wiki/Status.md) and
[docs/Runtimes.md](docs/Runtimes.md).

| Task | Model | Result |
|---|---|---|
| Load, answer, unload a language model (installed package) | Mistral | 1.53 s, 0.37 s, 0.03 s |
| Read a scanned invoice (vision) | olmOCR 2 7B, Qwen3.8 27B, Gemma 4 26B | all three read the invoice number and total correctly |
| Language model on the CPU while an image model holds the GPU | Mistral | answered in 3 s, image model stayed loaded |
| Draw an image | Qwen Image 2.1 Q4_K_M, 20 steps | 13 s at 512×512, about 80 s at 1024×1024 |
| Compose a song | MiniMax Music 3 Q4_0 | a 30 s song in 2 min 32 s (model already loaded) |
| Read a text aloud with a cloned voice | Chatterbox Q8_0 | 5 min 45 s of audio in 94 s, first sound after 0.9 s |
| Transcribe a German recording | Whisper large-v3-turbo (German) | 53 min 48 s in 49 to 53 s, about 60× real time |
| Round trip: text → speech → text, 1 504 words | Chatterbox + Whisper | word error rate 0.8 % to 1.9 % |

## Security and operation

- **Allowlist.** Only models listed in the configuration are loaded. Model names in requests are keys from that list, never
  paths, and no request value reaches a command line or a shell unchecked.
- **Secrets.** API keys and the Hugging Face token live in files with mode 0600 or in the environment. They never appear in
  the repository, in logs, in error messages or in process arguments visible in `ps`. `eliezer --set-hf-token` reads the
  token from standard input or a hidden prompt, never from an argument.
- **Controlled downloads.** Downloads are limited to allowed repositories, pinned to a commit and checked against the
  SHA-256 hashes Hugging Face lists. Interrupted downloads resume.
- **Tests without hardware.** Close to 400 tests run without a network and without a GPU, against stand-in backends that
  imitate llama.cpp, sd-server, audio.cpp and whisper-server, including their failures (crash while loading, never ready,
  death after the first answer). Hardware tests are marked and run only on request.

## Trying it

Eliezer comes as a Debian package (with the ROCm runtime and the helper programs bundled; about 1.05 GB for one GPU
architecture, no models included), as a container image with Compose files for Docker and Podman, or built from source with
Cargo. The model runtime is chosen at build time: `rocm`, `cuda`, `vulkan` or `cpu`. Installation, the systemd service,
containers, importing models from LM Studio and setting up keys are described in [INSTALL.MD](INSTALL.MD).

```bash
sudo eliezer --config /etc/eliezer/config.toml --import ~/.lmstudio/models --dry-run   # what would be taken over
sudo eliezer --new-key --name multigpt --save      # an API key for a client, shown once
eliezer search "qwen3 coder" --task llm --details  # Hugging Face candidates, judged for fit; nothing is downloaded
eliezer load qwen/qwen3-coder-30b --context 32768 --parallel 2
eliezer run qwen/qwen3-coder-30b "Explain mutexes in two sentences" --stats
eliezer ps                                         # loaded models, device, memory, idle time
```

`eliezer help` lists every command; `eliezer COMMAND --help` shows one.

## Status and limits

Version 0.6.0, developed since October 2026. Everything in the tables above is implemented and was checked live.

- **NVIDIA:** CUDA builds and NVML measurement exist, but they have never run on a real NVIDIA card. AMD is the tested
  platform.
- Speech synthesis reads digits (years, verse numbers) less reliably than words; a step that writes numbers as words is
  not built yet.
- Live speech input as a stream (`/v1/audio/transcriptions/stream`) is specified but not built.
- Not a goal: clusters, user management, billing or a web interface. Permissions, ranking and billing belong to the client
  (MultiGPT); Eliezer executes and reports facts.

## Project

- License: [GPL-3.0](LICENSE). Bundled third-party components and their licenses: [docs/wiki/Dependencies.md](docs/wiki/Dependencies.md).
- Specification: [docs/Specification.md](docs/Specification.md), roadmap with milestones: [docs/Implementation.md](docs/Implementation.md),
  findings about the third-party programs: [docs/Runtimes.md](docs/Runtimes.md).
- Eliezer is developed with AI coding agents working under written rules ([AGENTS.md](AGENTS.md)): small commits, every
  third-party interface checked against the real program before use, measurements recorded in the status log.
