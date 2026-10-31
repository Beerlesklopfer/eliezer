# Eliezer

**Replacement for LM Studio** for family use (NVIDIA and AMD, OpenAI- and Anthropic-compatible): a manager for local AI models on a machine with a scarce GPU. Named after Eliezer, the servant
of Abraham who managed the household and carried out his errands.

The origin of the name and what it means for the project are explained in the
[project wiki](docs/wiki/ProjectName.md).

Eliezer allocates GPU memory: chat, vision and image models are loaded on demand, unloaded when idle,
and take turns on one GPU. The service sits behind an
OpenAI-compatible interface and is driven by [MultiGPT](https://github.com/Beerlesklopfer/multi-gpt)
through its runner; permissions, ranking and billing stay there.

**Status:** Package version 0.6.0. Implemented: the embedded llama.cpp runtime loads and serves language models (also with image input and in
system memory) on AMD and other GPUs, behind the supervisor (queue, eviction, idle unloading, GPU priority for image models) and the OpenAI- and
Anthropic-compatible interfaces; a command line client; a model import; a search of Hugging Face and downloads with pins; the image service
(stable-diffusion.cpp); music and speech synthesis with voice cloning (audio.cpp) and speech recognition (whisper.cpp), each as a program started
on demand. Details and measurements: [docs/wiki/Status.md](docs/wiki/Status.md). Specification: [docs/Specification.md](docs/Specification.md),
roadmap: [docs/Implementation.md](docs/Implementation.md), rules for agents: [AGENTS.md](AGENTS.md), [checks](tests/m2/README.md).

Installation and operation: [INSTALL.MD](INSTALL.MD). The HTTP interface: [docs/wiki/API.md](docs/wiki/API.md).

## What it does

| Kind | Endpoint | Command |
|---|---|---|
| Chat, completion, embeddings, vision | `/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`, `/v1/messages` | `eliezer run MODEL [PROMPT]` |
| Images | `/v1/images/generations`, `/v1/images/edits` | – |
| Speech synthesis (voice cloning, streaming) | `/v1/audio/speech`, `/v1/audio/voices` | `eliezer speak` (or `say`), `eliezer voices` |
| Speech recognition | `/v1/audio/transcriptions` | `eliezer transcribe` |
| Models and memory | `/api/v1/models`, `/api/v1/status` | `ls`, `ps`, `load`, `unload`, `status`, `show` |
| Acquisition | `/api/v1/hub/*`, `/api/v1/models/download` | `search`, `inspect`, `get`, `download`, `pins`, `rm` |

`eliezer help` lists everything; `eliezer COMMAND --help` (or `eliezer help COMMAND`) shows one command only.

```bash
echo "Hallo Jörg" | eliezer say --file - --voice mama | play -t wav -        # read aloud while it is made
eliezer speak --file kapitel.txt --voice mama --format mp3 -o kapitel.mp3    # mp3, encoded on the fly
eliezer voices add --name mama --file mama.mp3 --file mama.txt --start 5:00 --length 30
eliezer voices                                                               # or: eliezer --list-voices
eliezer transcribe ~/Audio --recursive --out ~/Texts --language de
```

## Goals

- Replace LM Studio as the model server of MultiGPT. The `lmstudio` provider in MultiGPT keeps working until Eliezer fulfils the same contract: model list with state and memory estimate, load/unload with context and parallelism, reasoning options per model, statistics per response.
- Runtime for language models: llama.cpp, embedded as a crate  (the `llamacpp` backend).
- Image models: [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp) as a replaceable service, Rust later.
- Speech: [audio.cpp](https://github.com/0xShug0/audio.cpp) for synthesis and music, [whisper.cpp](https://github.com/ggml-org/whisper.cpp) for recognition.
- Manager: route requests by `model`, VRAM ledger, idle unloading, queue (implemented in Rust).
- Models are downloaded with a Hugging Face token (only on the GPU machine, never in the repository).
