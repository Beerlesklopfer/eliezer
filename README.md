# Eliezer

**Replacement for LM Studio** for family use (NVIDIA and AMD, OpenAI- and Anthropic-compatible): a manager for local AI models on a machine with a scarce GPU. Named after Eliezer, the servant
of Abraham who managed the household and carried out his errands.

The origin of the name and what it means for the project are explained in the
[project wiki](docs/wiki/ProjectName.md).

Eliezer allocates GPU memory: chat, vision and image models are loaded on demand, unloaded when idle,
and take turns on one GPU. The service sits behind an
OpenAI-compatible interface and is driven by [MultiGPT](https://github.com/Beerlesklopfer/multi-gpt)
through its runner; permissions, ranking and billing stay there.

**Status:** Package version 0.6.0. M4 and M5 are implemented: the embedded llama.cpp runtime loads and serves the language models (also with image input and in system memory) on AMD and other GPUs, behind the supervisor (queue, eviction, idle unloading, GPU priority for image models) and the OpenAI- and Anthropic-compatible interfaces, with a command line client, a model import, a search of Hugging Face and downloads with pins. The image service (stable-diffusion.cpp) follows in M6. [Checks](tests/m2/README.md). Specification: [docs/Specification.md](docs/Specification.md), roadmap: [docs/Implementation.md](docs/Implementation.md), rules for agents: [AGENTS.md](AGENTS.md).

Installation and operation: [INSTALL.MD](INSTALL.MD).

## Goals

- Replace LM Studio as the model server of MultiGPT. The `lmstudio` provider in MultiGPT keeps working until Eliezer fulfils the same contract: model list with state and memory estimate, load/unload with context and parallelism, reasoning options per model, statistics per response.
- Runtime for language models: [mistral.rs](https://github.com/EricLBuehler/mistral.rs) (Rust).
- Image models: stable-diffusion.cpp first as a replaceable service, Rust later.
- Manager: route requests by `model`, VRAM ledger, idle unloading, queue.
  The comparison with llama-swap is finished; the own Rust manager is being implemented.
- Models are downloaded with a Hugging Face token (only on the GPU machine, never in the repository).
