# Implementation Framework for Eliezer

As of 2026-10-19, draft. Eliezer replaces LM Studio as the model server of
[MultiGPT](https://github.com/Beerlesklopfer/multi-gpt) (there, milestone M16 = integration).
**Read first:** [../AGENTS.md](../AGENTS.md) (rules) and [Specification.md](Specification.md)
(structure, configuration, interfaces, scheduler, tests). This document organizes the work.

Decided: everything in Rust, runtime mistral.rs, models via Hugging Face token (only on the GPU machine),
a single image selection in chat. The target version is set by the user.

Way of working: milestones in order. After each milestone `cargo fmt --check`, `cargo clippy -D warnings`,
`cargo test` green, a short note in `docs/Status.md`, then wait for approval. The specification is kept
up to date with decisions (no silent deviations).

The results of each milestone are also recorded in `debian/changelog` in Debian format.
Version numbers are set by the user; open work is not reported as
completed. Target hardware: at least 8 GB VRAM, with an adapted model selection.

## Milestones

### M1 – Trial on the GPU machine (isolated probe programs, no production code)

**Completed:** [Result and decisions](../tests/m1/README.md),
milestone version 0.1.0; entry in `debian/changelog`.

Goal: **establish** the interfaces of third-party programs, do not guess.

Start, load and unload parameters are taken from the existing Ollama and
LM Studio installations and mapped onto the embedded runtimes.
Verified command lines, mappings and results are documented under
`tests/reference-clients/`.

- **M1-01** Embed mistral.rs **as a crate** in a small test program (no child process): load a model, answer, unload. Establish and record in `docs/Runtimes.md`: version, Cargo features per hardware, builder calls (text, vision), whether/how unloading and reloading work in-process and whether the VRAM is free afterwards, streaming and `usage`, tool calls, reasoning options, chat template, whether `mistralrs-server-core` can be embedded. Measurements: load time, tokens/s, VRAM, two simultaneous requests.
- **M1-02** stable-diffusion.cpp (server) with `qwen-image-2.1-UC-Q4_K_M.gguf`, text encoder and VAE: command line,
  endpoints, time per image, VRAM. Also in `docs/Runtimes.md`.
- **M1-02a** Determine the card vendor (`/sys/class/drm/card*/device/vendor`) (NVIDIA or AMD). Build mistral.rs and stable-diffusion.cpp on that card and check
  which acceleration runs (CUDA, ROCm/HIP, Vulkan); on AMD additionally first Candle directly (pure Rust, as provided for in the architecture plan), then the llama.cpp binding (build and measure Vulkan/ROCm). Result per backend in `docs/Runtimes.md`.
  If the mistral.rs crate does not run there, the **embedded** backend `llamacpp` (binding crate, Specification 8) becomes part of M4;
  then choose the binding crate and record its gaps (chat template, tool calls, vision, parallelism) in `docs/Runtimes.md`.
- **M1-03** Switch between chat and image on the 24 GB card by hand; for comparison llama-swap with the same models.
- **M1-04** Decision: off-the-shelf manager (llama-swap) or custom build; mistral.rs embedded (decided) – fallback `llamacpp` (also embedded) for cards the crate does not support.
  If it falls back to "off-the-shelf", M2 to M3 are replaced by an adaptation and the specification is shortened accordingly.

*Acceptance:* `docs/Runtimes.md` contains for each backend a **working command line and the measured values**; the decision is recorded in it.

### M2 – Scaffold

**Completed:** [Checks and results](../tests/m2/README.md),
version 0.2.0; installation in [INSTALL.MD](../INSTALL.MD).

- **M2-01** Workspace per Specification 3, license (user chooses), `rust-toolchain.toml`.
- **M2-02** CI (`.github/workflows/rust.yml`): format, Clippy, test, Linux only.
- **M2-03** Configuration (`eliezer-core::config`): structs with `deny_unknown_fields`, checks (duplicate keys,
  port ranges, paths), error messages with line numbers. Test: the example from Specification 4 loads; one error case each.
- **M2-03a** Central model store (`storage.models_dir`) and path check: all model components
  and the HF cache below it; reject paths outside the store, including via symlinks.
- **M2-04** Program frame (`eliezer`): arguments (`--config`, `--check-config`), `tracing`, start/stop with `SIGTERM`,
  `SIGHUP` reload as a placeholder, unit `packaging/eliezer.service`.

- **M2-05** Set up container build and `compose.yaml`, extended by `compose.volume.yaml`:
  own model directory as a bind mount or alternatively a persistent Docker/Podman volume;
  persistent state data, read-only configuration, separate secret files and
  GPU access per M1. Verify start, stop and re-creation with both storage variants;
  document Docker/Podman prerequisites and differences.

*Acceptance:* Compose configurations are verified, the service starts and model files are preserved
when the container is re-created. `cargo run -- --check-config beispiel.toml` reports "Configuration OK"; faulty files name line and field.

### M3 – Supervisor (without real backends)

- **M3-01** `eliezer-core`: VRAM calculation (Specification 6) with all architecture types; tests with the example values.
- **M3-02** State machine and scheduler (Specification 7), `Clock` trait; tests for eviction, pinning, batching,
  idle, queue, error state.
- **M3-03** `eliezer-gpu`: **one** trait `GpuProbe` (Specification 6a) with the methods `nvml`, `nvidia-smi`, `sysfs`, `rocm-smi`, `none` and test bench; vendor detection via `device/vendor`; uniform meaning (bytes, total used), timeout and fallback; one shared test suite.
- **M3-04** `eliezer-backend`: trait `Backend` and **fake-backend**; supervisor task (`Supervisor`) with mpsc/oneshot.
- **M3-05** `eliezer-api`: `/api/v1/models`, `/load`, `/unload`, `/api/v1/status`, `/healthz`, authentication, error format;
  chat forwarding `/v1/chat/completions` with **unbuffered SSE** against the fake-backend.
- **M3-05a** Anthropic interface (Specification 5.1a): module `eliezer-api::anthropic` with translation in both directions, `POST /v1/messages`
  (with and without streaming), `count_tokens`, model list, error format; tests against the fake-backend.
- **M3-06** Contract test (Specification 11) with `tests/contract/` files.

*Acceptance:* All integration tests from Specification 11 run without network and GPU; the same request yields the same content via the OpenAI and Anthropic interfaces; MultiGPT (provider `lmstudio`, `base_url`
pointing to Eliezer with fake-backend) lists models, loads and unloads.

### M4 – mistral.rs runtime

- **M4-01** Backend `mistralrs` as an embedded crate per `docs/Runtimes.md`: load/unload via the handle, check "VRAM free after unload" (`unload_leak`, `needs_restart`), compute work on dedicated threads, panic catching, feature `mistralrs-cuda`.
- **M4-01a** (only if M1 requires it) Backend `llamacpp` embedded via the chosen binding crate: load/unload, chat template, streaming, parallelism, tool calls, `usage`, vision as far as possible; same checks as M4-01.
- **M4-02** Chat and vision, tool calls, streaming with `usage`; chat template per model.
- **M4-03a** Anthropic interface against a real model: verify tool call and streaming with an Anthropic client (Specification 12, point 4).
- **M4-03** Reasoning options (`reasoning_effort` → backend), statistics in the last SSE chunk, cross-check of the VRAM calculation (measured value is stored).
- **M4-04** GPU tests (`#[ignore]`) and measurements in `docs/Status.md`.

*Acceptance:* With only `base_url` changed, MultiGPT works with a real chat model (model list, loading with context and
parallelism, chat with streaming, reasoning options, statistics).

### M5 – Model acquisition

- **M5-00** Targeted search for specialized models on Hugging Face: define search interface and
  result format; take purpose, format, quantization, license, backend support and
  VRAM requirement into account. Check candidates with model card and additional files; adding to
  the allowlist and downloading remain explicit administrative steps.
- **M5-01** Allowlist with revision/commit hash (`pins.json`), report deviations.
- **M5-02** `POST /api/v1/models/download`, progress in `/api/v1/status`, check `cache_limit_gb` (507), license notice on 401/403.
- **M5-03** Token from file (0600, check permissions at startup) or `HF_TOKEN`; passed to child processes only via the environment; check script `scripts/check-secrets.sh`
  (searches logs, `ps`, error messages).

*Acceptance:* Specialized models can be found in a targeted way and checked for suitability. Downloads land
exclusively in the central model store. Models can be loaded on request; no token in logs, process list or messages.

### M6 – Image service

- **M6-01** Backend `sdcpp` (Qwen-Image-GGUF) per `docs/Runtimes.md`; `/v1/images/generations`; exclusive scheduling (Specification 7, point 4).
- **M6-02** GPU test switching chat ↔ image, measurements.
- **M6-03** (later) FLUX.2 klein 4B in Rust, as soon as the architecture is available in mistral.rs/candle: separate plan, not part of the acceptance.

*Acceptance (overall):* Specification 12.

## Order

```
M1 ─> M2 ─> M3 ─┬─> M4 ─> M5
                └─> M6
```
