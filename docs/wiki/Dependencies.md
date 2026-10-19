# Dependencies and Requirements

The Cargo workspace is set up. The dependencies of the program frame are in
`Cargo.lock`; versions and rationale are documented in
[Dependency Decisions](../Dependencies.md). Model runtimes
are added only when they are actually integrated.

## Rust and development tools

| Component | Purpose | Requirement / verified state |
|---|---|---|
| Rust stable | Compiler, edition 2024 | Minimum version 1.88; locally 1.99.0 under `~/.cargo/bin` |
| Cargo | Build and tests | Locally 1.99.0 under `~/.cargo/bin` |
| rustfmt | Consistent formatting | Locally 1.10.0-stable |
| Clippy | Static code analysis | Locally 0.1.99 |

The user installation must come before the older system installation in `PATH`:

```bash
export PATH="$HOME/.cargo/bin:$PATH"
```

The project checks are `cargo fmt --check`,
`cargo clippy --all-targets -- -D warnings` and `cargo test`.
They cover the entire workspace; the standalone M1 GPU probe programs remain separate.

## Intended Rust crates

| Crate | Rationale |
|---|---|
| `tokio` | Asynchronous runtime and communication with the supervisor |
| `axum` | HTTP endpoints and streaming |
| `reqwest` with rustls | Outgoing HTTP requests without an OpenSSL dependency in the client |
| `serde` | Typed serialization and configuration |
| `tracing` | Structured operational logs |
| `mistralrs` | Intended embedded model runtime; the currently reviewed Git state 0.9.5 requires Rust 1.94. Version and hardware suitability must be determined before adoption |

New crates are added sparingly and recorded with their rationale in
[`docs/Dependencies.md`](../Dependencies.md).
This page replaces neither `Cargo.toml` nor the version pinning in `Cargo.lock`.

## Model runtimes

| Component | Role | State |
|---|---|---|
| mistral.rs | Language and vision models in the Eliezer process | Intended; model run and memory release still to be verified |
| Candle directly | Evaluation as a Rust fallback before llama.cpp | AMD suitability still open |
| llama.cpp with Rust binding | Optional embedded fallback, especially for AMD | Binding and features not yet decided |
| stable-diffusion.cpp | Separate image model process | Program and interface still to be verified |
| fake-backend | Test bench for automated tests without network and GPU | To be implemented in M3 |

The locally installed LM Studio `llama-server` has so far served only for the device check.
Its detection does not yet prove a working embedding of llama.cpp in Rust.

## GPU and access permissions

The target is Linux with a GPU and at least **8 GB VRAM**. Model size,
quantization and context must fit the available memory.

- NVIDIA: CUDA for the matching model runtime; VRAM measurement via NVML or `nvidia-smi`.
- AMD: ROCm/HIP or Vulkan depending on the verified backend; VRAM measurement via sysfs
  or `rocm-smi` as a fallback.
- On the current host, a Radeon RX 7900 XTX (`gfx1100`), HIP 7.2.26015
  and working ROCm and Vulkan device detection have been verified.
- The user belongs to `video` and `render`. `/dev/dri/card0`,
  `/dev/dri/renderD128` and `/dev/kfd` are present on the host.
  No further group changes are currently needed for this.
- In the agent sandbox the devices are not visible. GPU experiments need
  to run outside this sandbox. Containers also need the
  appropriate device access.

## System tools and operation

| Component | Purpose | State |
|---|---|---|
| CMake and C/C++ compiler | Build of stable-diffusion.cpp and possibly llama.cpp | Present locally; successful backend build still open |
| GPU development libraries | Build of the chosen GPU backend | Depends on the results in M1 |
| Docker Compose or Podman with Compose execution | Alternative to native operation; persistent model store | Docker Compose configuration validated, Podman operation tested with both storage types |
| systemd | Native service operation and restart | Unit at `packaging/eliezer.service` |
| Hugging Face | Model search and explicitly requested downloads | Runtime integration planned for M5 |

The `hf` CLI is an optional tool for manual model checks;
there is so far no requirement to install it for the later Eliezer service.
Models, tokenizers, image projectors, text encoders and VAE will in future reside in the dedicated
model directory or in the persistent container volume. Secrets are kept separately.

The measured results and still-open runtime checks are in
[Runtimes.md](../Runtimes.md). A local inventory without downloads or
model loading is possible with `bash tests/environment/check.sh` from the repository.
