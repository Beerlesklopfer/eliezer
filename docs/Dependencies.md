# Dependency Decisions

The overview of tools, runtimes, GPU requirements and access permissions
is in the [project wiki](wiki/Dependencies.md).

Concrete crate versions and additional dependencies have not been decided yet.
When they are adopted, version, purpose and rationale are documented here.

## Reviewed mistral.rs upstream

The commit retrieved directly,
[`465191d0b0a7e2432d327b0de06204ab3f94d3a2`](https://github.com/EricLBuehler/mistral.rs/commit/465191d0b0a7e2432d327b0de06204ab3f94d3a2),
carries workspace version **0.9.5** and requires **Rust 1.94**.
The latest published GitHub release is
[0.9.5](https://github.com/EricLBuehler/mistral.rs/releases/tag/v0.9.5)
of 19.10.2026; its Git tag points to the same commit.
A dependency on this release has not been fixed yet.
The minimum version 1.88 from the Eliezer requirements is not sufficient for this upstream;
this difference must be decided before adoption.

## Probe dependency for AMD

`llama-cpp-2 = 0.1.160` with feature `rocm` was chosen for the isolated M1 probe:
It offers the required embedded access to llama.cpp and
supports the existing AMD card via HIP. The transitive `llama-cpp-sys-2`
compiles C++ code; CMake, a C/C++ compiler and ROCm development libraries are
build prerequisites. The probe has its own `Cargo.lock` under `tests/`.
There is no corresponding dependency in a production workspace yet.

Integration decision and limits: [M1 result](../tests/m1/README.md).

## M2 – Program frame and configuration

| Crate | Pinned version | Rationale |
|---|---|---|
| `serde` | 1.0.229 | Typed, strictly checked configuration |
| `toml` | 0.9.8 | TOML parser with error positions; compatible with Rust 1.88 |
| `serde_path_to_error` | 0.1.20 | Field paths for type errors, without copying configuration values into messages |
| `tokio` | 1.53.3 | Asynchronous HTTP program frame, signals and orderly shutdown |
| `axum` | 0.8.9 | Intended HTTP server; initially only `/healthz` |
| `tracing` | 0.1.44 | Structured operational events |
| `tracing-subscriber` | 0.3.23 | Output of the events to stdout/journald |

The concrete versions and transitive dependencies are in `Cargo.lock`.
The CLI parser and the filesystem resolution need no additional crates.
`reqwest`, GPU bindings and model runtimes are added only when they are
actually used in later milestones.
The product license is **GPL-3.0-only**, per the user's decision.

## M3 – NVML measurement

| Crate | Pinned version | Rationale |
|---|---|---|
| `nvml-wrapper` | 0.13.0 | Measures VRAM through NVML (specification 6a) without starting `nvidia-smi` for every query: no process spawn, no text parsing, exact byte values. The library `libnvidia-ml` is loaded dynamically at runtime, so the binary builds and runs without the NVIDIA driver (the probe then reports `Unavailable` and the fallback chain continues with `nvidia-smi`). Pulls in `nvml-wrapper-sys`, `libloading`, `bitflags`, `thiserror` and `static_assertions`. Licence: MIT OR Apache-2.0. Declared `rust-version` 1.60, compatible with 1.88. |

## M4 – llama.cpp runtime: chat templates

| Crate | Pinned version | Rationale |
|---|---|---|
| `minijinja` | 2.24.0 | Renders the Jinja chat template stored in the GGUF (llama.cpp's Rust binding offers only a basic template without tools). A pure-Rust engine with few dependencies (`serde`, `memo-map`; with `json` also `serde_json`, with `preserve_order` also `indexmap`), no C code and no Python. Features: `json` (the `tojson` filter, replaced by our own variant with Python spacing and no HTML escaping), `loop_controls`, `preserve_order` (keeps the order of tool parameters as in the request, which the models saw in training). Licence: Apache-2.0. Declared `rust-version` 1.70, compatible with 1.88. Version 3.0.x was released less than two weeks before the decision, so the latest older stable release 2.24.0 was chosen. |
| `minijinja-contrib` | 2.24.0 | Feature `pycompat`: Python string, dict and list methods (`startswith`, `strip`, `split`, `items`, `get`, ...) that Hugging Face templates call. Same author and release train as `minijinja`, no further dependencies. Licence: Apache-2.0. Declared `rust-version` 1.70. |

`serde` (already a workspace dependency) is used for the `tojson` serializer; `serde_json` gets the
feature `preserve_order` in `eliezer-llamacpp` (pulls in `indexmap`, MIT OR Apache-2.0), so that JSON
objects keep their insertion order when they are rendered into prompts. `strftime_now` is
implemented with `std` only (UTC), so no date crate is needed.
