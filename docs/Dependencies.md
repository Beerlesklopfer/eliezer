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
