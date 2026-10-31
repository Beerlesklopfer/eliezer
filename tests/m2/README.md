# M2 – Program framework and installation

## Automated tests

```bash
export PATH="$HOME/.cargo/bin:$PATH"
cargo fmt --all --check
cargo clippy --workspace --all-targets --offline --locked -- -D warnings
cargo test --workspace --offline --locked
```

The dependencies must have been fetched once beforehand (`cargo fetch --locked`).
The tests themselves need no network and no GPU. The separate M1 probe programs
are not members of the product workspace.

| File | Content |
|---|---|
| `config.rs` | Complete configuration example, default values, unknown fields, type errors, quotas, model sources, duplicate keys, line/field and model store |
| `cli.rs` | CLI precedence, exit codes, safe error messages and real symlinks including unresolvable targets |

A `PathResolver` keeps file system resolution out of the pure core logic.
The CLI tests use only their own temporary directories.

## Manual operational checks

The following invocations need a built program framework or
local container operation. They address only loopback and load no models:

```bash
cargo build --offline --locked -p eliezer
python3 tests/m2/service.py
python3 tests/m2/verify-unit.py
podman build -t localhost/eliezer:0.2.0 .
python3 tests/m2/container.py
```

- `service.py`: `/healthz`, invalid and valid SIGHUP, no test values in the log,
  orderly SIGTERM. Valid SIGHUP files are only checked in M2.
- `verify-unit.py`: syntax check via `systemd-analyze verify`. In the temporary
  test copy only the not yet installed program path is replaced.
  No system user or system service is created in the process.
- `container.py`: both model stores, HTTP readiness, container re-creation,
  preservation of own model and state files. Uses a unique
  test project name and afterwards removes only that project's own volumes.

## Results of 19.10.2026

- Workspace with five crates, edition 2024, minimum Rust version 1.88.
- Rust 1.99.0: format, Clippy with `-D warnings` and nine tests succeeded.
- Dockerfile actually built with Rust 1.88.0: minimum version confirmed.
- HTTP/signal check and unit syntax check succeeded.
- Compose configurations validated with Docker Compose v5.6.0.
- Podman 5.4.2 / Podman-Compose 1.3.0: bind mount and named model volume
  including container re-creation verified in practice.
- The user mapping requires `x-podman.in_pod: false` with the existing provider.
- Portable tmpfs: `/run` with sticky bit; the unprivileged entry point creates
  `/run/eliezer` with its own identity and mode 0700. Direct tmpfs options
  `uid` and `gid` were rejected by the existing Podman and are not used.
- GPU additions are verified as a Compose configuration. No GPU backend
  is contained in the M2 image. According to the installed Podman documentation,
  the rootless GPU variant requires `crun`; on the host checked only `runc` is available.
  A GPU container run is therefore not claimed as demonstrated here.

Result files are in [results/](results/). Installation and permissions are
described in [INSTALL.MD](../../INSTALL.MD).

## Scope

M2 loads no models. Authentication, manager, model routes and adoption of
new configurations follow in M3; real backends in M4/M6. The `/healthz` endpoint,
brought forward, serves for the readiness check of the program framework.
