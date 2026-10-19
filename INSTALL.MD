# Installing Eliezer

Eliezer is licensed under **GPL-3.0-only**; the full license text is in [LICENSE](LICENSE).

The service checks the configuration, manages the models of the configuration
(load, unload, queue, eviction, idle unloading), measures GPU memory, authenticates
requests and serves the OpenAI- and Anthropic-compatible interfaces. It handles
SIGTERM and SIGHUP. The model runtimes (mistral.rs, llama.cpp, stable-diffusion.cpp)
are not connected yet (M4 to M6): loading a model fails with `502 backend_error`
until then. The runtime probe programs under `tests/` are standalone experiments.

## Requirements

- Linux; for later model operation a supported GPU with at least 8 GB VRAM.
- For building: Rust stable 1.88 or later and Cargo. The toolchain file selects `stable`;
  the declared minimum version remains 1.88.
- For containers: Docker with Compose or Podman with a Compose provider.
- A dedicated model directory or a persistent container volume.

Details on GPU runtimes and permissions: [Runtime requirements](docs/Runtimes.md).
The pure M2 scaffold needs no GPU. GPU development libraries
become necessary only when building the corresponding model backends.

## Building from source

In the checked-out repository:

```bash
export PATH="$HOME/.cargo/bin:$PATH"
rustc --version
cargo --version
cargo build --release --locked -p eliezer
```

The program is then located at `target/release/eliezer`.
If Rust is still missing, install it via the [official Rust site](https://www.rust-lang.org/tools/install).
On a machine with Rustup, Eliezer uses the user installation.

After fetching the dependencies once, the development checks can also be run
without network:

```bash
cargo fetch --locked
cargo fmt --all --check
cargo clippy --workspace --all-targets --offline --locked -- -D warnings
cargo test --workspace --offline --locked
```

GPU probe programs are not started in the process.

## Checking the configuration and starting locally

A complete example configuration is at
[packaging/config.example.toml](packaging/config.example.toml).

```bash
cargo run --locked -- --check-config packaging/config.example.toml
```

On success, `Configuration OK` appears. Invalid configurations
end with exit code 2 and a message with line and field.

For a user installation, the executable can be installed locally:

```bash
install -Dm755 target/release/eliezer "$HOME/.local/bin/eliezer"
mkdir -p "$HOME/.config/eliezer" "$HOME/.local/share/eliezer/models"
cp packaging/config.example.toml "$HOME/.config/eliezer/config.toml"
```

In this copy, set the following paths to the absolute model directory
actually used:

- `storage.models_dir`, for example `/home/NAME/.local/share/eliezer/models`.
- `huggingface.cache_dir`, for example `/home/NAME/.local/share/eliezer/models/hf`.
- All `source.file`, text encoder and VAE paths below the model store.

TOML files replace neither `$HOME` nor `~` with the home directory.
Absolute paths must be entered. Existing symlinks are resolved during the check;
even through them, model paths must not leave the shared store.
Download targets that do not exist yet are allowed within this store.

```bash
"$HOME/.local/bin/eliezer" --config "$HOME/.config/eliezer/config.toml" --check-config
"$HOME/.local/bin/eliezer" --config "$HOME/.config/eliezer/config.toml"
```

In a second terminal:

```bash
curl --fail http://127.0.0.1:1234/healthz
```

Expected response: `ok`. Ctrl+C shuts the service down in an orderly way.
As an alternative to the argument, `ELIEZER_CONFIG` can specify the configuration path;
`--config` takes precedence. With neither, `/etc/eliezer/config.toml` applies.

SIGHUP reloads the configuration and the key file. New models appear, changed
priorities, pins, memory budget and scheduler settings take effect, and models that were
removed from the file refuse new requests and are unloaded as soon as they are idle.
Loaded models stay loaded. The listen address changes only with a restart (the service
logs `restart_required`). An invalid configuration or a key file that is readable by
others is rejected with a log line (`config_rejected`); the service keeps running with
the previous state.

## Importing existing models (for example from LM Studio)

Set the service account in the configuration so imported files belong to it:

```toml
[server]
run_user = "eliezer"
run_group = "eliezer"   # optional; default: the primary group of run_user
```

Then move the GGUF files into the model store (needs root to hand the files over):

```bash
sudo eliezer --config /etc/eliezer/config.toml --import ~/.lmstudio/models --dry-run
sudo eliezer --config /etc/eliezer/config.toml --import ~/.lmstudio/models
```

`--dry-run` only shows the plan, `--copy` keeps the originals. The directory structure is kept,
existing files are never overwritten, and a rename within one file system takes no extra space.
At the end the command prints a `[[model]]` block per model; check it and add it to the
configuration, then reload with `systemctl reload eliezer`.

## Debian package

The package is built with `cargo-deb` (`cargo install cargo-deb`, Rust from rustup):

```bash
scripts/build-deb.sh
sudo apt install ./target/debian/eliezer_0.3.0-1_amd64.deb
```

It installs `/usr/bin/eliezer`, the unit `eliezer.service`, a default
`/etc/eliezer/config.toml` (no models, loopback only, kept on upgrades) and the example
configuration under `/usr/share/doc/eliezer/examples/`. The system user `eliezer` is created.
The service is neither enabled nor started: review the configuration first, then run
`sudo systemctl enable --now eliezer`. Removing the package keeps the configuration; purging
removes it. The user and `/var/lib/eliezer` (models) are always kept.
`tests/deb/check-package.sh` checks this life cycle in a clean Debian container.

## Running as a systemd service

The following setup steps require administrative privileges. Create the user
`eliezer` only if it does not exist yet:

```bash
sudo useradd --system --user-group --home-dir /var/lib/eliezer --shell /usr/sbin/nologin eliezer
sudo usermod -aG video,render eliezer
sudo install -Dm755 target/release/eliezer /usr/local/bin/eliezer
sudo install -d -m750 -o root -g eliezer /etc/eliezer
sudo install -d -m750 -o eliezer -g eliezer /var/lib/eliezer/models
sudo install -m640 -o root -g eliezer packaging/config.example.toml /etc/eliezer/config.toml
sudo install -m644 packaging/eliezer.service /etc/systemd/system/eliezer.service
```

The device groups must exist on the respective system and match the
device permissions. If the group names differ, also adjust
`SupplementaryGroups` in the unit. The permissions of the logged-in user
do not replace those of the service user.

Before starting, adapt the configuration to models and hardware, then:

```bash
sudo -u eliezer /usr/local/bin/eliezer --config /etc/eliezer/config.toml --check-config
sudo systemctl daemon-reload
sudo systemctl enable --now eliezer.service
systemctl status eliezer.service
journalctl -u eliezer.service
```

The unit creates `/var/lib/eliezer` and `/run/eliezer` with suitable ownership.
Write access is limited to the intended directories. For a
model directory outside `/var/lib/eliezer`, add its write permission with
`ReadWritePaths`. If it is under a home directory, `ProtectHome`
must also be adjusted accordingly.

## Container: model directory as a bind mount

The supplied [container configuration](packaging/config.container.toml)
starts the scaffold without models and without GPU. It is mounted read-only.
The port is by default reachable on the host only at `127.0.0.1:1234`.

```bash
mkdir -p models
docker compose build
docker compose up -d
curl --fail http://127.0.0.1:1234/healthz
docker compose logs eliezer
docker compose down
```

By default the container runs with UID/GID `10001:10001`. The host model directory
must be readable for this container identity and writable for downloads.
The actual host mapping depends on whether the container engine runs rootless.
An existing model inventory should be given suitable permissions beforehand.

Configurable are:

| Variable | Meaning | Default |
|---|---|---|
| `ELIEZER_CONFIG_FILE` | Host file for the configuration | `./packaging/config.container.toml` |
| `ELIEZER_MODELS_DIR` | Host directory for models | `./models` |
| `ELIEZER_PORT` | Host port published on `127.0.0.1` | `1234` |
| `ELIEZER_UID`, `ELIEZER_GID` | Identity of the container process | `10001` each |

With a different UID/GID, the volume and runtime directory permissions must match
as well. For the supplied rootless Podman variant, keep the default `10001:10001`;
its user mapping maps the calling user onto it.

For rootless Podman, add the user mapping:

```bash
podman-compose -f compose.yaml -f compose.podman.yaml up -d --build
podman-compose -f compose.yaml -f compose.podman.yaml down
```

For the M2 scaffold this variant needs neither GPU devices nor `crun`.
With the Podman Compose provider, the addition also disables the automatic
pod creation so that the user mapping works.

## Container: persistent model volume

The supplementary file replaces the model bind mount with a named volume:

```bash
docker compose -f compose.yaml -f compose.volume.yaml up -d --build
docker compose -f compose.yaml -f compose.volume.yaml down
```

Both the model volume and the state volume are preserved on `down` and when the
container is re-created. `down --volumes` would delete this data.
With the bind mount, the models remain in the host directory independently of the container.
A new model volume initially contains no models.

## GPU access in the container

The following additions pass devices through. The image does not yet contain a
model backend or ROCm/CUDA runtime libraries; device access alone
therefore does not enable inference yet.

For Docker on AMD, determine and set the actual device groups:

```bash
export ELIEZER_RENDER_GID="$(getent group render | cut -d: -f3)"
export ELIEZER_VIDEO_GID="$(getent group video | cut -d: -f3)"
docker compose -f compose.yaml -f compose.amd.yaml up -d
```

For GPU access under rootless Podman, the supplied addition is designed for `crun` and its
`keep-groups` support. The user must have access on the host
to `/dev/dri` and, for ROCm/HIP, to `/dev/kfd`. `crun` must be installed for this; the addition explicitly selects it as the runtime.

```bash
podman-compose -f compose.yaml -f compose.podman.yaml -f compose.amd.podman.yaml up -d --build
# Alternatively, additionally with a model volume:
podman-compose -f compose.yaml -f compose.volume.yaml -f compose.podman.yaml -f compose.amd.podman.yaml up -d --build
```

`compose.amd.yaml` is the Docker GPU variant, `compose.amd.podman.yaml` the rootless
Podman GPU variant; they are not combined with each other.
For NVIDIA a suitable NVIDIA container integration is required; an
NVIDIA Compose variant was not tested on the existing AMD hardware.

## Token and API keys

Secrets belong neither in the repository nor in images, Compose files or
command lines. Token and API key files configured later must be owned by the
service user and have mode `0600`. They are located outside the model store
and are mounted separately read-only in the container.

Authentication is active as soon as `server.api_keys_file` is set; without it the API
is open, so restrict the published address to the local machine in that case. Token
retrieval from Hugging Face is not implemented yet (M5).

Tests and measured results: [tests/README.md](tests/README.md).
