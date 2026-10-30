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

## Model store outside `/var/lib/eliezer`

By default the models live in `/var/lib/eliezer/models`. If they live elsewhere, for example on a big
disk under `/home`, name the real directory in the configuration (`storage.models_dir` and
`huggingface.cache_dir`, in that order of sections: `[storage]` and `[huggingface]`).

- **Reading** needs nothing more: the unit makes `/home` read-only instead of hiding it.
- **Writing** (model downloads and state files, later milestones) needs the directory opened with
  `ReadWritePaths` in a drop-in; `packaging/eliezer-models-in-home.conf.example` (installed under
  `/usr/share/doc/eliezer/examples/`) shows how.
- `/var/lib/eliezer` may be a symbolic link or a mount point. The unit does not use systemd's
  `StateDirectory=` (which refuses a symbolic link and fails at start with `Failed at step
  STATE_DIRECTORY ... Too many levels of symbolic links`); the package only creates the directory when
  nothing is there. The service user needs permission on the target: read for the models, write for
  the state files.

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

The command also enters the language and embedding models into the configuration (and enables
`[backend.llamacpp]` if needed). It validates the result exactly like the service does and writes it
only if it is valid, keeping the old file as `config.toml.bak`; use `--no-config` to get the blocks
printed instead. Image models are moved but only printed as a commented block (their runtime is not
connected yet). Review the added entries, then apply them with `systemctl reload eliezer`.

## Command line client

The `eliezer` program is also a client of the running service, in the style of `ollama` and `lms`.
It talks to the HTTP API; the address and the API key are taken from the configuration
(`server.listen`, `server.api_keys_file`) if it is readable, so on the machine that runs the
service no options are needed:

```bash
eliezer ls                          # configured models: key, type, size, max context, state
eliezer ps                          # loaded models: VRAM, context, parallel, idle time, requests
eliezer status                      # service, GPU memory (total/used/free/budget), queue
eliezer show qwen/qwen3-coder-30b   # details of one model and the endpoints serving it
eliezer load qwen/qwen3-coder-30b --context 32768 --parallel 2
eliezer unload qwen/qwen3-coder-30b # or: eliezer unload --all
eliezer run qwen/qwen3-coder-30b "Explain mutexes in two sentences" --stats
eliezer run qwen/qwen3-coder-30b    # interactive chat; an empty line or Ctrl-D ends it
echo "Summarize this" | eliezer run qwen/qwen3-coder-30b
```

Reading a text aloud (needs a `tts` model and, for a cloned voice, a voice added with `eliezer voices add`):

```bash
eliezer voices add mama mama-sample.wav --transcript "Es war einmal."   # the voices are listed by: eliezer voices
eliezer speak --voice mama --file kapitel.txt | sox -t wav - -t mp3 kapitel.mp3
eliezer speak --voice mama --file kapitel.txt | play -t wav -           # listen while it is read
eliezer speak --voice mama --file kapitel.txt --out kapitel.wav         # a plain WAV file
mkfifo /tmp/speech.fifo; sox -t wav /tmp/speech.fifo -t flac kapitel.flac &   # a named pipe
eliezer speak --voice mama --file kapitel.txt --out /tmp/speech.fifo
```

The audio is written while the text is read, passage by passage, so a player starts after the first sentences. Eliezer
produces `wav` and raw `pcm` (`--format pcm`) only: **mp3, flac and other formats are made by sox or ffmpeg**, not by
Eliezer. On a pipe the WAV has an open length, which sox, ffmpeg and mpv read until the end; a file named with `--out`
gets the real sizes. If a passage fails after the audio has started, the command exits with 1 (a file is not left behind).

`run` prints the answer while it is generated. Reasoning ("thinking") text is dimmed on a terminal
and goes to stderr when the output is piped, so stdout holds only the answer. `--stats` prints the
token statistics to stderr. Further options: `--system TEXT`, `--temperature F`, `--max-tokens N`,
`--reasoning none|low|medium|high`.

Options for every command: `--host URL` (otherwise `ELIEZER_HOST`, the configured listen address,
or `http://127.0.0.1:1234`), `--key-file PATH` (otherwise `ELIEZER_API_KEY_FILE`, `ELIEZER_API_KEY`
or the key file named in the configuration), `--json` (print the JSON of the API) and
`--config PATH`. The key is never accepted as a command-line argument and never printed.
`ELIEZER_TIMEOUT_SECS` changes how long a command waits for an answer (default 15 s for queries, one
hour for `load` and `run`). If a server accepts the connection but does not answer, the message says
it may be stopped (for example with Ctrl-Z), overloaded or hung.

From another machine: `ELIEZER_HOST=http://server:1234 ELIEZER_API_KEY_FILE=~/.eliezer-key eliezer ls`.

Exit codes: 0 ok, 1 request failed or server error (the message of the service is printed), 2 usage
error, 3 service not reachable (start it with `systemctl start eliezer`).

## Debian package

The package is built with `cargo-deb` (`cargo install cargo-deb`, Rust from rustup). It contains
the model runtime of one GPU API: `rocm` (the default), `vulkan`, `cuda`
or `cpu`. A package without a runtime would start but could not load any model, so there is none.

```bash
scripts/build-deb.sh            # rocm when ROCm is installed, otherwise cpu; or name the variant
sudo apt install ./target/debian/eliezer_0.6.0-1_amd64.deb
```

Build the distributable package with `scripts/build-deb.sh`, not a bare `cargo deb`.
The script bundles the ROCm runtime closure, HIP runtime compiler and rocBLAS/hipBLASLt
kernel assets under `/usr/lib/x86_64-linux-gnu/eliezer/` (the native Debian multiarch
triplet is detected at build time). Every bundled ELF uses a package-relative search
path; the installed binary does not require `/opt/rocm` or `LD_LIBRARY_PATH`.
License notices and a library manifest are installed under `/usr/share/doc/eliezer/`.
The package is several GB because GPU kernel data is included. Standard Debian
libraries remain apt dependencies; the kernel driver and GPU device permissions
remain host requirements.

For building the ROCm 7.2 variant, install AMD's matching SDK on the build machine
(default `/opt/rocm`, overridden by `ROCM_PATH`), plus `python3`, `patchelf`,
`binutils` and `dpkg-dev`. Debian 13's native HIP development packages are from 5.7
and cannot substitute for the ROCm 7.2 ABI used by this build. Debian supplies the
normal compiler/CMake/Clang prerequisites and the Vulkan development packages.
A build exclusively against Debian's older ROCm SDK has not been validated.

It installs `/usr/bin/eliezer`, the unit `eliezer.service`, a default
`/etc/eliezer/config.toml` (no models, loopback only, kept on upgrades) and the example
configuration under `/usr/share/doc/eliezer/examples/`. The system user `eliezer` is created.
The service is neither enabled nor started: review the configuration first (set `vram_budget_gb`
to a little less than the card's memory; the default of 8 GB refuses larger models), then run
`sudo systemctl enable --now eliezer` and check it with `eliezer status`. Removing the package keeps the configuration; purging
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

The unit creates `/run/eliezer`; the state directory `/var/lib/eliezer` is created by the package (or
by the `install -d` command above). Write access is limited to those two directories, and `/home` is
read-only for the service. For a model store that the service must write to outside
`/var/lib/eliezer`, add `ReadWritePaths` in a drop-in (see "Model store outside `/var/lib/eliezer`").

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

There are two different secrets:

| Secret | Direction | Where it lives |
|---|---|---|
| **API key** (an OpenAI-style token `sk-...`) | clients such as MultiGPT send it **to** Eliezer | `server.api_keys_file` (service); `~/.config/eliezer/api_key` (the user's copy for the command line client) |
| **Hugging Face token** (`hf_...`) | Eliezer sends it **to** Hugging Face to download models (M5) | `huggingface.token_file` (service); `~/.config/eliezer/hf_key` (the user's copy, the source of `--set-hf-token`) |

Both service files are mode `0600` and belong to the service account (`server.run_user`). Neither
secret is ever a command-line argument, a log line or part of the configuration.

### API key (the OpenAI token)

Without `server.api_keys_file` the API is open, so restrict the address to the local machine in
that case. Create the first key (as root, because it writes below `/etc/eliezer`):

```bash
sudo eliezer --new-key --name multigpt --save    # prints the key once
sudo systemctl reload eliezer                    # the running service reads the new key
eliezer --list-keys                              # names and masked keys, never the keys
sudo eliezer --revoke-key multigpt               # then reload again
```

`--new-key` makes a random key (256 bits), stores it in the key file (named in the configuration,
otherwise `keys` next to the configuration file, which it then enters into the configuration),
and shows it **once**. `--save` also stores a copy in `~/.config/eliezer/api_key` of the user who
ran `sudo`, which the command line client picks up (it never overwrites an existing file). Several
keys can exist at the same time, one per client. Enter the key as the **API key** (the "OpenAI token")
in the client, for example MultiGPT's provider settings; it sends it as `Authorization: Bearer <key>`.
Anthropic clients may use the `x-api-key` header. A wrong or missing key is answered with
`401` and `code: invalid_api_key`; `/healthz` stays open. If the last key is revoked the service
refuses to start until a new one exists (it fails closed).

### Hugging Face token

```bash
hf auth login                  # once, as usual; this creates ~/.cache/huggingface/token
sudo eliezer --set-hf-token    # copies it to the service
```

The token is never an argument. It is taken from the first of these that has one: standard input
(`... | sudo eliezer --set-hf-token`), `~/.config/eliezer/hf_key`, the default file of the Hugging
Face tools (`$HF_TOKEN_PATH`, `$HF_HOME/token`, otherwise `~/.cache/huggingface/token`), and
finally a prompt without echo. Under `sudo` these are the files of the user who ran `sudo`. The
command says which one it used and warns when that file can be read by other users (`chmod 600`).
It must look like `hf_...`; it is stored in `huggingface.token_file` (default `hf-token` next to
the configuration). Model downloads that use it arrive with M5.

Tests and measured results: [tests/README.md](tests/README.md).
