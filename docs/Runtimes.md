# Runtime Requirements

Target hardware: at least 8 GB VRAM; larger models need correspondingly more memory.

## Mandatory requirements (must-haves)

| Area | Requirement |
|---|---|
| Operating system | Linux with a working GPU driver |
| Graphics card | An NVIDIA or AMD GPU with at least 8 GB VRAM; model, quantization and context fit into the available memory budget |
| GPU runtime | Acceleration matching the chosen backend: CUDA for NVIDIA, verified ROCm/HIP or Vulkan support for AMD |
| Device access | The service user can access the GPU; on AMD the required DRM devices and, for ROCm/HIP, additionally `/dev/kfd`. Groups and ACLs must allow this access |
| Container operation | When using Docker/Podman, GPU devices and required driver libraries are available in the container |
| Model store | Own, configurable directory for all model components; in the container as a bind mount or persistent volume |
| Model inventory | Only configured models are loaded; required tokenizers, image projectors, text encoders and VAE are fully present |
| Memory management | Loaded models can be unloaded; the release of VRAM is measured |
| Secrets | Tokens and API keys come from protected files (0600) or the environment and do not appear in logs or process arguments |

## Permission management

Eliezer runs as its own unprivileged service user `eliezer`. For a
user installation the following requirements apply accordingly to the
executing user. The running service needs neither `root` nor `sudo`.
Setting up the user, groups and system directories is up to the administration.

| Resource | Required permissions |
|---|---|
| GPU devices | Access via the actual device groups or suitable ACLs. On AMD, `render` and possibly `video` are particularly relevant; ROCm/HIP additionally needs access to `/dev/kfd` |
| Configuration | Readable by the service, changeable only by the administration; all parent directories must be searchable |
| HF token and API key file | Owned by the executing service user, file mode `0600`; other users get no access |
| Model directory | Read and search for inference; additionally write for downloads and cache management |
| State directory `/var/lib/eliezer` | Read, write and search for the service, in particular for `pins.json` and `measured.json` |
| Runtime directory `/run/eliezer` | Writable by the service, in particular for PID files; provide at service start |
| Backend programs and libraries | Executable or readable by the service; not replaceable through HTTP requests |

Group memberships apply to the service process as actually started.
After a change, it must be restarted with the new groups.
The permissions of the logged-in user alone do not prove the permissions of a separately
set-up service user. Devices are not released blanket-wise to all users.

### API access and model selection

- If `server.api_keys_file` is set, the protected endpoints require a
  configured API key; `/healthz` remains reachable without authentication.
- Without a key file, the API is reachable without authentication. The configured
  bind address and network exposure determine who can reach it.
- Only models on the allowlist are executable. Model keys from requests
  are checked against this list; they do not grant arbitrary file access.
- User roles, individual model permissions and billing are managed by MultiGPT.
  The Eliezer API keys do not distinguish roles; whoever gets direct API access
  also gets access to the protected management endpoints.

### Container operation

The container process receives the required GPU devices and device groups.
UID/GID mapping, supplementary groups and, where applicable, ACLs must allow access to devices and mounted files even with
rootless Docker/Podman operation.
The check is made with the identity of the container process.

Configuration and secret files are mounted read-only.
Model and state volumes must have suitable permissions for the container identity;
the secret files must keep their restrictive access permissions.
Container operation should work without a privileged container.

### Native service operation

The systemd unit uses `User=eliezer`, the required device groups,
`NoNewPrivileges` and `ProtectSystem=strict`. Write access is limited specifically to
the model, state and runtime directories. A differing
model directory must be explicitly taken into account.

## Additionally when building from source

- Rust stable 1.88 or later with Cargo, edition 2024; rustfmt and Clippy for the project checks.
- The development libraries and GPU tools required for the chosen backend.
- CMake and a C/C++ compiler when building stable-diffusion.cpp or llama.cpp.

Concrete backend versions and supported GPU combinations are documented as supported only after a
confirmed build and model run. An installed library
or successful device detection alone is not sufficient for that.

## Hugging Face metadata API (probed 2026-10-20)

Checked against the live service with `curl`, without a token; used by the search (`eliezer-hub`).

- `GET https://huggingface.co/api/models?search=TEXT&filter=gguf&filter=license:LICENSE&pipeline_tag=TAG&sort=downloads&direction=-1&limit=N`
  answers a JSON array. Each item has `id` (`Organisation/Repository`), `downloads`, `likes`, `tags`
  (including `gguf` and `license:<tag>`), often `pipeline_tag` (`text-generation`, `image-text-to-text`,
  `text-to-image`, `feature-extraction`), `createdAt`. It lists no files.
- `GET https://huggingface.co/api/models/{org}/{repo}?blobs=true` answers an object with `sha` (the commit),
  `pipeline_tag`, `tags`, `gated` (`false` or a string), `downloads`, `gguf` (`architecture`, `total`,
  `totalFileSize`) and `siblings[]` with `rfilename`, `size` and `lfs.sha256`. `gguf.architecture` is
  missing for some repositories (the support is then reported as `unknown`).
- Seen: `lmstudio-community/olmOCR-2-7B-1025-GGUF` (architecture and license in the tags);
  `abenzerps/Qwen-Image-2.1-Uncensored-Quantized` (sha `7959e658915f52274897fb573ffeda49de8dc16b`,
  `gguf.architecture` = `qwen_image21`, files `text_encoders/…` and `vae/…`).
- Status codes handled: 401, 403 and 404 as not found, 429 as busy, others as an unexpected answer.
