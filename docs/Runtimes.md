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

## stable-diffusion.cpp `sd-server` (checked 2026-10-20, commit b13a482)

Built with HIP for gfx1100 as in `tests/runtime/sdcpp/README.md` (`-DSD_HIPBLAS=ON`), run through Eliezer's own adapter.

- Start: `--diffusion-model FILE`, `--llm FILE` (text encoder; for Qwen Image the Qwen3-VL file), `--vae FILE`,
  `--listen-ip`, `--listen-port`, `--offload-to-cpu`, `--steps N`, `--cfg-scale F`, `--sampling-method NAME`,
  `--log-level warn`. The methods are those of `sd-server --help`.
- Ready: `GET /v1/models` answers 200 (3 s after the start with Qwen Image 2.1 and the weights in RAM).
- Draw: `POST /v1/images/generations` with `prompt`, `n`, `size` (`WIDTHxHEIGHT`) and `output_format` (`png`), answer
  `created`, `output_format` and `data[].b64_json`.
- Settings inside a prompt: `<sd_cpp_extra_args>{...}</sd_cpp_extra_args>` (documented in `examples/server/api.md`)
  sets native options per request; Eliezer removes the name from user prompts.
- Measured with Qwen Image 2.1 Q4_K_M, the int8 encoder and the bf16 VAE, 20 steps, cfg 4: 512x512 in 13 s,
  768x512 twice in 50 s, 1024x1024 in 80 s. The weights stay in RAM (idle VRAM +0.6 GB); the VRAM rose by 6.4 GB at most
  during 1024x1024. A child exits when it is killed and leaves no process behind after unloading or a stop of the service.
- FLUX.2 klein 4B (checked 2026-10-20, as `docs/flux2.md` of the commit describes): `--diffusion-model` the Q8_0 file of
  `leejet/FLUX.2-klein-4B-GGUF`, `--llm` `Qwen3-4B-Q8_0.gguf` of `unsloth/Qwen3-4B-GGUF`, `--vae`
  `split_files/vae/flux2-vae.safetensors` of `Comfy-Org/flux2-dev` (same size as the VAE of the gated
  `black-forest-labs/FLUX.2-dev`, but not gated), `--steps 4 --cfg-scale 1.0`. A fox at 1024x1024 took 12 s (2.4 s per step)
  on the RX 7900 XTX; the picture is sharp and matches the prompt. Fetched with `eliezer download` (hash verified).
- Editing (`POST /v1/images/edits`, multipart: `prompt`, `n`, `size`, `output_format`, `image[]`; checked 2026-10-20): both models
  edit with their normal start arguments, no extra option needed. FLUX.2 klein 4B (4 steps): a 1024x1024 fox made blue in
  27 s including the load. Qwen Image 2.1 UC Q4_K_M with the int8 encoder (20 steps, cfg 4): a 512x512 fox made blue in 35 s
  including the load; the safetensors encoder works for editing (the docs ask for `--llm_vision` only with a GGUF encoder).
- MOSS-Image-SFT (checked 2026-10-20, `docs/moss_image.md` of the commit): `--diffusion-model` `MOSS-Image-SFT-Q8_0.gguf` of
  `leejet/MOSS-Image-SFT-GGUF`, `--llm` `Qwen3.5-9B-Q4_K_M.gguf` of `unsloth/Qwen3.5-9B-GGUF`, `--vae` the FLUX.2 VAE (as
  for FLUX.2 klein), `--tokenizer` the `tokenizer/tokenizer.json` of `OpenMOSS-Team/MOSS-Image-SFT` (the external file is
  required: Qwen3.5 has another vocabulary than the embedded one; `extra.tokenizer`), `--cfg-scale 3.0`, default steps (20).
  A cat with a wooden sign "moss-image" at 1024x1024: the text is rendered correctly; 4 min 17 s including the load, 768x768 in
  2 min 21 s (warm), with the weights in RAM (`offload_to_cpu` default) and 10.6 GB VRAM in use. Two requests at once: the second
  waits inside the program and is answered after the first. Slower than Qwen Image 2.1 (80 s at 1024x1024): MOSS runs two
  passes per step (CFG) and a 9B text encoder.
- The server also offers an asynchronous native API (`/sdcpp/v1/img_gen`, `/jobs/{id}`, cancel); Eliezer does not use it yet.

