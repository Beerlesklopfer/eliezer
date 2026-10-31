# Qwen Image 2.1 via stable-diffusion.cpp

Manual GPU experiment; not part of the ordinary network- and GPU-free tests.
Source: `leejet/stable-diffusion.cpp`, commit
`b13a48219d5cdf408edcb93da6cd69b612ed03d3` of 18.10.2026.
The server help was checked directly after the build.

## Build

Check out the sources including submodules into a local working directory:

```bash
git clone --recurse-submodules https://github.com/leejet/stable-diffusion.cpp.git /tmp/eliezer-m1-sdcpp
git -C /tmp/eliezer-m1-sdcpp checkout b13a48219d5cdf408edcb93da6cd69b612ed03d3
git -C /tmp/eliezer-m1-sdcpp submodule update --init --recursive
cmake -S /tmp/eliezer-m1-sdcpp -B /tmp/eliezer-m1-sdcpp/build -G Ninja \
  -DCMAKE_C_COMPILER=/opt/rocm/llvm/bin/clang \
  -DCMAKE_CXX_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DSD_HIPBLAS=ON -DCMAKE_BUILD_TYPE=Release \
  -DGPU_TARGETS=gfx1100 -DAMDGPU_TARGETS=gfx1100 \
  -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
  -DSD_SERVER_BUILD_FRONTEND=OFF -DSD_WEBP=OFF -DSD_WEBM=OFF
cmake --build /tmp/eliezer-m1-sdcpp/build --target sd-server -j 4
```

For the local build, the ROCm and OpenMP libraries are additionally made available via
`LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/lib/llvm/lib`.
The paths must be adapted to the installation. No system libraries are modified.

## Probe

```bash
LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/lib/llvm/lib \
python3 tests/runtime/sdcpp/probe.py \
  --server /tmp/eliezer-m1-sdcpp/build/bin/sd-server \
  --diffusion /path/qwen-image-2.1-UC-Q4_K_M.gguf \
  --encoder /path/qwen3vl_8b_int8_convrot.safetensors \
  --vae /path/qwen_image_2.1_vae_bf16.safetensors \
  --chat-probe /path/eliezer-m1-probe \
  --chat-model /path/Mistral-7B-Instruct-v0.3-Q4_K_M.gguf
```

The script runs chat → image → chat in sequence, checks the HTTP model list and
the base64 PNG reply, measures the total VRAM with a 50 ms sampling interval and
terminates the server it started. Results, test image and synthetic test logs
end up in `target/` and are not versioned. Verified results are recorded
separately in `results/`. No models are downloaded.

## Verified interface

- Start: `--diffusion-model`, `--llm`, `--vae`, `--listen-ip`, `--listen-port`.
- Memory-saving operation: `--offload-to-cpu`.
- Readiness: `GET /v1/models`, expecting HTTP 200 and a model list.
- Generation: `POST /v1/images/generations`, JSON with `prompt`, `n`, `size`,
  `output_format`; the reply contains `created` and `data[].b64_json`.
- Probe: PNG, 512 × 512, 4 Euler steps, CFG 6. This is a functional test,
  not a quality assessment and no evidence for other resolutions or step counts.

Qwen Image 2.1 uses Qwen3-VL-8B and its own 2.1 VAE. The earlier
Qwen2.5-VL encoder is not an interchangeable alternative for this model.

## Results of 19.10.2026

Complete measurements: [results/2026-10-19.json](results/2026-10-19.json).

- ROCm build and direct HTTP call succeeded.
- Chat → image → chat with local files succeeded; all started processes terminated.
- Server ready to respond after 0.202 s. The model list does not prove loaded weights:
  the image model is loaded only with the first image request.
- First image request including loading: 8.087 s, PNG of 512 × 512 pixels,
  251 054 bytes; four Euler steps, CFG 6.
- Peak allocation of the whole card: 8 374 263 808 bytes at a baseline of
  1 194 037 248 bytes (increase of about 7.18 GB).
- After the image server ended, exactly 1 194 037 248 bytes again in this measurement.
- Both chat process runs succeeded. After the first process ended, the baseline
  allocation was reached again; after the second, about 91 MB above it. The measurement also captures the desktop.
- This specific image trial is not 8 GB evidence: the overall peak exceeds
  8 GB. For the smallest target class, other models or further memory settings are needed.
