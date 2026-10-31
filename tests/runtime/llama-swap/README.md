# Comparison with llama-swap

Manual GPU comparison with the same model files and the same image request
as in the [direct trial](../sdcpp/README.md).
Source revision checked: `a2796fb0306215c6883454a8707dfb22d25c3068` of 14.10.2026.

## Build and invocation

The source revision requires Go 1.27.1. The isolated build uses automatic
Go toolchain selection; cache, toolchain and modules stay under `/tmp`:

```bash
git clone https://github.com/mostlygeek/llama-swap.git /tmp/eliezer-m1-llama-swap
git -C /tmp/eliezer-m1-llama-swap checkout a2796fb0306215c6883454a8707dfb22d25c3068
cd /tmp/eliezer-m1-llama-swap
GOPATH=/tmp/eliezer-m1-go GOMODCACHE=/tmp/eliezer-m1-go/pkg/mod \
  GOCACHE=/tmp/eliezer-m1-go/cache go build -o llama-swap .
```

Invocation from the Eliezer repository:

```bash
LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/lib/llvm/lib \
python3 tests/runtime/llama-swap/probe.py \
  --swap /tmp/eliezer-m1-llama-swap/llama-swap \
  --chat-server /path/llama-server \
  --chat-model /path/Mistral-7B-Instruct-v0.3-Q4_K_M.gguf \
  --image-server /tmp/eliezer-m1-sdcpp/build/bin/sd-server \
  --diffusion /path/qwen-image-2.1-UC-Q4_K_M.gguf \
  --encoder /path/qwen3vl_8b_int8_convrot.safetensors \
  --vae /path/qwen_image_2.1_vae_bf16.safetensors
```

The script generates a configuration with two models, validates it with
`llama-swap -validate` and runs chat → image → chat through the proxy.
The default group planning lets only one model run at a time.
All listeners are restricted to `127.0.0.1`, ports 24606 to 24608.
At the end, the server and child processes are terminated via their own process group.

## Limits of the comparison

For chat, the existing Vulkan `llama-server` from LM Studio is used;
for images, the same HIP `sd-server` as in the direct trial. The model files and
requests are identical; the chat runtimes differ (Vulkan process
versus embedded ROCm binding). The measurement serves the functional and
switching comparison, not an isolated speed assessment of the manager.

Measurements, synthetic logs, configuration and test image are created under `target/`.
Verified result summaries are in `results/`.

## Results of 19.10.2026

Measurements: [results/2026-10-19.json](results/2026-10-19.json).

- Source revision built successfully with the automatically fetched Go toolchain.
- Configuration with two models validated successfully.
- Chat → image → chat through the proxy succeeded, each including loading/switching:
  1.537 s → 8.714 s → 2.090 s.
- Both chat replies report 16 prompt tokens and 32 output tokens.
- Image reply: valid PNG, 251 054 bytes, identical synthetic prompt
  and identical settings as in the direct trial.
- VRAM after the comparison ended: 1 173 655 552 bytes against a baseline of
  1 221 361 664 bytes; no remaining model process found.
- Decision: The ready-made manager handles the verified switch.
  Eliezer remains the in-house build, because the chosen architecture embeds the
  language runtime directly and provides the LM Studio management and its own VRAM bookkeeping.
  This creates no dependency on llama-swap for later operation.
