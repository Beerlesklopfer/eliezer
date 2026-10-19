# M1 – Feasibility decision

Completed on 19.10.2026, milestone version 0.1.0.
Hardware checked: Radeon RX 7900 XTX (`gfx1100`, about 24 GiB VRAM), Linux,
ROCm/HIP 7.2. The minimum requirement of 8 GB is not a statement that every model
or the image configuration checked here fits into 8 GB.

## Results

| Question checked | Result and evidence |
|---|---|
| Current mistral.rs state | Release and workspace v0.9.5, tag and commit `465191d0b0a7e2432d327b0de06204ab3f94d3a2`; Rust 1.94 at least |
| mistral.rs on AMD | At the state checked, SDK/core offer no ROCm/Vulkan backend. No AMD GPU model run is available through these backends; [source check](../runtime/mistralrs/README.md) |
| Candle before llama.cpp | The exact Candle commit pulled in by mistral.rs also offers CPU/CUDA/Metal, no ROCm/Vulkan; [evidence](../runtime/candle/README.md) |
| Embedded AMD fallback | `llama-cpp-2 0.1.160` with `rocm` built successfully, model loaded and reloaded; [probe program and measurements](../runtime/llamacpp/README.md) |
| Streaming and token counts | Token-by-token output with timestamps and actual token counts; no OpenAI/Anthropic SSE implementation yet |
| Parallelism | Two sequences at the same time in a shared context verified in practice |
| Unloading | Warm model cycles release the model share again. A one-time residual allocation compared with a cold process remains; full return only after the process ends. The restart path remains necessary |
| Image server | HIP build of stable-diffusion.cpp, Qwen Image 2.1 with local text encoder and VAE; [HTTP probe and measurements](../runtime/sdcpp/README.md) |
| Chat → image → chat | Direct trial succeeded; first image request including loading 8.087 s at 512 × 512 and four steps |
| Ready-made manager | The same model switch with llama-swap succeeded; [comparison](../runtime/llama-swap/README.md) |
| Start parameters | Actual help output of Ollama, LM Studio and backend programs checked; [mapping](../reference-clients/README.md) |

The source check does not replace positive runtime evidence for mistral.rs or Candle.
For the existing AMD card this is a negative suitability result; therefore the
embedded fallback provided for in the specification was verified in practice.

## Decision for the implementation

1. Continue with the in-house build. llama-swap works in the comparison, but does not become
   a runtime dependency of Eliezer. Eliezer gets its own manager,
   the VRAM bookkeeping and the LM-Studio-compatible management contract.
2. On AMD, plan for the verified embedded llama.cpp binding. mistral.rs is not
   offered as a GPU runtime on this card. Support for other
   hardware with mistral.rs is not confirmed by this AMD check.
3. Run image models with the verified stable-diffusion.cpp server as a child process.
4. Measure the residual allocation and implement the specified `unload_leak` / `needs_restart`
   path. A guaranteed return to the cold baseline allocation through
   handle drop alone is not demonstrated for the ROCm binding checked.
5. The low-level binding interface requires our own implementation of HTTP/SSE,
   tool calls, thinking options, cancellation and vision. The gaps will be
   implemented and verified in M4; none of these capabilities is claimed as present here.
6. The current mistral.rs release requires Rust 1.94. Eliezer's requirement of 1.88
   is not silently changed. Before this optional backend is adopted, the
   minimum version must be decided explicitly.

## Check status

- Rust probe: format check and Clippy with `-D warnings` green.
- Normal Cargo test run: one GPU test explicitly ignored, no network or GPU access.
- GPU test on request: one test passed; no automatic proof of complete memory release.
- Python check programs syntax-checked, direct switch and proxy comparison executed.
- M1 is a feasibility decision with these limits, not an acceptance of the later service.
  M2 begins only after approval.
