# Burn (feature check, 19.10.2026)

Question: can [Burn](https://crates.io/crates/burn) run our GGUF language models on the AMD GPU,
as an alternative to the embedded llama.cpp?

This is a registry and repository check only. No Burn program was built or run.

## Findings

| Item | Result |
|---|---|
| Version | `burn 0.22.0`, 6 Oct 2026; `MIT OR Apache-2.0`; `rust-version = 1.95` |
| GPU features | `cuda`, `rocm`, `vulkan`, `wgpu`, `metal`, `cpu`. `burn-rocm 0.22.0` is a HIP backend built on `cubecl-hip 0.11.0`. Unlike mistral.rs and Candle, Burn therefore offers an AMD backend. |
| What it is | A general tensor and deep-learning framework (autodiff, training, ONNX import). It is not an LLM runtime. |
| GGUF | Not supported: [issue #1187](https://github.com/tracel-ai/burn/issues/1187) "import weights from GGUF files" is open. |
| Model implementations | The official `tracel-ai/models` repository contains albert, bert, minilm, mobilenetv2, resnet, squeezenet and yolox. There is no first-party LLM. The only Llama implementation found on crates.io is a third-party vendored copy (`maolan-llama 0.0.3`, Llama 3 only). |
| Open ROCm issues | #4202 segfault when `cpu` and `rocm` are enabled together; #5974 docs.rs cannot build `burn-rocm`. Closed: #3463 (Metal LLM inference slow and wrong). |
| Rust version | `rust-version = 1.95` is above the project's `rust-version = "1.88"` (and the Docker build image). |

## Conclusion for Eliezer

Not suitable as the chat runtime now. Our models are quantized GGUF files with architectures that
Burn does not implement (Qwen3 MoE, Qwen3.8 with linear attention, Gemma 4, gpt-oss with MXFP4).
Using Burn would mean writing each architecture, the Q4_K_M and MXFP4 kernels, tokenizers and
sampling from scratch, with no reference results. llama.cpp (`llama-cpp-2` with `rocm`) already loads
these files and was verified on the card in M1 (see [../llamacpp/README.md](../llamacpp/README.md)).

Where Burn could become interesting later:

- **Embeddings:** `bert-burn` and `minilm-burn` exist in pure Rust and could serve small
  embedding models without a C++ toolchain.
- **Pure-Rust fallback:** If Burn gains GGUF import and LLM implementations, re-check this table
  (the `rust-version` conflict has to be resolved first).
