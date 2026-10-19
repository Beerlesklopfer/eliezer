# Runtime checks (M4): llama.cpp through the HTTP API

Manual GPU checks of the embedded llama.cpp runtime with real models. They are not part of
`cargo test` (they need the card and the model files). Everything runs through the public API
of the service, started on a free loopback port with a temporary configuration.

## Scripts

| Script | What it does |
|---|---|
| `live-check.py MODEL.gguf [--tools] [--context N] [--parallel N] --binary PATH` | Load, chat, streaming, usage, stop sequences, `max_tokens`, two parallel requests, prompt-prefix reuse, a prompt longer than the context, the Anthropic endpoint, tool calls (`--tools`), unload with the VRAM check, orderly shutdown. |
| `vram-cycles.py MODEL.gguf` | Five load/unload cycles; how much VRAM stays after each unload? |

Build the binary with a GPU API and keep it in a directory no package build touches, because
`cargo deb` and `cargo build` share `target/release`:

```bash
CARGO_TARGET_DIR=target/gpu CMAKE_POSITION_INDEPENDENT_CODE=ON CMAKE_HIP_FLAGS=-fPIC \
  cargo build --release -p eliezer --features rocm
python3 -I tests/m4/live-check.py /var/lib/eliezer/models/<file>.gguf --binary target/gpu/release/eliezer --tools
```

A binary built without a GPU feature has no model runtime; loading fails with `502 backend_error`.
The other tests of this directory (`template.rs`, `output.rs`, `import*.rs`, `cli_client.rs`)
run without a GPU as part of `cargo test`.

## Results of 19 Oct 2026

Radeon RX 7900 XTX (`gfx1100`, 24 GB), ROCm 7.2.0, llama.cpp from `llama-cpp-2 0.1.160`
(`rocm`), context 8192, two parallel sequences, all layers on the GPU. Models are the Q4_K_M files
of the former LM Studio library (gpt-oss: MXFP4). "Checks" counts the checks of `live-check.py`.

| Model | Checks | Load | Speed¹ | Two requests | Prefix cache (cold / repeat) | Residual VRAM² |
|---|---|---:|---:|---:|---|---:|
| Mistral 7B Instruct v0.3 | 19 / 19 | 1.7 s | 130 tokens/s | 0.33 s | 0.31 / 0.06 s | +0.23 GB |
| olmOCR 2 7B | 19 / 19 | 1.9 s | 124 tokens/s | 0.16 s | 0.28 / 0.06 s | +0.30 GB |
| gpt-oss 20B | 21 / 21 | 4.3 s | 166 tokens/s | 0.51 s | 0.31 / 0.05 s | +0.39 GB |
| Gemma 4 26B A4B | 21 / 21 | 6.6 s | 109 tokens/s | 0.08 s | 0.31 / 0.07 s | +0.43 GB |
| Qwen3.8 27B | 21 / 21 | 6.6 s | 36 tokens/s | 1.35 s | 0.97 / 0.92 s | +0.31 GB |
| Qwen3 Coder 30B A3B | 21 / 21 | 10.2 s | 139 tokens/s | 0.16 s | 0.31 / 0.06 s | +0.25 GB |

¹ 120 tokens including the prompt, temperature 0. ² Over the baseline of the idle card, measured
within 10 s after the unload, inside the running service.

What the checks showed:

- **Tool calls** are detected for Qwen3 Coder and Qwen3.8 (XML), Gemma 4 and gpt-oss (Harmony);
  `finish_reason` becomes `tool_calls`. Mistral and olmOCR have templates without tools.
- **Reasoning models** (gpt-oss, Qwen3.8) answer in `content` and send the thinking separately.
  Stop sequences apply to the visible answer only.
- **Streaming** equals the non-streaming answer at temperature 0; the first frame arrives within
  about 10 ms.
- **Qwen3.8 does not reuse the prompt prefix** (0.97 s vs 0.92 s): it is a hybrid model with
  recurrent layers, which cannot drop the tail of a sequence. The engine then starts the slot from
  scratch, as designed. The other five models reuse it (0.3 s → 0.06 s).
- **VRAM:** unloading returns the model memory every time. A constant 0.2–0.45 GB stays above
  the baseline (the ROCm runtime of the process). Five load/unload cycles of Qwen3 Coder 30B left
  +0.26, +0.35, +0.27, +0.26 and +0.28 GB: it does not grow, and it is gone when the process
  ends. This agrees with the M1 finding (244 MB).
- **CPU for comparison** (Mistral 7B, 32 threads): load 2.1 s, 14 tokens/s, two requests 4.3 s, the
  long prompt 15.5 s.

## Faults the checks found (all fixed)

- A packaged binary built without a GPU feature loads nothing: a plain `cargo deb` now builds the
  ROCm package.
- gpt-oss ends a tool call with `<|call|>`, which llama.cpp also reports as end of generation. The
  engine now hands that token to the parser, so the call completes.
- Stop sequences also matched inside the reasoning; they now apply to the visible answer only.
- A prompt longer than the context window marked the model as crashed; it is now a 400 and the
  model keeps running.
