# Embedded llama.cpp with ROCm

Manual GPU experiment with `llama-cpp-2 = 0.1.160`, feature `rocm`.
The probe program is independent of the later Eliezer workspace and is not started
by ordinary tests. It requires an existing local GGUF model.
It performs neither downloads nor HTTP requests.

## Scope

Check GPU offload, load the model with all layers, generate 32 tokens,
drop context and model, observe memory release for up to ten seconds
and repeat the sequence in the same process. What is measured is the total
VRAM allocation of the card via AMD sysfs; foreign processes can influence the values.
The observation uses a tolerance of 100 MB relative to the previous allocation.
Two parallel sequences are checked; tool calls, vision and parallel
HTTP requests are therefore not yet demonstrated.

## Build and run

For the architecture `gfx1100` checked and a local ROCm installation:

```bash
export PATH="$HOME/.cargo/bin:$PATH"
export LD_LIBRARY_PATH=/opt/rocm/lib
CARGO_BUILD_JOBS=4 AMDGPU_TARGETS=gfx1100 \
  CMAKE_POSITION_INDEPENDENT_CODE=ON CMAKE_HIP_FLAGS=-fPIC \
  cargo build --locked --manifest-path tests/runtime/llamacpp/Cargo.toml

tests/runtime/llamacpp/target/debug/eliezer-m1-probe /path/to/model.gguf
```

Device access must be available to the executing process. Running
in a sandbox without GPU devices says nothing about host support.
The measurement source `card0` used in the program must match the GPU under test.

## First trial of 19.10.2026 (before the extension)

- First build without PIC flags: HIP objects could not be linked into the
  Rust program (`R_X86_64_32` and `R_X86_64_32S` respectively).
- Fresh build with `CMAKE_POSITION_INDEPENDENT_CODE=ON` and
  `CMAKE_HIP_FLAGS=-fPIC` succeeded. Rust program and llama.cpp are embedded;
  no `llama-server` was started for this model trial.
- `cargo fmt --check` and `cargo clippy --all-targets -- -D warnings` succeeded.
- Manual GPU run with local `Mistral-7B-Instruct-v0.3-Q4_K_M.gguf` succeeded:
  two load/answer/unload cycles in the same process, exit code 0.
- First load: 1 467 ms; 33/33 layers on the GPU, model buffer 4 097.52 MiB.
- Context 512 tokens; prompt 16 tokens; output 32 tokens each.
  Time to first token: 57 ms and 23 ms respectively.
  Total generation time including prompt processing: 299 ms and 255 ms respectively.
- VRAM before the first load: 1 377 079 296 bytes, with model 5 821 272 064 bytes,
  with context 6 094 217 216 bytes. After the first unload: 1 620 803 584 bytes
  after 10 008 ms; the 100 MB tolerance relative to the baseline allocation was not reached.
- Second cycle: with context 6 109 556 736 bytes, after unload 1 620 811 776 bytes
  after 100 ms. Reloading and answering work.
- The first residual difference is about 244 MB. The cause has not yet been established
  (the whole card was measured). Complete memory release is therefore
  not demonstrated. The probe program logs the values; exit code 0
  is not an automatic acceptance proof for memory release here.
- When CMake options change, previously generated backend objects must be
  rebuilt; the binding does not reliably detect these environment changes.

## Extended check and final finding

The current experiment uses the chat template from the GGUF, prints tokens with
timestamps immediately and, in the third run, operates two sequences
in a shared context of 1 024 tokens. The context budget is shared.

Measurement logs: [three runs](results/2026-10-19.txt) and
[control with explicit HIP initialization](results/2026-10-19-hip-init.txt).
The explicit initialization uses `hipFree(NULL)` in accordance with the
installed HIP header. The only FFI call is justified in the source code.

| Run | Load time | Prompt per sequence | Output | First token | Time including prompt | Output tokens/s including prompt |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 292 ms | 16 | 32 | 57 ms | 301 ms | 106.3 |
| 2 | 267 ms | 16 | 32 | 23 ms | 257 ms | 124.5 |
| 3, two sequences | 262 ms | 16 | 2 × 32 | 48 ms each | 317 ms | 201.9 total |

After the warm runs, 4 096 and 0 bytes respectively remain in addition to the
respective baseline value. The cold baseline allocation, however, is not fully
reached again within the process, not even after backend drop.
The additional HIP initialization does not remove this finding. The exact
cause of the one-time residual allocation is not proven. After the process ends,
the baseline allocation is reached again in the switching tests, or the
deviation lies within the observed foreign allocation.

**Conclusion:** Loading, token output, two sequences and reloading work.
There is no guarantee of complete release down to the cold baseline allocation;
the `unload_leak` / `needs_restart` path required by the specification
remains necessary for the later embedding. The GPU check is a
functional test, not an automatic memory-leak acceptance.

The current test run contains one ignored hardware test:

```bash
export PATH="$HOME/.cargo/bin:$PATH"
export LD_LIBRARY_PATH=/opt/rocm/lib
# Normal test run: the GPU test stays ignored.
cargo test --locked --manifest-path tests/runtime/llamacpp/Cargo.toml
# Only explicitly, on the GPU machine:
ELIEZER_TEST_MODEL=/path/to/model.gguf \
  cargo test --locked --manifest-path tests/runtime/llamacpp/Cargo.toml -- --ignored --nocapture
```

Format check, Clippy with `-D warnings`, normal test run (1 ignored GPU test)
and explicitly started GPU test (1 passed) were executed.

## Known binding gaps for M4

- OpenAI/Anthropic responses and SSE events must be built by Eliezer itself.
- Token counts are available from the actual token sequences; the HTTP `usage`
  must be derived from them by the adapter.
- The binding does not provide a complete tool-call interface.
  Tool definitions, template integration and detection of the output are built in M4.
- Thinking options must be translated into template and parameters depending on the model.
- Vision requires the additional `mtmd` integration and matching projectors.
  This combination was neither built nor tested here; it remains a separate M4 verification.
- Cancellation and parallel HTTP requests are tasks of the later adapter.
  Two sequences in one context are demonstrated in practice here.
