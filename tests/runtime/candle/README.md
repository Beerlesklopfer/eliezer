# Candle feature check

## Procedure

On 19.10.2026 `cargo info candle-core` was run and the downloaded
source of `candle-core 0.11.0` was inspected, in particular `Cargo.toml` and `src/device.rs`.

Source: [crate 0.11.0](https://docs.rs/crate/candle-core/0.11.0/source/).

## Result

- Device variants: CPU, CUDA, Metal.
- No ROCm/Vulkan feature in this published version.
- Hence no direct AMD GPU backend is demonstrated for this version.
- This is a source check, not a GPU or model run. Other git revisions or
  forks are not assessed by this check.

## Exact dependency of the new mistral.rs release

In addition, the Candle commit `66a8cf184a5a519671454066b1b9efd446ec9f5c`
pinned in mistral.rs v0.9.5 was fetched directly from GitHub.
`candle-core/Cargo.toml` and `candle-core/src/device.rs` likewise show only
CPU/CUDA/Metal as devices and no ROCm/Vulkan features.
The negative AMD finding therefore also holds for the git dependency actually
pulled in there, not only for the separately published crate 0.11.0.
