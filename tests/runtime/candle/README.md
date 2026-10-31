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

## Re-check of 19 Oct 2026 (AMD progress)

Question: is Candle prioritizing an AMD backend? Checked on GitHub, without building anything.

- `main` (last commit `12bb5d1f`, 14 Oct 2026) and tag `0.11.0` still offer only CPU, CUDA, `cutile`,
  `cudnn`, `nccl`, MKL, Accelerate and Metal in `candle-core`. None of the last 60 commits touches ROCm,
  HIP, Vulkan or wgpu; the recent commits are CPU/CUDA fixes and CI pinning.
- Open, unmerged pull requests: #3801 "Feat/rocm backend" (+18 491/−523, 14 comments, last updated 28 Sep),
  #3424 "Initial implementation for ROCm support" (+6 567, last updated 31 Jul), #3379 "experimental wgpu
  backend" (+53 881, last updated 15 Aug) and #3974 Intel GPU/SYCL. No reviewer is requested on any of them.
- Issue #3985 offers an out-of-tree Vulkan backend (`Device::Vulkan`) for upstreaming.

So the ROCm work exists as large external pull requests, but no maintainer decision to merge one is
visible. Candle stays unusable on the AMD card until one of them lands in a release. Re-check when
`candle-core` shows a `rocm` (or `vulkan`) feature.
