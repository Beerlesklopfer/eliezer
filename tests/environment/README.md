# Environment check

Invocation from the repository:

```bash
bash tests/environment/check.sh
```

The script reports Rust versions, visible GPU devices, sysfs memory figures,
tools in the PATH and existing LM Studio runtimes. It loads no models,
needs no network and reads no secrets. Missing optional tools are reported
without making the invocation fail.

## Check result of 19.10.2026

- Syntax check with `bash -n` and execution succeeded.
- User installation: Rust and Cargo 1.99.0.
- Inside the sandbox the GPU devices are missing.
- Cross-check on the host: `/dev/dri` and `/dev/kfd` present;
  ROCm and the existing Vulkan `llama-server` detect the Radeon RX 7900 XTX.
- This is an inventory, not proof of a model run.
