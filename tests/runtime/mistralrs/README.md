# Version and feature check of mistral.rs

## Procedure

Fetched directly from GitHub on 19.10.2026 and the manifests read:

```bash
git ls-remote https://github.com/EricLBuehler/mistral.rs.git HEAD refs/heads/master
```

Commit checked: `465191d0b0a7e2432d327b0de06204ab3f94d3a2`.
Sources: [workspace manifest](https://github.com/EricLBuehler/mistral.rs/blob/465191d0b0a7e2432d327b0de06204ab3f94d3a2/Cargo.toml),
[SDK manifest](https://github.com/EricLBuehler/mistral.rs/blob/465191d0b0a7e2432d327b0de06204ab3f94d3a2/mistralrs/Cargo.toml)
and [core manifest](https://github.com/EricLBuehler/mistral.rs/blob/465191d0b0a7e2432d327b0de06204ab3f94d3a2/mistralrs-core/Cargo.toml).

## Result

- Release and workspace version 0.9.5, minimum Rust version 1.94.
- `git ls-remote ... refs/tags/v0.9.5` confirms the tag on the same commit.
  The earlier claim that 0.9.4 was the latest release was based on a stale web query.
- No ROCm/Vulkan features in SDK and core; CUDA and Metal present.
- The earlier `cargo info` output 0.8.1 is not evidence of the current upstream state.
- This is a source check. No Rust model run with mistral.rs has taken place yet;
  no statements about memory release or throughput are derived from it.
