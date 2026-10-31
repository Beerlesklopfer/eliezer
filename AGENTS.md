# AGENTS.md – Working instructions for Codex and other agents

Eliezer is a model manager written in **Rust** that replaces LM Studio for
[MultiGPT](https://github.com/Beerlesklopfer/multi-gpt). The full description is in
[docs/Specification.md](docs/Specification.md); the milestones with acceptance criteria are in
[docs/Implementation.md](docs/Implementation.md). **Read both before you write code.**

## How to work

- Work through the milestones in order (M1 … M6). For each milestone: implement, test, add a short note to
  `docs/Status.md` (what works, what does not, measurements) and the results to `debian/changelog`, then stop and wait for approval.
- Do not invent anything: check the interfaces and command lines of third-party programs and crates (mistral.rs and llama.cpp as crates,
  stable-diffusion.cpp) **first against the installed program or the crate documentation and a probe program**, and record the
  result in `docs/Runtimes.md`. The specification only names what has to be clarified there.
- Make small, traceable commits with an English description in the imperative mood.
- The user decides version numbers. No tags, no releases, no `cargo publish`.
- All checks and probe programs, including their documentation, live under
  `tests/` in suitable subdirectories. Build output is not versioned.

## Language and style

- **English everywhere**: identifiers, types, functions, API fields, documentation, comments,
  error messages for humans, log lines, commit messages and file names.
- Rust stable, `rust-version = "1.88"`, edition 2024. Workspace with several crates (see Specification 3).
- `cargo fmt --check`, `cargo clippy --all-targets -- -D warnings` and `cargo test` must pass.
- No `unsafe` without a justification in a comment; no `unwrap()`/`expect()` outside tests and `main`.
- Async with `tokio`, HTTP server `axum`, client `reqwest` (rustls), serialization `serde`.
  Add new dependencies sparingly and justify them in `docs/Dependencies.md`.

## Security (binding)

- The Hugging Face token and API keys are **never** in the repository, in logs, in error messages,
  in test data or in process arguments (visible in `ps`). They come from a file (0600) or
  the environment and are passed on to child processes only through environment variables.
- Tests run **without network and without a GPU**: background programs are replaced by a test bench
  (`fake-backend`, Specification 9). Tests that need a GPU carry `#[ignore]`
  and a reason; they run only on request (`cargo test -- --ignored`).
- Only what is on the allowlist in the configuration is loaded.
- No input from HTTP requests reaches command lines, paths or a shell without being checked against
  the configuration (model keys are names from the list, not paths).

## Non-goals

No cluster, no user management, no billing, no web interface. Permissions, ranking
and billing live in MultiGPT; Eliezer only executes and reports facts.
