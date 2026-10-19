# Tests

All tests, probe programs, instructions and result logs live here.
Conclusion of the first runtime evaluation: [M1 result](m1/README.md).
Automated tests run without network and GPU; manual hardware checks are started
explicitly. The future Cargo workspace includes hardware checks only as ignored
tests with a stated reason.

```text
tests/
├── README.md
├── m1/README.md              Conclusion and decisions
├── m2/                      Configuration, CLI, service and container checks
├── reference-clients/
│   └── README.md             Verified Ollama / LM Studio command lines
├── environment/
│   ├── README.md             Prerequisites and invocation
│   └── check.sh              Local inventory
└── runtime/
    ├── mistralrs/README.md   Version and feature check
    ├── candle/README.md      Feature check
    ├── sdcpp/               Image server and chat/image switching
    ├── llama-swap/          Comparison with a ready-made manager
    └── llamacpp/
        ├── README.md         Experiment, procedure and results
        ├── Cargo.toml        Isolated probe program
        ├── Cargo.lock        Pinned dependencies
        └── src/main.rs       Load, generate, unload, reload
```

Further runtimes get their own directories under `runtime/`.
Contract tests and integration tests will be added later under `contract/` and
`integration/` respectively. Rust build output in `target/` stays ignored.
