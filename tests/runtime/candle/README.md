# Featureprüfung Candle

## Durchführung

Am 19.10.2026 `cargo info candle-core` ausgeführt und die heruntergeladene
Quelle von `candle-core 0.11.0` geprüft, insbesondere `Cargo.toml` und `src/device.rs`.

Quelle: [Crate 0.11.0](https://docs.rs/crate/candle-core/0.11.0/source/).

## Ergebnis

- Gerätevarianten: CPU, CUDA, Metal.
- Kein ROCm-/Vulkan-Feature in dieser veröffentlichten Version.
- Damit ist kein direktes AMD-GPU-Backend dieser Version nachgewiesen.
- Dies ist eine Quellenprüfung, kein GPU- oder Modelllauf. Andere Git-Stände oder
  Forks sind durch diese Prüfung nicht bewertet.

## Exakte Abhängigkeit der neuen mistral.rs-Release

Zusätzlich wurde der in mistral.rs v0.9.5 festgehaltene Candle-Commit
`66a8cf184a5a519671454066b1b9efd446ec9f5c` direkt von GitHub abgerufen.
`candle-core/Cargo.toml` und `candle-core/src/device.rs` zeigen ebenfalls nur
CPU/CUDA/Metal als Geräte und keine ROCm-/Vulkan-Features.
Damit gilt der negative AMD-Befund auch für die dort tatsächlich eingebundene
Git-Abhängigkeit, nicht nur für die separat veröffentlichte Crate 0.11.0.
