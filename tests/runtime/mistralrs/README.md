# Versions- und Featureprüfung mistral.rs

## Durchführung

Am 19.10.2026 direkt von GitHub abgerufen und die Manifeste gelesen:

```bash
git ls-remote https://github.com/EricLBuehler/mistral.rs.git HEAD refs/heads/master
```

Geprüfter Commit: `465191d0b0a7e2432d327b0de06204ab3f94d3a2`.
Quellen: [Workspace-Manifest](https://github.com/EricLBuehler/mistral.rs/blob/465191d0b0a7e2432d327b0de06204ab3f94d3a2/Cargo.toml),
[SDK-Manifest](https://github.com/EricLBuehler/mistral.rs/blob/465191d0b0a7e2432d327b0de06204ab3f94d3a2/mistralrs/Cargo.toml)
und [Core-Manifest](https://github.com/EricLBuehler/mistral.rs/blob/465191d0b0a7e2432d327b0de06204ab3f94d3a2/mistralrs-core/Cargo.toml).

## Ergebnis

- Release und Workspace-Version 0.9.5, Mindest-Rust-Version 1.94.
- `git ls-remote ... refs/tags/v0.9.5` bestätigt den Tag auf demselben Commit.
  Die frühere Angabe 0.9.4 als neueste Release beruhte auf einer veralteten Web-Abfrage.
- Keine ROCm-/Vulkan-Features in SDK und Core; CUDA und Metal vorhanden.
- Die vorherige `cargo info`-Ausgabe 0.8.1 ist kein Nachweis des aktuellen Upstreamstands.
- Dies ist eine Quellenprüfung. Ein Rust-Modelllauf mit mistral.rs ist noch nicht erfolgt;
  Aussagen zu Speicherfreigabe oder Durchsatz werden daraus nicht abgeleitet.
