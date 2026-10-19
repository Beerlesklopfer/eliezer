# Abhängigkeitsentscheidungen

Die Übersicht über Werkzeuge, Laufzeiten, GPU-Voraussetzungen und Zugriffsrechte
steht im [Projekt-Wiki](wiki/Abhaengigkeiten.md).

Konkrete Crate-Versionen und zusätzliche Abhängigkeiten sind noch nicht beschlossen.
Bei ihrer Aufnahme werden hier jeweils Version, Zweck und Begründung dokumentiert.

## Geprüfter mistral.rs-Upstream

Der direkt abgerufene Commit
[`465191d0b0a7e2432d327b0de06204ab3f94d3a2`](https://github.com/EricLBuehler/mistral.rs/commit/465191d0b0a7e2432d327b0de06204ab3f94d3a2)
führt Workspace-Version **0.9.5** und verlangt **Rust 1.94**.
Die letzte veröffentlichte GitHub-Release ist
[0.9.5](https://github.com/EricLBuehler/mistral.rs/releases/tag/v0.9.5)
vom 19.10.2026; ihr Git-Tag zeigt auf denselben Commit.
Eine Abhängigkeit auf diese Release ist noch nicht festgelegt.
Die Mindestversion 1.88 aus der Eliezer-Vorgabe genügt für diesen Upstream nicht;
diese Differenz muss vor der Aufnahme entschieden werden.

## Probeabhängigkeit für AMD

`llama-cpp-2 = 0.1.160` mit Feature `rocm` wurde für die isolierte M1-Probe
gewählt: Es bietet den benötigten eingebetteten Zugriff auf llama.cpp und
unterstützt die vorhandene AMD-Karte über HIP. Das transitive `llama-cpp-sys-2`
übersetzt C++-Code; CMake, C/C++-Compiler und ROCm-Entwicklungsbibliotheken sind
Bauvoraussetzungen. Die Probe hat einen eigenen `Cargo.lock` unter `tests/`.
Es besteht noch keine entsprechende Abhängigkeit in einem produktiven Workspace.

Einbauentscheidung und Grenzen: [M1-Ergebnis](../tests/m1/README.md).
