# Eliezer

**Ersatz für LM Studio** im Familienbetrieb (NVIDIA und AMD, OpenAI- und Anthropic-kompatibel): Verwalter für lokale KI-Modelle auf einem Rechner mit knapper GPU. Benannt nach Eliezer, dem Knecht
Abrahams, der das Haus verwaltet und Aufträge ausführt.

Die Herkunft des Namens und seine Bedeutung für das Projekt erklärt das
[Projekt-Wiki](docs/wiki/Projektname.md).

Eliezer teilt den Grafikspeicher zu: Chat-, Vision- und Bildmodelle laden bei Bedarf, werden im
Leerlauf entladen und wechseln sich auf einer GPU ab. Der Dienst steht hinter einer
OpenAI-kompatiblen Schnittstelle und wird von [MultiGPT](https://github.com/Beerlesklopfer/multi-gpt)
über den Runner gesteuert; Rechte, Rangfolge und Abrechnung bleiben dort.

**Stand:** M1 abgeschlossen; Laufzeit-Probeprogramme vorhanden, noch kein produktiver Dienst. [Ergebnisse](tests/m1/README.md). Spezifikation: [docs/Spezifikation.md](docs/Spezifikation.md), Roadmap: [docs/Implementierung.md](docs/Implementierung.md), Regeln für Agenten: [AGENTS.md](AGENTS.md).

## Vorhaben

- Ersetzt LM Studio als Modellserver von MultiGPT. Der Provider `lmstudio` in MultiGPT läuft weiter, bis Eliezer denselben Vertrag erfüllt: Modellliste mit Zustand und Speicherschätzung, Laden/Entladen mit Kontext und Parallelität, Denkoptionen je Modell, Statistik je Antwort.
- Laufzeit für Sprachmodelle: [mistral.rs](https://github.com/EricLBuehler/mistral.rs) (Rust).
- Bildmodelle: zunächst stable-diffusion.cpp als austauschbarer Dienst, später Rust.
- Verwalter: Anfragen nach `model` verteilen, VRAM-Buch, Leerlauf-Entladen, Warteschlange.
  Vergleich mit llama-swap abgeschlossen; der eigene Rust-Verwalter wird umgesetzt.
- Modelle laden per Hugging-Face-Token (nur auf dem GPU-Rechner, nie im Repository).
