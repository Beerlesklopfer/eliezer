# Eliezer

Verwalter für lokale KI-Modelle auf einem Rechner mit knapper GPU. Benannt nach Eliezer, dem Knecht
Abrahams, der das Haus verwaltet und Aufträge ausführt.

Eliezer teilt den Grafikspeicher zu: Chat-, Vision- und Bildmodelle laden bei Bedarf, werden im
Leerlauf entladen und wechseln sich auf einer GPU ab. Der Dienst steht hinter einer
OpenAI-kompatiblen Schnittstelle und wird von [MultiGPT](https://github.com/Beerlesklopfer/multi-gpt)
über den Runner gesteuert; Rechte, Rangfolge und Abrechnung bleiben dort.

**Stand:** Entwurf, noch kein Code. Der Plan steht in MultiGPT unter `docs/plans/Runner-Plotter.md`.

## Vorhaben

- Laufzeit für Sprachmodelle: [mistral.rs](https://github.com/EricLBuehler/mistral.rs) (Rust).
- Bildmodelle: zunächst stable-diffusion.cpp als austauschbarer Dienst, später Rust.
- Verwalter: Anfragen nach `model` verteilen, VRAM-Buch, Leerlauf-Entladen, Warteschlange.
  Vorbild: llama-swap und LocalAI; ob fertig eingesetzt oder in Rust gebaut, ist noch offen.
- Modelle laden per Hugging-Face-Token (nur auf dem GPU-Rechner, nie im Repository).
