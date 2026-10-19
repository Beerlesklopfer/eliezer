# Umgebungsprüfung

Aufruf aus dem Repository:

```bash
bash tests/environment/check.sh
```

Das Skript nennt Rust-Versionen, sichtbare GPU-Geräte, sysfs-Speicherangaben,
Werkzeuge im PATH und vorhandene LM-Studio-Laufzeiten. Es lädt keine Modelle,
benötigt kein Netz und liest keine Geheimnisse. Fehlende optionale Werkzeuge
werden gemeldet, ohne den Aufruf fehlschlagen zu lassen.

## Prüfergebnis vom 19.10.2026

- Syntaxprüfung mit `bash -n` und Ausführung erfolgreich.
- Benutzerinstallation: Rust und Cargo 1.99.0.
- Innerhalb der Sandbox fehlen die GPU-Geräte.
- Gegenprüfung auf dem Host: `/dev/dri` und `/dev/kfd` vorhanden;
  ROCm und der vorhandene Vulkan-`llama-server` erkennen Radeon RX 7900 XTX.
- Das ist eine Bestandsaufnahme, kein Nachweis eines Modelllaufs.
