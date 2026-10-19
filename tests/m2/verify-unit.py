"""Prüft die Unit-Syntax ohne Installation eines Systemdienstes."""
from pathlib import Path
import subprocess
import tempfile

repo = Path(__file__).resolve().parents[2]
binary = repo/'target/debug/eliezer'
if not binary.is_file():
    raise SystemExit('Zuerst cargo build ausführen')
with tempfile.TemporaryDirectory(prefix='eliezer-unit-') as directory:
    unit = Path(directory)/'eliezer.service'
    # Nur der noch nicht installierte Programmpfad wird im Prüfexemplar ersetzt.
    text = (repo/'packaging/eliezer.service').read_text().replace(
        'ExecStart=/usr/local/bin/eliezer ', f'ExecStart="{binary}" ')
    unit.write_text(text)
    subprocess.run(['systemd-analyze', 'verify', str(unit)], check=True, timeout=15)
print('Unit-Syntax geprüft; der Systemdienst wurde nicht installiert.')
