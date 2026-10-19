"""Checks the unit syntax without installing a system service."""
from pathlib import Path
import subprocess
import tempfile

repo = Path(__file__).resolve().parents[2]
binary = repo/'target/debug/eliezer'
if not binary.is_file():
    raise SystemExit('Run cargo build first')
with tempfile.TemporaryDirectory(prefix='eliezer-unit-') as directory:
    unit = Path(directory)/'eliezer.service'
    # Only the not yet installed program path is replaced in the test copy.
    text = (repo/'packaging/eliezer.service').read_text().replace(
        'ExecStart=/usr/local/bin/eliezer ', f'ExecStart="{binary}" ')
    unit.write_text(text)
    subprocess.run(['systemd-analyze', 'verify', str(unit)], check=True, timeout=15)
print('Unit syntax checked; the system service was not installed.')
