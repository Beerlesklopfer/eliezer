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
lines = [l.strip() for l in (repo/'packaging/eliezer.service').read_text().splitlines() if not l.strip().startswith('#')]
# systemd refuses a symbolic link as StateDirectory, which breaks a model store on another disk.
assert not any(l.startswith('StateDirectory') for l in lines), 'StateDirectory= must not be used'
# /home must stay readable (a model store there), never hidden and never writable by default.
assert 'ProtectHome=read-only' in lines, 'ProtectHome must be read-only'
assert any(l.startswith('ReadWritePaths=-/var/lib/eliezer') for l in lines), 'state path must be optional'
print('Unit syntax checked; the system service was not installed.')
