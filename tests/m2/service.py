"""Manuelle Prüfung des Programmrahmens über Loopback, ohne externes Netz/GPU."""
import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import tempfile
import time
import urllib.request


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--binary', type=Path, default=Path('target/debug/eliezer'))
    args = parser.parse_args()
    binary = args.binary.resolve()
    with tempfile.TemporaryDirectory(prefix='eliezer-service-') as directory:
        root = Path(directory)
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            port = sock.getsockname()[1]
        config = root/'config.toml'
        valid = f'[server]\nlisten="127.0.0.1:{port}"\n[gpu]\n'
        config.write_text(valid)
        log = root/'service.log'
        with log.open('w') as output:
            process = subprocess.Popen([str(binary), '--config', str(config)], stdout=output, stderr=output)
            try:
                opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
                deadline = time.monotonic()+5
                while True:
                    try:
                        with opener.open(f'http://127.0.0.1:{port}/healthz', timeout=1) as response:
                            assert response.read() == b'ok'
                        break
                    except OSError:
                        if time.monotonic() > deadline:
                            raise TimeoutError('Dienst nicht bereit')
                        time.sleep(0.05)
                config.write_text(valid.replace(f'127.0.0.1:{port}', 'VERTRAULICHER_WERT'))
                os.kill(process.pid, signal.SIGHUP)
                deadline = time.monotonic()+5
                while 'config_rejected' not in log.read_text():
                    if time.monotonic() > deadline:
                        raise TimeoutError('Ungültiger Reload nicht gemeldet')
                    time.sleep(0.05)
                assert 'VERTRAULICHER_WERT' not in log.read_text()
                config.write_text(valid)
                os.kill(process.pid, signal.SIGHUP)
                deadline = time.monotonic()+5
                while 'config_check' not in log.read_text():
                    if time.monotonic() > deadline:
                        raise TimeoutError('Gültiger Reload nicht gemeldet')
                    time.sleep(0.05)
                os.kill(process.pid, signal.SIGTERM)
                assert process.wait(timeout=5) == 0
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait(timeout=5)
        print(json.dumps({'healthz': 'ok', 'invalid_reload': 'rejected', 'valid_reload': 'checked', 'sigterm': 'clean', 'secret_value_in_log': False}))


if __name__ == '__main__':
    main()
