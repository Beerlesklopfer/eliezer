"""Manual compose acceptance test using only its own temporary test data."""
import json
import os
from pathlib import Path
import socket
import subprocess
import tempfile
import time
import urllib.request

REPO = Path(__file__).resolve().parents[2]


def main():
    results = {}
    with tempfile.TemporaryDirectory(prefix='eliezer-compose-') as directory:
        root = Path(directory)
        models = root/'models'
        models.mkdir()
        override = REPO/'compose.podman.yaml'
        for variant in ('bind', 'volume'):
            project = f'eliezer-m2-{variant}-{os.getpid()}'
            with socket.socket() as sock:
                sock.bind(('127.0.0.1', 0))
                port = sock.getsockname()[1]
            env = {**os.environ, 'ELIEZER_MODELS_DIR': str(models), 'ELIEZER_PORT': str(port),
                   'ELIEZER_CONFIG_FILE': str(REPO/'packaging/config.container.toml')}
            command = ['podman-compose', '-p', project, '-f', str(REPO/'compose.yaml')]
            if variant == 'volume':
                command += ['-f', str(REPO/'compose.volume.yaml')]
            command += ['-f', str(override)]
            log_path = root/f'{variant}.log'
            with log_path.open('w') as log:
                def compose(*args, check=True):
                    return subprocess.run(command+list(args), cwd=REPO, env=env, stdout=log,
                                          stderr=log, timeout=90, check=check)

                def container_id():
                    output = subprocess.check_output(['podman', 'ps', '-q', '--filter',
                        f'label=io.podman.compose.project={project}'], text=True, timeout=10).strip().splitlines()
                    if len(output) != 1:
                        raise RuntimeError('Exactly one own test container expected')
                    return output[0]

                def health():
                    deadline = time.monotonic()+15
                    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
                    while True:
                        try:
                            with opener.open(f'http://127.0.0.1:{port}/healthz', timeout=1) as response:
                                assert response.read() == b'ok'
                            return
                        except OSError:
                            if time.monotonic() > deadline:
                                raise TimeoutError('Container not ready')
                            time.sleep(0.1)
                try:
                    compose('up', '-d', '--no-build')
                    health()
                    first = container_id()
                    subprocess.run(['podman', 'exec', first, 'sh', '-c',
                        'printf pattern > /var/lib/eliezer/models/m2-sentinel; printf state > /var/lib/eliezer/m2-state'], check=True, timeout=10)
                    compose('down')
                    compose('up', '-d', '--no-build')
                    health()
                    second = container_id()
                    assert first != second
                    values = subprocess.check_output(['podman', 'exec', second, 'sh', '-c',
                        'cat /var/lib/eliezer/models/m2-sentinel /var/lib/eliezer/m2-state'], text=True, timeout=10)
                    assert values == 'patternstate'
                    results[variant] = {'healthz': 'ok', 'recreated': True,
                        'model_persisted': True, 'state_persisted': True}
                except Exception:
                    print(log_path.read_text())
                    raise
                finally:
                    # Remove only the project volumes newly created here.
                    compose('down', '--volumes', check=False)
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
