"""Manueller GPU-Versuch; lädt ausschließlich explizit übergebene lokale Dateien."""
import argparse
import base64
import json
import os
from pathlib import Path
import struct
import subprocess
import threading
import time
import urllib.error
import urllib.request

OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def memory():
    return int(Path('/sys/class/drm/card0/device/mem_info_vram_used').read_text())


def request(url, body=None, timeout=600):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})
    with OPENER.open(req, timeout=timeout) as response:
        return json.load(response)


def stop(process):
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)


def child_env():
    return {key: value for key, value in os.environ.items() if key in (
        'PATH', 'LD_LIBRARY_PATH', 'HIP_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES',
        'CUDA_VISIBLE_DEVICES')}


def sample_memory(values, done):
    while not done.wait(0.05):
        values.append(memory())


def chat_probe(binary, model, output):
    before = memory()
    started = time.monotonic()
    with output.open('w') as log:
        subprocess.run([str(binary), str(model)], stdout=log, stderr=log,
                       env=child_env(), timeout=180, check=True)
    time.sleep(1)
    return {'seconds': time.monotonic()-started, 'before_bytes': before,
            'after_exit_bytes': memory()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--server', type=Path, required=True)
    parser.add_argument('--diffusion', type=Path, required=True)
    parser.add_argument('--encoder', type=Path, required=True)
    parser.add_argument('--vae', type=Path, required=True)
    parser.add_argument('--chat-probe', type=Path)
    parser.add_argument('--chat-model', type=Path)
    parser.add_argument('--port', type=int, default=24609)
    parser.add_argument('--output', type=Path, default=Path(__file__).parent/'target')
    args = parser.parse_args()
    for path in (args.server, args.diffusion, args.encoder, args.vae):
        if not path.is_file():
            parser.error(f'Datei fehlt: {path}')
    if bool(args.chat_probe) != bool(args.chat_model):
        parser.error('Chat-Probe und Chat-Modell müssen gemeinsam angegeben werden')
    args.output.mkdir(parents=True, exist_ok=True)
    results = {}
    if args.chat_probe:
        results['chat_before_image'] = chat_probe(args.chat_probe, args.chat_model, args.output/'chat-before.log')
    baseline = memory()
    samples = [baseline]
    done = threading.Event()
    sampler = threading.Thread(target=sample_memory, args=(samples, done), daemon=True)
    sampler.start()
    command = [str(args.server), '--diffusion-model', str(args.diffusion), '--llm',
               str(args.encoder), '--vae', str(args.vae), '--listen-ip', '127.0.0.1',
               '--listen-port', str(args.port), '--offload-to-cpu', '--cfg-scale', '6.0',
               '--sampling-method', 'euler', '--steps', '4', '--log-level', 'warn']
    process = None
    started = time.monotonic()
    try:
        with (args.output/'server.log').open('w') as log:
            process = subprocess.Popen(command, stdout=log, stderr=log, env=child_env())
            url = f'http://127.0.0.1:{args.port}'
            while True:
                if process.poll() is not None:
                    raise RuntimeError(f'Bildserver beendet: {process.returncode}; siehe server.log')
                try:
                    models = request(url+'/v1/models', timeout=2)
                    break
                except (OSError, urllib.error.URLError):
                    if time.monotonic()-started > 180:
                        raise TimeoutError('Bildserver nicht innerhalb von 180 Sekunden bereit')
                    time.sleep(0.2)
            results['server_ready_seconds'] = time.monotonic()-started
            results['model_list_received'] = isinstance(models.get('data'), list)
            results['image_ready_bytes'] = memory()
            started_image = time.monotonic()
            response = request(url+'/v1/images/generations', {
                'prompt': 'A red wooden cube on a white table, studio photograph',
                'n': 1, 'size': '512x512', 'output_format': 'png'})
            results['first_image_seconds_including_model_load'] = time.monotonic()-started_image
            data = base64.b64decode(response['data'][0]['b64_json'], validate=True)
            if data[:8] != b'\x89PNG\r\n\x1a\n':
                raise ValueError('Antwort enthält kein PNG')
            dimensions = struct.unpack('>II', data[16:24])
            if dimensions != (512, 512):
                raise ValueError(f'Unerwartete Bildgröße: {dimensions}')
            (args.output/'image.png').write_bytes(data)
            results['image_dimensions'] = dimensions
            results['image_size_bytes'] = len(data)
    finally:
        if process is not None and process.poll() is None:
            stop(process)
        done.set()
        sampler.join(timeout=2)
        time.sleep(1)
        results['image_peak_bytes'] = max(samples)
        results['image_baseline_bytes'] = baseline
        results['after_image_exit_bytes'] = memory()
        (args.output/'results.json').write_text(json.dumps(results, indent=2)+'\n')
    if args.chat_probe:
        results['chat_after_image'] = chat_probe(args.chat_probe, args.chat_model, args.output/'chat-after.log')
    (args.output/'results.json').write_text(json.dumps(results, indent=2)+'\n')
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
