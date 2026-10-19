"""Manual comparison chat -> image -> chat against an isolated llama-swap."""
import argparse
import base64
import importlib.util
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import time

source = Path(__file__).parents[1]/'sdcpp'/'probe.py'
spec = importlib.util.spec_from_file_location('image_probe', source)
shared = importlib.util.module_from_spec(spec)
spec.loader.exec_module(shared)


def main():
    parser = argparse.ArgumentParser()
    for name in ('swap', 'chat-server', 'chat-model', 'image-server', 'diffusion', 'encoder', 'vae'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path(__file__).parent/'target')
    args = parser.parse_args()
    for name, value in vars(args).items():
        if name != 'output' and not value.is_file():
            parser.error(f'File missing: {value}')
    args.output.mkdir(parents=True, exist_ok=True)
    chat_command = [str(args.chat_server), '--model', str(args.chat_model), '--host', '127.0.0.1', '--port', '24607', '--ctx-size', '1024', '--parallel', '1', '--n-gpu-layers', '999']
    image_command = [str(args.image_server), '--diffusion-model', str(args.diffusion), '--llm', str(args.encoder), '--vae', str(args.vae), '--listen-ip', '127.0.0.1', '--listen-port', '24608', '--offload-to-cpu', '--cfg-scale', '6.0', '--sampling-method', 'euler', '--steps', '4', '--log-level', 'warn']
    # JSON is valid YAML; paths are safely quoted as arguments.
    config = {'healthCheckTimeout': 180, 'captureBuffer': 0, 'logLevel': 'warn',
              'logToStdout': 'none', 'models': {
                  'chat': {'cmd': shlex.join(chat_command), 'proxy': 'http://127.0.0.1:24607', 'checkEndpoint': '/health'},
                  'image': {'cmd': shlex.join(image_command), 'proxy': 'http://127.0.0.1:24608', 'checkEndpoint': '/v1/models'}}}
    config_path = args.output/'config.yaml'
    config_path.write_text(json.dumps(config, indent=2)+'\n')
    subprocess.run([str(args.swap), '-config', str(config_path), '-validate'], env=shared.child_env(), check=True, timeout=10)
    results = {'baseline_bytes': shared.memory(), 'requests': []}
    process = None
    try:
        with (args.output/'swap.log').open('w') as log:
            process = subprocess.Popen([str(args.swap), '-config', str(config_path), '-listen', '127.0.0.1:24606'], env=shared.child_env(), stdout=log, stderr=log, start_new_session=True)
            start = time.monotonic()
            while True:
                try:
                    shared.request('http://127.0.0.1:24606/v1/models', timeout=2)
                    break
                except OSError:
                    if process.poll() is not None or time.monotonic()-start > 30:
                        raise RuntimeError('llama-swap not ready; see swap.log')
                    time.sleep(0.2)
            for kind in ('chat', 'image', 'chat'):
                started = time.monotonic()
                if kind == 'chat':
                    data = shared.request('http://127.0.0.1:24606/v1/chat/completions', {
                        'model': 'chat', 'messages': [{'role': 'user', 'content': 'Answer briefly: What is Rust?'}],
                        'max_tokens': 32, 'temperature': 0})
                    if not data['choices'][0]['message'].get('content'):
                        raise ValueError('Empty chat reply')
                    detail = {'usage': data.get('usage')}
                else:
                    data = shared.request('http://127.0.0.1:24606/v1/images/generations', {
                        'model': 'image', 'prompt': 'A red wooden cube on a white table, studio photograph',
                        'n': 1, 'size': '512x512', 'output_format': 'png'})
                    image = base64.b64decode(data['data'][0]['b64_json'], validate=True)
                    if image[:8] != b'\x89PNG\r\n\x1a\n':
                        raise ValueError('No PNG in the image reply')
                    (args.output/'image.png').write_bytes(image)
                    detail = {'image_size_bytes': len(image)}
                results['requests'].append({'kind': kind, 'seconds_including_load_and_swap': time.monotonic()-started, 'vram_bytes': shared.memory(), **detail})
    finally:
        if process is not None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=10)
            except ProcessLookupError:
                pass
        time.sleep(1)
        results['after_exit_bytes'] = shared.memory()
        (args.output/'results.json').write_text(json.dumps(results, indent=2)+'\n')
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
