"""Repeated load/unload cycles of one model: how much VRAM stays after each unload?

Usage: python3 -I tests/m4/vram-cycles.py MODEL.gguf   (run from the repository root, ROCm build)
A constant small residue is the runtime context of the process; a growing one is a leak.
See tests/m4/README.md for measurements.
"""
import importlib.util, json, os, socket, subprocess, sys, tempfile, time
from pathlib import Path
spec = importlib.util.spec_from_file_location("lc", "tests/m4/live-check.py")
lc = importlib.util.module_from_spec(spec); spec.loader.exec_module(lc)
model = Path(sys.argv[1]).resolve()
with socket.socket() as s:
    s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]
d = Path(tempfile.mkdtemp(prefix="eliezer-cycles-"))
(d/"config.toml").write_text(f'''[server]
listen = "127.0.0.1:{port}"
[storage]
models_dir = "{model.parent}"
[huggingface]
cache_dir = "{model.parent}/hf"
[gpu]
vendor = "auto"
vram_budget_gb = 24.0
reserve_gb = 0.5
[backend.llamacpp]
enabled = true
[[model]]
key = "m"
backend = "llamacpp"
source = {{ file = "{model}" }}
kind = "llm"
max_context = 8192
default_context = 8192
parallel = 2
vram_gb = {model.stat().st_size/1e9+1.5:.1f}
''')
p = subprocess.Popen(["target/release/eliezer", "--config", str(d/"config.toml")], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
try:
    while True:
        try:
            if lc.call(port, "/healthz", timeout=1)[0] == 200: break
        except OSError: time.sleep(0.1)
    used = lambda: lc.call(port, "/api/v1/status")[1]["gpu"]["used_bytes"] / 1e9
    base = used(); print(f"baseline {base:.2f} GB")
    for i in range(1, 6):
        t = time.monotonic(); st, _ = lc.call(port, "/api/v1/models/load", {"model": "m"}); load = time.monotonic() - t
        st2, r = lc.call(port, "/v1/chat/completions", {"model": "m", "messages": [{"role": "user", "content": "Say hi."}], "max_tokens": 8, "temperature": 0})
        loaded = used()
        lc.call(port, "/api/v1/models/unload", {"instance_id": "m"}); time.sleep(3)
        print(f"cycle {i}: load {load:.1f} s (HTTP {st}, chat {st2}), loaded {loaded:.2f} GB, after unload {used():.2f} GB = baseline + {used()-base:+.2f} GB", flush=True)
    p.terminate(); p.wait(30)
finally:
    if p.poll() is None: p.kill()
time.sleep(4)
print("after process exit:", f"{int(open('/sys/class/drm/card0/device/mem_info_vram_used').read())/1e9:.2f} GB")
