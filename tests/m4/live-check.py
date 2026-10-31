"""Live check of the llama.cpp runtime through the HTTP API, with a real GGUF model.

Usage:
    python3 -I tests/m4/live-check.py MODEL.gguf [--binary target/release/eliezer] [--tools]
                                       [--context 4096] [--parallel 2] [--keep-log]

Starts the service on a free loopback port with a temporary configuration, exercises the API and
prints measurements. It needs the real model file, so it is not part of `cargo test`. Build the
binary with a runtime feature first (`--features rocm`, `vulkan`, `cuda` or `cpu`).
`--tools` adds a tool-call check for models whose chat template supports tools.
Nothing is downloaded and no model is changed; the model file is only read.
"""
import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok), detail))
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""), flush=True)


def call(port, path, body=None, timeout=600):
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=None if body is None else json.dumps(body).encode(),
        headers={"content-type": "application/json"},
        method="GET" if body is None else "POST",
    )
    try:
        with OPENER.open(request, timeout=timeout) as response:
            raw = response.read()
            try:
                return response.status, json.loads(raw or b"null")
            except ValueError:
                return response.status, raw.decode("utf-8", "replace")
    except urllib.error.HTTPError as error:
        raw = error.read()
        try:
            return error.code, json.loads(raw)
        except ValueError:
            return error.code, raw.decode("utf-8", "replace")


def stream(port, path, body, timeout=600):
    """Returns (frames, seconds_to_first_frame, seconds_total) for an SSE response."""
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=json.dumps(body).encode(),
        headers={"content-type": "application/json"},
        method="POST",
    )
    start = time.monotonic()
    first = None
    frames = []
    with OPENER.open(request, timeout=timeout) as response:
        for line in response:
            line = line.decode().strip()
            if not line.startswith("data:"):
                continue
            if first is None:
                first = time.monotonic() - start
            payload = line[5:].strip()
            if payload != "[DONE]":
                frames.append(json.loads(payload))
    return frames, first, time.monotonic() - start


def chat(port, content, **extra):
    body = {"model": "m", "messages": [{"role": "user", "content": content}], "temperature": 0, **extra}
    started = time.monotonic()
    status, data = call(port, "/v1/chat/completions", body)
    return status, data, time.monotonic() - started


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("--binary", type=Path, default=Path("target/release/eliezer"))
    parser.add_argument("--context", type=int, default=4096)
    parser.add_argument("--parallel", type=int, default=2)
    parser.add_argument("--tools", action="store_true")
    parser.add_argument("--keep-log", action="store_true")
    args = parser.parse_args()
    model = args.model.resolve()
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    with tempfile.TemporaryDirectory(prefix="eliezer-live-") as directory:
        root = Path(directory)
        config = root / "config.toml"
        config.write_text(
            f"""[server]
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
max_context = {args.context}
default_context = {args.context}
parallel = {args.parallel}
vram_gb = {model.stat().st_size / 1e9 + 1.5:.1f}
"""
        )
        log = root / "service.log"
        env = dict(os.environ, LD_LIBRARY_PATH=f"/opt/rocm/lib:{os.environ.get('LD_LIBRARY_PATH', '')}")
        with log.open("w") as out:
            process = subprocess.Popen([str(args.binary), "--config", str(config)], stdout=out, stderr=out, env=env)
        try:
            deadline = time.monotonic() + 10
            while True:
                try:
                    if call(port, "/healthz", timeout=1)[0] == 200:
                        break
                except OSError:
                    pass
                if process.poll() is not None or time.monotonic() > deadline:
                    raise SystemExit("service did not start:\n" + log.read_text())
                time.sleep(0.1)
            run(port, args)
            process.send_signal(signal.SIGTERM)
            check("orderly shutdown (SIGTERM, exit 0)", process.wait(timeout=60) == 0)
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
            if args.keep_log or any(not ok for _, ok, _ in RESULTS):
                print("---- service log ----")
                print(log.read_text()[-4000:])
    failed = [name for name, ok, _ in RESULTS if not ok]
    print(f"\n{len(RESULTS) - len(failed)} of {len(RESULTS)} checks passed")
    sys.exit(1 if failed else 0)


def run(port, args):
    _, before = call(port, "/api/v1/status")
    baseline = before["gpu"]["used_bytes"] if before["gpu"]["probe"] != "none" else None
    status, models = call(port, "/api/v1/models")
    check("model is listed, not loaded", status == 200 and models["models"][0]["loaded_instances"] == [])

    started = time.monotonic()
    status, loaded = call(port, "/api/v1/models/load", {"model": "m"})
    load_seconds = time.monotonic() - started
    check("load", status == 200 and loaded.get("status") == "loaded", f"{load_seconds:.1f} s")
    _, models = call(port, "/api/v1/models")
    cfg = (models["models"][0]["loaded_instances"] or [{}])[0].get("config", {})
    check("loaded config is reported", cfg == {"context_length": args.context, "parallel": args.parallel}, str(cfg))

    status, reply, seconds = chat(port, "Reply with exactly one word: pong", max_tokens=400)
    text = reply["choices"][0]["message"]["content"] if status == 200 else ""
    check("chat answers", status == 200 and text.strip() != "", repr(text[:60]))
    if status == 200:
        usage = reply["usage"]
        check(
            "usage is reported",
            usage["prompt_tokens"] > 0 and usage["completion_tokens"] > 0,
            f"{usage['prompt_tokens']} in, {usage['completion_tokens']} out, {seconds:.2f} s",
        )

    status, long_reply, seconds = chat(port, "Write a short paragraph about the sea.", max_tokens=120)
    if status == 200:
        out = long_reply["usage"]["completion_tokens"]
        print(f"      speed: {out} tokens in {seconds:.2f} s = {out / seconds:.1f} tokens/s (includes prompt)")

    body = {"model": "m", "messages": [{"role": "user", "content": "Count from 1 to 8."}], "temperature": 0, "max_tokens": 48}
    status, whole = call(port, "/v1/chat/completions", body)
    frames, first, total = stream(port, "/v1/chat/completions", dict(body, stream=True, stream_options={"include_usage": True}))
    streamed = "".join(f["choices"][0]["delta"].get("content", "") for f in frames if f.get("choices"))
    check("stream equals non-stream output (temperature 0)", status == 200 and streamed == whole["choices"][0]["message"]["content"])
    check("first SSE frame arrives before the end", first is not None and first < total, f"{first:.2f} s of {total:.2f} s")
    check("final usage frame", bool(frames) and "usage" in frames[-1] and frames[-1]["usage"]["completion_tokens"] > 0)

    status, stopped, _ = chat(port, "Count from 1 to 9, separated by commas.", max_tokens=400, stop=["5"])
    content = stopped["choices"][0]["message"]["content"] if status == 200 else "5"
    check("stop sequence cuts the answer", status == 200 and "5" not in content and "4" in content, repr(content))

    status, short, _ = chat(port, "Tell me a long story.", max_tokens=5)
    check(
        "max_tokens ends with finish_reason=length",
        status == 200 and short["choices"][0]["finish_reason"] == "length" and short["usage"]["completion_tokens"] == 5,
    )

    results = {}

    def worker(name, prompt):
        results[name] = chat(port, prompt, max_tokens=60)

    threads = [threading.Thread(target=worker, args=(i, f"Name one animal that starts with the letter {l}.")) for i, l in enumerate("AB")]
    started = time.monotonic()
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    parallel_seconds = time.monotonic() - started
    check("two parallel requests both answer", all(r[0] == 200 for r in results.values()), f"{parallel_seconds:.2f} s together")

    prompt = "Summarize this text in one sentence. " + "The quick brown fox jumps over the lazy dog. " * 60
    _, cold, cold_seconds = chat(port, prompt, max_tokens=8)
    _, warm, warm_seconds = chat(port, prompt, max_tokens=8)
    print(f"      prefix reuse: first {cold_seconds:.2f} s, repeat {warm_seconds:.2f} s")
    check("repeating a long prompt is not slower (prefix cache)", warm_seconds <= cold_seconds * 1.2)

    huge = "word " * (args.context * 2)
    status, error, _ = chat(port, huge, max_tokens=8)
    check("prompt longer than the context is a 400, not a crash", status == 400, str(error)[:80])
    status, _, _ = chat(port, "Still alive?", max_tokens=8)
    check("service still works after the refused request", status == 200)

    status, anth = call(
        port,
        "/v1/messages",
        {"model": "m", "max_tokens": 400, "temperature": 0, "messages": [{"role": "user", "content": "Say hello."}]},
    )
    # Reasoning models put a "thinking" block before the text block.
    texts = [b["text"] for b in anth.get("content", []) if b.get("type") == "text"] if status == 200 else []
    check("Anthropic endpoint answers", status == 200 and any(t.strip() for t in texts), repr(texts[0][:40]) if texts else str(anth)[:120])

    if args.tools:
        tool = {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather for a city",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}}, "required": ["city"]},
            },
        }
        status, called, _ = chat(port, "What is the weather in Paris right now? Use the tool.", tools=[tool], max_tokens=1500)
        calls = (called["choices"][0]["message"].get("tool_calls") if status == 200 else None) or []
        ok = bool(calls) and calls[0]["function"]["name"] == "get_weather" and "city" in json.loads(calls[0]["function"]["arguments"])
        check("tool call is detected", ok, json.dumps(calls)[:160] if calls else str(called)[:160])
        check("finish_reason is tool_calls", status == 200 and called["choices"][0]["finish_reason"] == "tool_calls")

    status, _ = call(port, "/api/v1/models/unload", {"instance_id": "m"})
    check("unload", status == 200)
    _, state = call(port, "/api/v1/status")
    check("status shows unloaded", state["models"][0]["state"] == "unloaded", f"probe={state['gpu']['probe']}")
    if baseline is not None:
        # Unloading must give the GPU memory back inside the running service (spec: unload_leak).
        started = time.monotonic()
        used = state["gpu"]["used_bytes"]
        # A constant 0.2-0.35 GB stays: the ROCm runtime context of the process. It does not grow
        # with repeated loads (see tests/m4/vram-cycles.py); model memory must be gone.
        while used > baseline + 512_000_000 and time.monotonic() - started < 10:
            time.sleep(0.2)
            used = call(port, "/api/v1/status")[1]["gpu"]["used_bytes"]
        waited = time.monotonic() - started
        check(
            "unload returns the model memory (within 512 MB of the baseline, 10 s)",
            used <= baseline + 512_000_000,
            f"baseline {baseline / 1e9:.2f} GB, now {used / 1e9:.2f} GB, waited {waited:.1f} s",
        )


if __name__ == "__main__":
    main()
