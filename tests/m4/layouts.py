"""Moved directories must not make the service fail: every layout of the model store, run with the real
binary under the same restrictions as the systemd unit.

Usage:
    python3 -I tests/m4/layouts.py MODEL.gguf [--binary target/gpu/release/eliezer]

Each layout is built below ~/.cache/eliezer-layouts (so `ProtectHome` matters) from hard links to
the given model file, which must live on the same file system. The service then runs as a transient
`systemd-run --user` service with the sandbox settings of packaging/eliezer.service
(ProtectSystem=strict, ProtectHome=read-only, NoNewPrivileges, PrivateTmp) and is driven with the
command line client: ls, load, run, ps, unload. The system service itself is not touched and no
root rights are needed. It needs a GPU or enough RAM to load the model.
"""
import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

HOME = Path.home()
BASE = HOME / ".cache" / "eliezer-layouts" / "run"
RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok)))
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""), flush=True)


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def config(port, models_dir, files, extra=""):
    entries = "".join(
        f'[[model]]\nkey = "{key}"\nbackend = "llamacpp"\nsource = {{ file = "{path}" }}\nkind = "llm"\n'
        f"max_context = 4096\ndefault_context = 4096\nvram_gb = 6.0\n"
        for key, path in files.items()
    )
    return (
        f'[server]\nlisten = "127.0.0.1:{port}"\n[storage]\nmodels_dir = "{models_dir}"\n'
        f'[huggingface]\ncache_dir = "{models_dir}/hf"\n[gpu]\nvendor = "auto"\nvram_budget_gb = 22.0\n'
        f"reserve_gb = 1.0\n[backend.llamacpp]\nenabled = true\n{extra}{entries}"
    )


def sandbox_props(writable):
    # The settings of packaging/eliezer.service that matter for paths.
    return [
        "-p", "ProtectSystem=strict",
        "-p", "ProtectHome=read-only",
        "-p", "NoNewPrivileges=yes",
        "-p", "PrivateTmp=yes",
        "-p", f"ReadWritePaths=-{writable}",
    ]


def run_sandboxed(binary, args, writable):
    """One short command under the unit's restrictions; returns (exit code, output)."""
    out = subprocess.run(
        ["systemd-run", "--user", "--pipe", "--wait", "--collect", "-q", *sandbox_props(writable), str(binary), *args],
        capture_output=True, text=True, timeout=120,
    )
    return out.returncode, out.stdout + out.stderr


def cli(binary, cfg, *args, timeout=120):
    env = dict(os.environ, ELIEZER_TIMEOUT_SECS="60")
    out = subprocess.run([str(binary), "--config", str(cfg), *args], capture_output=True, text=True, timeout=timeout, env=env)
    return out.returncode, out.stdout + out.stderr


def serve_and_drive(binary, cfg, writable, key="m"):
    """Starts the service under the unit's restrictions and drives it with the client."""
    unit = f"eliezer-layout-{os.getpid()}-{int(time.time() * 1000) % 100000}"
    subprocess.run(
        ["systemd-run", "--user", f"--unit={unit}", "--collect", "-q", *sandbox_props(writable), str(binary), "--config", str(cfg)],
        check=True,
    )
    try:
        for _ in range(100):
            rc, _ = cli(binary, cfg, "status", timeout=20)
            if rc == 0:
                break
            time.sleep(0.2)
        else:
            journal = subprocess.run(["journalctl", "--user", "-u", unit, "-n", "15", "--no-pager", "-o", "cat"], capture_output=True, text=True).stdout
            return False, "the service did not come up: " + journal.strip()[-300:]
        rc, listing = cli(binary, cfg, "ls")
        if rc != 0 or key not in listing:
            return False, f"ls failed: {listing.strip()[-200:]}"
        rc, loaded = cli(binary, cfg, "load", key)
        if rc != 0:
            return False, f"load failed: {loaded.strip()[-300:]}"
        rc, answer = cli(binary, cfg, "run", key, "Say hello.", "--max-tokens", "12", "--temperature", "0")
        if rc != 0 or not answer.strip():
            return False, f"run failed: {answer.strip()[-200:]}"
        rc, _ = cli(binary, cfg, "unload", "--all")
        return rc == 0, f"answer {answer.strip()[:30]!r}"
    finally:
        subprocess.run(["systemctl", "--user", "stop", unit], capture_output=True)
        subprocess.run(["systemctl", "--user", "reset-failed", unit], capture_output=True)


def fresh():
    shutil.rmtree(BASE, ignore_errors=True)
    BASE.mkdir(parents=True)


def link_model(model, directory, name="Mistral.gguf"):
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / name
    os.link(model, target)
    return target


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("--binary", type=Path, default=Path("target/gpu/release/eliezer"))
    args = parser.parse_args()
    model, binary = args.model.resolve(), args.binary.resolve()
    if not str(model).startswith(str(HOME)):
        sys.exit("the model file must be below your home directory (hard links need one file system)")

    def layout(name, build):
        """build(base) -> (config text, writable path); then the service is run and driven."""
        fresh()
        port = free_port()
        text, writable = build(port)
        cfg = BASE / "config.toml"
        cfg.write_text(text)
        ok, detail = serve_and_drive(binary, cfg, writable)
        check(name, ok, detail)

    # 1. the default shape: a real directory
    def real(port):
        f = link_model(model, BASE / "lib/models")
        return config(port, BASE / "lib/models", {"m": f}), BASE / "lib"
    layout("1  real directories (the default shape)", real)

    # 2. the state directory itself is a link to another disk (the case that broke the unit)
    def lib_link(port):
        link_model(model, BASE / "disk2/models")
        (BASE / "lib").symlink_to(BASE / "disk2")
        return config(port, BASE / "lib/models", {"m": BASE / "lib/models/Mistral.gguf"}), BASE / "lib"
    layout("2  /var/lib/eliezer is a symbolic link to another directory", lib_link)

    # 3. only the model directory was moved; a link stays at the old place
    def models_link(port):
        link_model(model, BASE / "disk2/models")
        (BASE / "lib").mkdir()
        (BASE / "lib/models").symlink_to(BASE / "disk2/models")
        return config(port, BASE / "lib/models", {"m": BASE / "lib/models/Mistral.gguf"}), BASE / "lib"
    layout("3  models moved, a link left at the old place", models_link)

    # 4. same, but the configuration names the physical paths of the new place
    def physical(port):
        link_model(model, BASE / "disk2/models")
        (BASE / "lib").mkdir()
        (BASE / "lib/models").symlink_to(BASE / "disk2/models")
        return config(port, BASE / "lib/models", {"m": BASE / "disk2/models/Mistral.gguf"}), BASE / "lib"
    layout("4  link at the old place, physical paths in the model entries", physical)

    # 5. the model store lives somewhere else entirely, named in the configuration
    def elsewhere(port):
        f = link_model(model, BASE / "other/place/models")
        return config(port, BASE / "other/place/models", {"m": f}), BASE / "lib"
    layout("5  the model store configured at another place", elsewhere)

    # 6. a link inside the store that stays inside it
    def inner(port):
        link_model(model, BASE / "lib/models/real")
        (BASE / "lib/models/alias").symlink_to(BASE / "lib/models/real")
        return config(port, BASE / "lib/models", {"m": BASE / "lib/models/alias/Mistral.gguf"}), BASE / "lib"
    layout("6  a link inside the store that stays inside", inner)

    # 7. the file is moved while the model is loaded: the loaded model keeps working
    fresh()
    port = free_port()
    f = link_model(model, BASE / "lib/models")
    cfg = BASE / "config.toml"
    cfg.write_text(config(port, BASE / "lib/models", {"m": f}))
    unit = f"eliezer-layout-move-{os.getpid()}"
    subprocess.run(["systemd-run", "--user", f"--unit={unit}", "--collect", "-q", *sandbox_props(BASE / "lib"), str(binary), "--config", str(cfg)], check=True)
    try:
        for _ in range(100):
            if cli(binary, cfg, "status", timeout=20)[0] == 0:
                break
            time.sleep(0.2)
        rc, _ = cli(binary, cfg, "load", "m")
        os.makedirs(BASE / "moved", exist_ok=True)
        os.rename(f, BASE / "moved/Mistral.gguf")
        rc2, answer = cli(binary, cfg, "run", "m", "Say hello.", "--max-tokens", "8", "--temperature", "0")
        check("7a the model file is moved away while the model is loaded", rc == 0 and rc2 == 0 and answer.strip(), f"answer {answer.strip()[:30]!r}")
        cli(binary, cfg, "unload", "--all")
        rc3, message = cli(binary, cfg, "load", "m")
        check("7b loading the moved file fails with a clear message, the service stays up", rc3 != 0 and "model file" in message.lower() and "moved" in message.lower(), message.strip()[:160])
        check("7c ... and the service still answers", cli(binary, cfg, "status")[0] == 0)
        os.rename(BASE / "moved/Mistral.gguf", f)
        rc4, _ = cli(binary, cfg, "load", "m")
        check("7d moving the file back makes it loadable again without a restart", rc4 == 0)
        cli(binary, cfg, "unload", "--all")
    finally:
        subprocess.run(["systemctl", "--user", "stop", unit], capture_output=True)
        subprocess.run(["systemctl", "--user", "reset-failed", unit], capture_output=True)

    # 8. configuration problems of moved directories are reported, not swallowed
    fresh()
    port = free_port()
    (BASE / "lib/models").mkdir(parents=True)
    cfg = BASE / "config.toml"
    # 8a the file named in the configuration is not there (moved without updating the entry)
    cfg.write_text(config(port, BASE / "lib/models", {"m": BASE / "lib/models/gone.gguf"}))
    rc, out = run_sandboxed(binary, ["--check-config", str(cfg)], BASE / "lib")
    check("8a a model file that is not there is reported by --check-config", "gone.gguf" in out and "not found" in out.lower(), out.strip()[:200])
    # 8b a link inside the store that leaves the store is refused, and the message says what to do
    link_model(model, BASE / "outside")
    (BASE / "lib/models/leave").symlink_to(BASE / "outside")
    cfg.write_text(config(port, BASE / "lib/models", {"m": BASE / "lib/models/leave/Mistral.gguf"}))
    rc, out = run_sandboxed(binary, ["--check-config", str(cfg)], BASE / "lib")
    check("8b a link that leaves the store is refused with advice", rc == 2 and "models_dir" in out and "link" in out.lower(), out.strip()[:260])
    # 8c the store directory is gone
    cfg.write_text(config(port, BASE / "nowhere/models", {"m": BASE / "nowhere/models/x.gguf"}))
    rc, out = run_sandboxed(binary, ["--check-config", str(cfg)], BASE / "lib")
    check("8c a missing model store is reported with its path", "nowhere" in out, out.strip()[:200])

    failed = [n for n, ok in RESULTS if not ok]
    print(f"\n{len(RESULTS) - len(failed)} of {len(RESULTS)} checks passed")
    shutil.rmtree(BASE, ignore_errors=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
