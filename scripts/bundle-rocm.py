#!/usr/bin/env python3
"""Bundle the ROCm runtime closure and GPU kernels in an existing Eliezer deb."""
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile


def output(*args, **kwargs):
    return subprocess.check_output(args, text=True, **kwargs)


def main():
    # Usage: bundle-rocm.py PACKAGE.deb [--sd-server PATH]
    arguments = sys.argv[1:]
    sd_server = None
    if "--sd-server" in arguments:
        at = arguments.index("--sd-server")
        sd_server = Path(arguments[at + 1]).resolve()
        del arguments[at : at + 2]
    package = Path(arguments[0]).resolve()
    rocm = Path(os.environ.get("ROCM_PATH", "/opt/rocm")).resolve()
    triplet = output("dpkg-architecture", "-qDEB_HOST_MULTIARCH").strip()
    # libomp, which sd-server needs, lives beside the ROCm compiler.
    environment = dict(os.environ, LD_LIBRARY_PATH=f"{rocm / 'lib'}:{rocm / 'lib/llvm/lib'}")
    with tempfile.TemporaryDirectory(prefix="eliezer-bundle-", dir=package.parent) as temporary:
        root = Path(temporary) / "root"
        subprocess.run(["dpkg-deb", "--raw-extract", str(package), str(root)], check=True)
        private = root / "usr/lib" / triplet / "eliezer"
        private.mkdir(parents=True, exist_ok=True)
        executable = root / "usr/bin/eliezer"
        pending = [executable, rocm / "lib/libhiprtc.so", rocm / "lib/libhiprtc-builtins.so"]
        # The image runtime is a second program in the package; its libraries join the same private directory.
        drawing = None
        if sd_server is not None:
            # A program run by Eliezer, not by users: the multiarch libexec directory of the package.
            drawing = root / "usr/libexec" / triplet / "eliezer" / "sd-server"
            drawing.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(sd_server, drawing)
            drawing.chmod(0o755)
            pending.append(drawing)
        copied = {}
        system = set()
        inspected = set()
        while pending:
            path = pending.pop().resolve()
            if path in inspected:
                continue
            inspected.add(path)
            if path not in (executable, drawing):
                destination = private / path.name
                shutil.copy2(path, destination)
                soname = output("patchelf", "--print-soname", str(path)).strip()
                for name in {path.name, soname} - {""}:
                    alias = private / name
                    if alias != destination and not alias.exists():
                        alias.symlink_to(destination.name)
                copied[path.name] = {"source": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            listing = output("ldd", str(path), env=environment)
            if "not found" in listing:
                raise RuntimeError(f"Unresolved dependencies for {path}:\n{listing}")
            for line in listing.splitlines():
                match = re.search(r"=> (/\S+)", line)
                if not match:
                    continue
                dependency = Path(match[1]).resolve()
                if dependency.is_relative_to(rocm) or str(dependency).startswith("/opt/amdgpu/"):
                    pending.append(dependency)
                else:
                    system.add(dependency.name)
        # rocBLAS and hipBLASLt load their architecture-specific code objects at runtime.
        # Preserve their complete trees, including metadata; do not infer these from ldd.
        for component in ["rocblas", "hipblaslt"]:
            source = rocm / "lib" / component
            if not source.is_dir():
                raise RuntimeError(f"Missing kernel assets: {source}")
            shutil.copytree(source, private / component, symlinks=False)
        for library in private.glob("*.so*"):
            if not library.is_symlink():
                subprocess.run(["patchelf", "--set-rpath", "$ORIGIN", str(library)], check=True)
        subprocess.run(["patchelf", "--set-rpath", f"$ORIGIN/../lib/{triplet}/eliezer", str(executable)], check=True)
        if drawing is not None:
            subprocess.run(["patchelf", "--set-rpath", f"$ORIGIN/../../../lib/{triplet}/eliezer", str(drawing)], check=True)
            # stable-diffusion.cpp and ggml are MIT licensed; their license text goes with the program.
            licence = sd_server.parents[2] / "src/LICENSE"
            if licence.is_file():
                target = root / "usr/share/doc/eliezer/stable-diffusion.cpp-LICENSE"
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(licence, target)
        licenses = root / "usr/share/doc/eliezer/rocm-licenses"
        for source in (rocm / "share/doc").rglob("*"):
            if source.is_file() and ("license" in source.name.lower() or "copyright" in source.name.lower()):
                destination = licenses / source.relative_to(rocm / "share/doc")
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
        for source in Path("/usr/share/doc").glob("libdrm*/copyright"):
            destination = licenses / source.parent.name / "copyright"
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        manifest = {"directory": f"/usr/lib/{triplet}/eliezer", "libraries": copied,
                    "kernel_assets": ["rocblas/library", "hipblaslt/library"], "system_libraries": sorted(system)}
        (root / "usr/share/doc/eliezer/bundled-libraries.json").write_text(json.dumps(manifest, indent=2) + "\n")
        control = root / "DEBIAN/control"
        text = control.read_text()
        size = sum(p.stat().st_size for p in root.rglob("*") if p.is_file() and not p.is_symlink())
        text = re.sub(r"^Installed-Size:.*$", f"Installed-Size: {(size + 1023) // 1024}", text, flags=re.M)
        control.write_text(text)
        # Recompute checksums after modifying the ELF files and adding runtime assets.
        checksums = []
        for path in sorted(root.rglob("*")):
            relative = path.relative_to(root)
            if path.is_file() and not path.is_symlink() and relative.parts[0] != "DEBIAN":
                checksums.append(f"{hashlib.md5(path.read_bytes()).hexdigest()}  {relative}\n")
        (root / "DEBIAN/md5sums").write_text("".join(checksums))
        built = Path(temporary) / package.name
        subprocess.run(["dpkg-deb", "--root-owner-group", "-Zzstd", "-z3", "--threads-max=4", "--build", str(root), str(built)], check=True)
        os.replace(built, package)
        print(f"Bundled {len(copied)} libraries and GPU kernel assets into {package}")


if __name__ == "__main__":
    main()
