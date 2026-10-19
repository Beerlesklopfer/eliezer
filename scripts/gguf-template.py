"""Prints GGUF metadata keys (and optionally writes tokenizer.chat_template) without dependencies.

Usage: python3 -I scripts/gguf-template.py MODEL.gguf [OUT.jinja]
Reads only the header; the tensor data is never touched.
"""
import struct
import sys

SCALARS = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}


def read(f, fmt):
    size = struct.calcsize(fmt)
    return struct.unpack(fmt, f.read(size))[0]


def read_string(f):
    return f.read(read(f, "<Q")).decode("utf-8", "replace")


def read_value(f, kind):
    if kind in SCALARS:
        return read(f, SCALARS[kind])
    if kind == 8:
        return read_string(f)
    if kind == 9:
        inner, count = read(f, "<I"), read(f, "<Q")
        return [read_value(f, inner) for _ in range(count)]
    raise ValueError(f"unknown GGUF value type {kind}")


def metadata(path):
    with open(path, "rb") as f:
        if f.read(4) != b"GGUF":
            raise ValueError("not a GGUF file")
        read(f, "<I")  # version
        read(f, "<Q")  # tensor count
        count = read(f, "<Q")
        for _ in range(count):
            key = read_string(f)
            yield key, read_value(f, read(f, "<I"))


if __name__ == "__main__":
    wanted = {}
    for key, value in metadata(sys.argv[1]):
        if key in ("general.name", "general.architecture", "tokenizer.chat_template", "tokenizer.ggml.model") or key.endswith((".context_length", ".block_count", ".embedding_length")):
            wanted[key] = value
    if len(sys.argv) > 2 and "tokenizer.chat_template" in wanted:
        open(sys.argv[2], "w", encoding="utf-8").write(wanted["tokenizer.chat_template"])
    for key, value in wanted.items():
        shown = f"<{len(value)} chars>" if key == "tokenizer.chat_template" else value
        print(f"{key} = {shown}")
