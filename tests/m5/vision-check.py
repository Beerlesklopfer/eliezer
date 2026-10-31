"""Live check of image input against a running Eliezer (manual GPU test, not part of `cargo test`).

Usage: python3 -I tests/m5/vision-check.py BASE_URL VISION_MODEL_KEY [TEXT_ONLY_MODEL_KEY]

A service with API keys needs the key in the environment (ELIEZER_API_KEY); it is never an argument.

Draws a page with a few lines of text (needs Pillow), sends it as a base64 data URI and checks that
the model reads the numbers back. Also checks the refusals: a link instead of data, and (when a
second model key is given) a model without a projector.
"""
import base64
import io
import json
import os
import sys
import urllib.error
import urllib.request

from PIL import Image, ImageDraw, ImageFont

LINES = ["INVOICE 4711", "Date: 2026-10-20", "Total: 89.50 EUR"]


def page() -> bytes:
    image = Image.new("RGB", (900, 360), "white")
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 54)
    except OSError:
        font = ImageFont.load_default(size=54)
    for index, line in enumerate(LINES):
        draw.text((40, 30 + index * 100), line, fill="black", font=font)
    out = io.BytesIO()
    image.save(out, "PNG")
    return out.getvalue()


def chat(base: str, model: str, content, **extra) -> tuple[int, dict]:
    body = {"model": model, "messages": [{"role": "user", "content": content}],
            "max_tokens": 120, "temperature": 0, **extra}
    headers = {"Content-Type": "application/json"}
    if os.environ.get("ELIEZER_API_KEY"):
        headers["Authorization"] = "Bearer " + os.environ["ELIEZER_API_KEY"]
    request = urllib.request.Request(base + "/v1/chat/completions", json.dumps(body).encode(), headers)
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as error:
        return error.code, json.load(error)


def main() -> int:
    base, model = sys.argv[1].rstrip("/"), sys.argv[2]
    data = "data:image/png;base64," + base64.b64encode(page()).decode()
    ask = [{"type": "text", "text": "Read all text in this image."},
           {"type": "image_url", "image_url": {"url": data}}]
    failures = []

    status, answer = chat(base, model, ask)
    text = answer.get("choices", [{}])[0].get("message", {}).get("content", "") if status == 200 else str(answer)
    print("answer:", text.strip().replace("\n", " | "), "| usage:", answer.get("usage"))
    if status != 200:
        failures.append(f"image request failed: {status}")
    for needle in ("4711", "89.50"):
        if needle not in text:
            failures.append(f"{needle!r} not found in the answer")

    status, _ = chat(base, model, "Say hello in one word.")
    print("text only:", status)
    if status != 200:
        failures.append("a text-only request to the vision model failed")

    link = [{"type": "text", "text": "x"}, {"type": "image_url", "image_url": {"url": "http://127.0.0.1:1/a.png"}}]
    status, answer = chat(base, model, link)
    print("link:", status, answer.get("error", {}).get("message"))
    if status != 400:
        failures.append(f"a link instead of data was not refused (status {status})")

    if len(sys.argv) > 3:
        status, answer = chat(base, sys.argv[3], ask)
        print("model without projector:", status, answer.get("error", {}).get("message"))
        if status != 400:
            failures.append(f"image sent to a text-only model was not refused (status {status})")

    for failure in failures:
        print("FAIL:", failure)
    print("OK" if not failures else "FAILED")
    return 1 if failures else 0


sys.exit(main())
