#!/usr/bin/env python3
"""Image-text matching server for the VLFM-style search: BLIP-2's ITC head (the same model and head VLFM uses,
`blip2_image_text_matching` / match_head "itc"), via transformers' Blip2ForImageTextRetrieval.

  POST /score   {"frame": base64 JPEG, "text": str}  -> {"cosine": float, "latency_ms": float}
  GET  /health  -> {"ok": true, "model": ..., "gpu_now_gb": ...}
The cosine is how well the whole picture matches the sentence ("Seems like there is a refrigerator ahead."): a
promise about the direction, not a detection. Env: ITM_MODEL (Salesforce/blip2-itm-vit-g), ITM_PORT (8091),
HF_HOME (cache), HF_HUB_OFFLINE=1 once cached.
"""
import base64
import io
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch
from PIL import Image
from transformers import AutoProcessor, Blip2ForImageTextRetrieval

MODEL = os.environ.get("ITM_MODEL", "Salesforce/blip2-itm-vit-g")
PORT = int(os.environ.get("ITM_PORT", "8091"))


class ITM:
    def __init__(self):
        t = time.time()
        self.processor = AutoProcessor.from_pretrained(MODEL)
        self.model = Blip2ForImageTextRetrieval.from_pretrained(MODEL, dtype=torch.float16).to("cuda").eval()
        self.lock = threading.Lock()
        print(f"[itm] {MODEL} loaded in {time.time() - t:.1f} s, {torch.cuda.memory_allocated() / 2**30:.2f} GB", flush=True)

    def cosine(self, image: Image.Image, text: str) -> float:
        inputs = self.processor(images=image, text=text, return_tensors="pt").to("cuda", torch.float16)
        with self.lock, torch.inference_mode():
            out = self.model(**inputs, use_image_text_matching_head=False)   # ITC: cosine, not the ITM classifier
        return float(out.logits_per_image[0, 0])


def serve(itm: ITM):
    class H(BaseHTTPRequestHandler):
        def _send(self, code, obj):
            body = json.dumps(obj).encode()
            self.send_response(code); self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)

        def do_GET(self):
            if self.path == "/health":
                return self._send(200, {"ok": True, "model": MODEL,
                                        "gpu_now_gb": round(torch.cuda.memory_allocated() / 2**30, 2)})
            self._send(404, {"error": "not found"})

        def do_POST(self):
            if self.path != "/score":
                return self._send(404, {"error": "not found"})
            try:
                req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                img = Image.open(io.BytesIO(base64.b64decode(req["frame"]))).convert("RGB")
                t = time.monotonic()
                c = itm.cosine(img, str(req["text"]))
                self._send(200, {"cosine": round(c, 5), "latency_ms": round((time.monotonic() - t) * 1000, 1)})
            except Exception as e:
                self._send(500, {"error": f"{type(e).__name__}: {e}"})

        def log_message(self, fmt, *args):
            pass

    print(f"[itm] serving on 127.0.0.1:{PORT}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", PORT), H).serve_forever()


def main():
    itm = ITM()
    grey = Image.new("RGB", (640, 480), (128, 128, 128))
    t = time.monotonic()
    c = itm.cosine(grey, "Seems like there is a ladder ahead.")
    print(f"[itm] warm-up done: cosine {c:.4f} in {(time.monotonic() - t) * 1000:.0f} ms", flush=True)
    serve(itm)


if __name__ == "__main__":
    main()
