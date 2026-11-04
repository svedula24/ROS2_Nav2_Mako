#!/usr/bin/env python3
"""Real-time B VLM server: Cosmos-Reason1-7B (zero-shot) answers "what's the next move?" from camera frames.

Loads the model once (bf16, GPU) and serves:
  POST /decide  {"instruction": str, "frames": [base64 JPEG, oldest first, last = current view],
                 "history": [moves already executed, as text] (optional), "max_new_tokens": 24 (optional)}
             -> {"text": str, "latency_ms": float, "prompt_tokens": int, "images": int}
  GET  /health  -> {"ok": true, "model": ..., "warm": bool, "gpu_peak_gb": float, "gpu_now_gb": float}
The reply is free text in the move grammar (src/m20_vlmnav/m20_vlmnav/moves.py parses it; nothing here interprets it).
Reasoning is turned down on purpose: Cosmos-Reason thinks out loud by default (its card suggests 4096+ output tokens),
which is far too slow for a decision every second, so the prompt asks for the move only and output is capped.

Env: COSMOS_MODEL (nvidia/Cosmos-Reason1-7B), COSMOS_PORT (8090), COSMOS_MAX_SIDE (448: frames are downscaled so
the longer side is at most this; 448x336 = 192 image tokens each), HF_HOME (model cache), HF_HUB_OFFLINE=1 once cached.
  python3 cosmos_server.py                  serve
  python3 cosmos_server.py --selftest a.jpg one decision on 8 copies of a.jpg, print it and the timings, exit
"""
import base64
import io
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch
import transformers
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

# `dtype=` is honoured from 4.56; older versions ignore it and load fp32 (~30 GB: an OOM instead of a model)
assert tuple(int(x) for x in transformers.__version__.split(".")[:2]) >= (4, 56), \
    f"transformers >= 4.56 needed for dtype=bfloat16, found {transformers.__version__}"
MAX_NEW_TOKENS = 64

MODEL = os.environ.get("COSMOS_MODEL", "nvidia/Cosmos-Reason1-7B")
PORT = int(os.environ.get("COSMOS_PORT", "8090"))
MAX_SIDE = int(os.environ.get("COSMOS_MAX_SIDE", "448"))

SYSTEM = ("You control a four-legged robot walking indoors. You see only its forward camera. "
          "You answer with exactly one action and nothing else: no reasoning, no explanation.")
USER = ("Instruction: \"{instruction}\"\n"
        "Above are {n} frames from the robot's forward camera, oldest first; the last frame is the current view.\n"
        "Actions already taken for this instruction, in order: {history}\n"
        "What is the robot's next action? If the actions taken already complete the instruction, answer stop. "
        "Answer with exactly one of these forms:\n"
        "  move forward X cm\n  turn left X degrees\n  turn right X degrees\n  stop\n"
        "X is a number you choose (for example 25, 50 or 75 cm; 15, 30 or 45 degrees). "
        "Answer \"stop\" only when the instruction is complete. The robot cannot move backward: "
        "to get out of a dead end, turn around.")


class Cosmos:
    def __init__(self):
        t = time.time()
        self.processor = AutoProcessor.from_pretrained(MODEL)
        # no device_map: it needs `accelerate`, which the image lacks; load in bf16 and move (16 GB, a few seconds)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL, dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()
        self.lock = threading.Lock()                       # one generation at a time (one GPU, one robot)
        self.warm = False
        print(f"[cosmos] {MODEL} loaded in {time.time() - t:.1f} s, "
              f"{torch.cuda.memory_allocated() / 2**30:.1f} GB on the GPU", flush=True)

    @staticmethod
    def _image(b64: str) -> Image.Image:
        im = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
        s = MAX_SIDE / max(im.size)
        if s < 1:
            im = im.resize((round(im.width * s), round(im.height * s)), Image.BILINEAR)
        return im

    def decide(self, instruction: str, frames_b64, max_new_tokens: int = 24, history=None) -> dict:
        """history: the moves already executed for this instruction, as text ("turn left 90 degrees",
        "move forward 50 cm (blocked by an obstacle)"). The frames are the model's memory of what it saw; this is
        its memory of what it did, so a completed 'turn left 90 degrees' can end in 'stop'."""
        images = [self._image(f) for f in frames_b64]
        hist = "; ".join(f"{i + 1}. {h}" for i, h in enumerate(history or [])) or "none yet"
        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
            {"role": "user", "content": [{"type": "image"} for _ in images]
             + [{"type": "text", "text": USER.format(instruction=instruction, n=len(images), history=hist)}]},
        ]
        prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[prompt], images=images, return_tensors="pt").to(self.model.device)
        with self.lock, torch.inference_mode():
            t = time.monotonic()
            out = self.model.generate(**inputs, max_new_tokens=max(1, min(int(max_new_tokens), MAX_NEW_TOKENS)),
                                      do_sample=False)
            latency = (time.monotonic() - t) * 1000.0
        new = out[0, inputs["input_ids"].shape[1]:]
        text = self.processor.decode(new, skip_special_tokens=True).strip()
        self.warm = True
        return {"text": text, "latency_ms": round(latency, 1), "prompt_tokens": int(inputs["input_ids"].shape[1]),
                "images": len(images)}


def gpu():
    return {"gpu_peak_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
            "gpu_now_gb": round(torch.cuda.memory_allocated() / 2**30, 2)}


def serve(cosmos: Cosmos):
    class H(BaseHTTPRequestHandler):
        def _send(self, code, obj):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/health":
                return self._send(200, {"ok": True, "model": MODEL, "warm": cosmos.warm, **gpu()})
            self._send(404, {"error": "not found"})

        def do_POST(self):
            if self.path != "/decide":
                return self._send(404, {"error": "not found"})
            try:
                req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                if not req.get("frames"):
                    return self._send(400, {"error": "no frames"})
                out = cosmos.decide(str(req.get("instruction", "")), req["frames"], int(req.get("max_new_tokens", 24)),
                                    [str(h) for h in req.get("history") or []])
                self._send(200, {**out, **gpu()})
            except Exception as e:                          # report, don't crash the server
                self._send(500, {"error": f"{type(e).__name__}: {e}"})

        def log_message(self, fmt, *args):                  # one line per decision is printed by decide's caller
            pass

    print(f"[cosmos] serving on 127.0.0.1:{PORT}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", PORT), H).serve_forever()


def main():
    cosmos = Cosmos()
    if len(sys.argv) >= 3 and sys.argv[1] == "--selftest":
        b64 = base64.b64encode(open(sys.argv[2], "rb").read()).decode()
        for n in (1, 8, 8):                                 # the first call also pays warm-up
            r = cosmos.decide("go to the ladder", [b64] * n)
            print(f"[selftest] {n} frame(s): {r} {gpu()}", flush=True)
        return
    # warm-up: one real decision on a grey frame, so the first instruction doesn't pay CUDA init / graph setup
    buf = io.BytesIO()
    Image.new("RGB", (640, 480), (128, 128, 128)).save(buf, format="JPEG")
    r = cosmos.decide("stop", [base64.b64encode(buf.getvalue()).decode()] * 8)
    print(f"[cosmos] warm-up done: {r} {gpu()}", flush=True)
    serve(cosmos)


if __name__ == "__main__":
    main()
