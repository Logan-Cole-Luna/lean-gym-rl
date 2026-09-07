#!/usr/bin/env python3
"""FormalRx-4b as a generic chat endpoint, for the NL->FL faithfulness metric.

WHY A SERVER AND NOT AN IMPORT. A 4B judge in bf16 is ~8GB and the card is 16GB.
`MixtureOfMathExperts/scripts/judge_server.py` already loads this exact model and
says so in its own docstring: "a 4B judge loaded twice does not fit alongside
anything else on a 16 GB card." So this process is an ALTERNATIVE to that one,
not a companion -- stop that server before starting this one.

WHY NOT JUST CALL THAT SERVER. It exposes `/judge` and `/probe` only, both with
prompts hardcoded for statement alignment. The faithfulness metric needs two
different prompts (AutoFaith's NL-block extraction and whole-proof judgement), so
it needs a generic endpoint. Rather than edit a sibling project's running
service, this serves the same model through the same loader.

THE LOADER IS IMPORTED, NOT COPIED. `utils.hf_judge.build_chat_generate` handles
the Qwen3 chat template, the missing-template fallback, bf16 + device_map, and
greedy decoding. Reimplementing it here would mean two judges that differ in
ways nobody tracked.

    # stop the other server first, then:
    MOME_ROOT=~/workspace/MixtureOfMathExperts \
      ~/workspace/MixtureOfMathExperts/.venv_judge/bin/python \
      scripts/eval/faithfulness_judge_server.py --port 8732
    curl -s localhost:8732/health
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

# Read-only import from the sibling project; nothing there is modified.
MOME_ROOT = Path(os.environ.get(
    "MOME_ROOT", Path.home() / "workspace" / "MixtureOfMathExperts"))
sys.path.insert(0, str(MOME_ROOT / "scripts"))

_GEN = None
_LOCK = threading.Lock()          # one model, one GPU: serialise generation
_STATS = {"calls": 0, "errors": 0, "total_s": 0.0}


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):    # keep stdout for real logging
        pass

    def _send(self, code: int, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            mean = _STATS["total_s"] / _STATS["calls"] if _STATS["calls"] else 0.0
            self._send(200, {"ok": _GEN is not None, "mean_s": round(mean, 2), **_STATS})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/generate":
            self._send(404, {"error": "not found"})
            return
        n = int(self.headers.get("Content-Length", 0))
        try:
            req = json.loads(self.rfile.read(n) or b"{}")
        except Exception as e:
            self._send(400, {"error": f"bad json: {e}"})
            return

        # Either `messages`, or `system` + `user` for convenience.
        messages = req.get("messages")
        if not messages:
            user = req.get("user") or req.get("prompt")
            if not user:
                self._send(400, {"error": "need `messages`, or `user`/`prompt`"})
                return
            messages = ([{"role": "system", "content": req["system"]}]
                        if req.get("system") else []) + [{"role": "user", "content": user}]

        t0 = time.time()
        try:
            with _LOCK:
                raw = _GEN(messages, temperature=float(req.get("temperature") or 0.0))
            _STATS["calls"] += 1
            self._send(200, {"text": raw})
        except Exception as e:
            # A generation failure is not a verdict. The caller maps this to
            # "unknown", never to a score -- see reward/faithfulness.py.
            _STATS["errors"] += 1
            self._send(500, {"error": f"{type(e).__name__}: {e}"})
        finally:
            _STATS["total_s"] += time.time() - t0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="LARK-Lab/FormalRx-4b")
    ap.add_argument("--port", type=int, default=8732)
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    # AutoFaith's prompts are ~2.5k and ~1.9k tokens BEFORE the blocks are
    # interpolated, and FLBlock.to_dict recurses full premises and goals at every
    # node. hf_judge defaults to 8192 and truncates silently, which would drop
    # the FL blocks off the end of the judge prompt. FormalRx-4b has 262k context.
    ap.add_argument("--max-input-tokens", type=int, default=32768)
    args = ap.parse_args()

    global _GEN
    print(f"[faith-judge] loading {args.model} ...", flush=True)
    from utils.hf_judge import build_chat_generate
    _GEN = build_chat_generate(args.model, max_new_tokens=args.max_new_tokens,
                               max_input_tokens=args.max_input_tokens,
                               enable_thinking=False, sampling=True)
    # Warm before announcing readiness: the first call compiles kernels and would
    # otherwise be attributed to the metric's latency.
    try:
        _GEN([{"role": "user", "content": "Reply with OK."}])
    except Exception as e:
        print(f"[faith-judge] warmup failed: {e}", flush=True)

    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"[faith-judge] ready on http://127.0.0.1:{args.port}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
