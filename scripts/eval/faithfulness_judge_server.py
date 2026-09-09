#!/usr/bin/env python3
"""A local judge model as a generic chat endpoint, for the NL->FL faithfulness
metric.

WHY A SERVER AND NOT AN IMPORT. The judge is a 7-8B model holding ~15GB of
weights, and it must not be constructed once per Lean worker: the GRPO agent
loop runs 24 of them. One process owns the model and one GPU; everything else
talks HTTP to it.

WHY A GENERIC ENDPOINT. The faithfulness metric needs two DIFFERENT prompts
(AutoFaith's NL-block extraction, then its whole-proof judgement), so a
`/judge`-shaped endpoint with a prompt baked in cannot serve it. `/generate`
takes `messages`, or `system` + `user`.

THE LOADER IS IMPORTED, NOT COPIED. `utils.hf_judge.build_chat_generate` handles
the ChatML fallback for a model that ships no chat template, bf16, the vLLM /
transformers backend choice, and the request batching without which this cannot
keep up with an RL agent loop. Reimplementing it here would mean two judges that
differ in ways nobody tracked.

TWO JUDGES, ONE PER SIDE, DELIBERATELY DIFFERENT. See `--model` below.

    source hpc/cc_env.sh
    python scripts/eval/faithfulness_judge_server.py \
        --model Qwen/Qwen2.5-7B-Instruct --port 8732
    curl -s localhost:8732/health          # {"ok": true, "model": ...}
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

# Read-only import from the sibling project; nothing there is modified.
MOME_ROOT = Path(os.environ.get(
    "MOME_ROOT", Path.home() / "workspace" / "MixtureOfMathExperts"))
sys.path.insert(0, str(MOME_ROOT / "scripts"))

_GEN = None
# NO LOCK HERE ANY MORE. `utils.hf_judge` batches concurrent requests into one
# `vllm.generate` call, which is the only reason this endpoint can keep up with
# an RL agent loop -- a mutex around it would serialise every caller again and
# throw the batching away. The HF fallback backend keeps its own lock, because
# `transformers.generate` really is one-at-a-time.
_STATS = {"calls": 0, "errors": 0, "dropped": 0, "total_s": 0.0, "model": None}


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):    # keep stdout for real logging
        pass

    def _send(self, code: int, payload: dict) -> bool:
        """Write one response. False if the client already hung up.

        A DISCONNECT IS NORMAL, NOT AN ERROR. `reward/faithfulness.py` gives up
        after FAITH_JUDGE_TIMEOUT and closes the socket, which is the correct
        behaviour on its side -- it records `no_judge` (unknown) and moves on.
        Without this the write raises BrokenPipeError, the handler's `except`
        then tries to report the failure by sending a 500 ON THE SAME DEAD
        SOCKET, and that raises too: two full tracebacks per slow generation.
        Observed on the Phi-4 gate, which runs ~104s/row against a 180s client
        timeout, so the margin is thin enough for this to happen routinely.
        """
        body = json.dumps(payload).encode()
        try:
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return True
        except (BrokenPipeError, ConnectionResetError):
            _STATS["dropped"] = _STATS.get("dropped", 0) + 1
            return False

    def do_GET(self):
        if self.path == "/health":
            mean = _STATS["total_s"] / _STATS["calls"] if _STATS["calls"] else 0.0
            # `model` is in here so a caller can assert WHICH judge answered:
            # the training and eval judges must differ, and a stale server left
            # running on the wrong port is otherwise invisible.
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

        # Optional JSON schema. Passed straight to the backend, which turns it
        # into vLLM constrained decoding -- the only reliable way to stop a math
        # model writing unquoted Lean notation into a JSON value.
        schema = req.get("json_schema")
        if schema is not None and not isinstance(schema, dict):
            self._send(400, {"error": "`json_schema` must be an object"})
            return

        t0 = time.time()
        try:
            raw = _GEN(messages, temperature=float(req.get("temperature") or 0.0),
                       json_schema=schema)
            _STATS["calls"] += 1
            self._send(200, {"text": raw})
        except Exception as e:
            # A generation failure is not a verdict. The caller maps this to
            # "unknown", never to a score -- see reward/faithfulness.py.
            # `_send` returns False rather than raising if the client is gone.
            _STATS["errors"] += 1
            self._send(500, {"error": f"{type(e).__name__}: {e}"})
        finally:
            _STATS["total_s"] += time.time() - t0


def main() -> None:
    ap = argparse.ArgumentParser()
    # FormalRx-8b, not the 4b this file first named: the 8b is what is resident
    # in the offline HF cache on Narval, and compute nodes have no internet.
    # The judge is chosen per SIDE, and the two sides must not share one --
    # scoring a policy with the judge it was trained against measures how well
    # it learned to please that judge, not faithfulness. Current assignment:
    #   training loop  Qwen/Qwen2.5-7B-Instruct   (generic, cheap, gets gamed)
    #   evaluation     LARK-Lab/FormalRx-8b       (Lean-specialised, held out)
    # Both are Qwen-lineage, so their errors are not fully independent; that is
    # a stated caveat, not a solved problem. No usable non-Qwen judge is cached
    # here (gpt-oss-20b is MXFP4, internlm2 needs a transformers-4.x remote code).
    ap.add_argument("--model", default="LARK-Lab/FormalRx-8b")
    ap.add_argument("--port", type=int, default=8732)
    # 4096, not 2048. MEASURED on the first faith_eval run: 8 of the first 14
    # rows came back `bad_json` from the NL-block step alone, at 30-58s each --
    # i.e. the reply was long, and a JSON object cut off by the token budget is
    # returned as "the judge produced nothing usable", which is indistinguishable
    # from a real coverage limit. That would have made the validity gate measure
    # our own budget instead of the structural term-mode ceiling it exists to
    # measure. 24576 + 4096 = 28672 still fits BOTH judges' context
    # (FormalRx-8b 40960, Qwen2.5-7B-Instruct 32768).
    ap.add_argument("--max-new-tokens", type=int, default=4096)
    # AutoFaith's prompts are ~2.5k and ~1.9k tokens BEFORE the blocks are
    # interpolated, and FLBlock.to_dict recurses full premises and goals at every
    # node, so the real prompt is several times either figure. An over-long
    # prompt RAISES rather than truncating (utils.hf_judge), because the blocks
    # being judged sit at the TAIL and a silent cut would delete exactly them.
    # This is a request cap, not a context claim: hf_judge separately clamps
    # vLLM's max_model_len to whatever the model's own config allows
    # (FormalRx-8b 40960, Qwen2.5-7B-Instruct 32768) and says so at load.
    ap.add_argument("--max-input-tokens", type=int, default=24576)
    ap.add_argument("--backend", choices=("vllm", "hf"), default=None,
                    help="default: vllm when vllm._C imports, else transformers")
    args = ap.parse_args()

    global _GEN
    print(f"[faith-judge] loading {args.model} ...", flush=True)
    from utils.hf_judge import build_chat_generate
    _GEN = build_chat_generate(args.model, max_new_tokens=args.max_new_tokens,
                               max_input_tokens=args.max_input_tokens,
                               enable_thinking=False, sampling=True,
                               backend=args.backend)
    _STATS["model"] = args.model
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
