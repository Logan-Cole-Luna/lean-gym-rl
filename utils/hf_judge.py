#!/usr/bin/env python3
"""One chat-completion callable for a local judge model, behind one function.

WHY THIS FILE IS HERE. `scripts/eval/faithfulness_judge_server.py` was written
against `MixtureOfMathExperts/scripts/utils/hf_judge.py` and inserts that
sibling project on `sys.path` before importing it. That checkout does not exist
on Narval (`~/workspace` is empty), so the import fell through to nothing and
the server could not start. This module satisfies the same import from the repo
root, which `hpc/job_prelude.sh` already puts on `PYTHONPATH`. The server needs
no edit: when the sibling project IS present its copy still wins, because its
path is inserted at position 0.

TWO BACKENDS, AND THE DEFAULT IS vLLM ON PURPOSE. The faithfulness reward calls
this from inside the GRPO agent loop, where ~24% of rollouts reach the SOLVED row
and each one costs two judge generations. Served one request at a time through
`transformers.generate`, that is the whole step; served through vLLM's continuous
batching it overlaps with itself. `transformers` is kept as a fallback for a
machine where vLLM is broken -- `import vllm` succeeds while vllm is unusable
(see hpc/NARVAL_NOTES.md), so the probe below imports `vllm._C`, not `vllm`.

BATCHING IS DONE HERE, NOT BY THE CALLER. `vllm.LLM.generate` is not safe to
call from several threads at once, and the server's own `_LOCK` would serialise
every request if it were the only guard. `_Batcher` instead collects the
requests that arrive inside a short window and issues them as ONE `generate`
call, so N concurrent callers cost roughly one generation's latency rather than
N. The window is small enough (25ms) to be invisible next to a multi-second
generation, and a lone request is not delayed by more than it.

NO SILENT TRUNCATION. An over-long prompt RAISES. The FL blocks AutoFaith
interpolates land at the END of the judge prompt, so a length cap that quietly
cuts the tail deletes exactly the evidence being judged and returns a confident
verdict about nothing. The caller (reward/faithfulness.py) maps a raised error
to `no_judge`, i.e. UNKNOWN, which is the honest answer.

MISSING CHAT TEMPLATE. `LARK-Lab/FormalRx-8b` ships none, and
`apply_chat_template` raises without one. Its vocabulary is stock Qwen3 ChatML
(`<|im_start|>`, `<|im_end|>`, `<think>`, eos `<|im_end|>`), so `_CHATML` below
reproduces the Qwen3 template's structure rather than inventing a format. Its
`enable_thinking=False` branch is reproduced too: Qwen3 does not suppress
thinking with a flag, it PREFILLS an empty `<think></think>` pair into the
assistant turn, and without that a thinking model spends the whole token budget
reasoning and returns no JSON.
"""
from __future__ import annotations

import os
import threading
import time
from typing import Callable

# Qwen3's own template, reduced to the parts a judge prompt uses (system, user,
# assistant; no tools, no multi-turn thinking history). `add_generation_prompt`
# and the empty-think prefill are the two behaviours that must match, or the
# model answers in the wrong shape.
_CHATML = (
    "{% for m in messages %}"
    "{{ '<|im_start|>' + m['role'] + '\n' + m['content'] + '<|im_end|>\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}"
    "{% if not enable_thinking %}{{ '<think>\n\n</think>\n\n' }}{% endif %}"
    "{% endif %}"
)


def _vllm_usable() -> bool:
    """`import vllm` is not the test -- `_C.abi3.so` loads lazily, so a broken
    install imports fine and fails at first use. See hpc/NARVAL_NOTES.md."""
    try:
        import vllm._C  # noqa: F401
        return True
    except Exception:
        return False


def _position_limit(model: str) -> int | None:
    """The model's own max position count, or None if it does not say."""
    try:
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(model, trust_remote_code=True)
        n = getattr(cfg, "max_position_embeddings", None)
        return int(n) if n else None
    except Exception:
        return None


def _load_tokenizer(model: str, enable_thinking: bool):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    if not getattr(tok, "chat_template", None):
        print(f"[hf_judge] {model} ships no chat template; using ChatML", flush=True)
        tok.chat_template = _CHATML
    return tok


def _render(tok, messages: list[dict], enable_thinking: bool) -> str:
    """messages -> prompt string. Tolerates a template that predates the
    `enable_thinking` kwarg, which older Qwen2 templates do (they raise
    TypeError on an unexpected keyword rather than ignoring it)."""
    try:
        return tok.apply_chat_template(messages, tokenize=False,
                                       add_generation_prompt=True,
                                       enable_thinking=enable_thinking)
    except TypeError:
        return tok.apply_chat_template(messages, tokenize=False,
                                       add_generation_prompt=True)


class _Batcher:
    """Collect concurrent requests into one `generate` call.

    One worker thread owns the model. Callers append and wait on their own
    Event, so a caller never touches the engine and the engine is never entered
    twice. A request that raises carries the exception back to ITS OWN caller
    only; one bad prompt must not fail the batch it happened to land in.
    """

    def __init__(self, run: Callable[..., list[str]],
                 window_s: float, max_batch: int):
        self._run, self._window, self._max = run, window_s, max_batch
        self._q: list[dict] = []
        self._cv = threading.Condition()
        threading.Thread(target=self._loop, daemon=True).start()

    def submit(self, prompt: str, temperature: float,
               json_schema: dict | None = None) -> str:
        item = {"prompt": prompt, "temperature": temperature,
                "json_schema": json_schema,
                "done": threading.Event(), "text": None, "exc": None}
        with self._cv:
            self._q.append(item)
            self._cv.notify()
        item["done"].wait()
        if item["exc"] is not None:
            raise item["exc"]
        return item["text"]

    def _loop(self) -> None:
        while True:
            with self._cv:
                while not self._q:
                    self._cv.wait()
            # Outside the lock: let more requests accumulate for one window
            # before claiming the batch. This is the whole point of the class.
            time.sleep(self._window)
            with self._cv:
                batch, self._q = self._q[:self._max], self._q[self._max:]
            try:
                texts = self._run([b["prompt"] for b in batch],
                                  [b["temperature"] for b in batch],
                                  [b["json_schema"] for b in batch])
                for b, t in zip(batch, texts):
                    b["text"] = t
            except Exception as e:
                for b in batch:
                    b["exc"] = e
            finally:
                for b in batch:
                    b["done"].set()


def build_chat_generate(model: str, *, max_new_tokens: int = 2048,
                        max_input_tokens: int = 32768,
                        enable_thinking: bool = False, sampling: bool = True,
                        backend: str | None = None,
                        gpu_memory_utilization: float | None = None,
                        batch_window_s: float = 0.025,
                        max_batch: int = 32) -> Callable:
    """Return `gen(messages, temperature=0.0) -> str`.

    `sampling=False` pins greedy decoding regardless of what a caller asks for,
    which is what an offline metric wants: a judge that answers differently on a
    re-run makes a paired comparison unreproducible. The default leaves the
    per-request temperature in charge, and both faithfulness call sites pass 0.0.
    """
    backend = backend or ("vllm" if _vllm_usable() else "hf")
    tok = _load_tokenizer(model, enable_thinking)

    # THE INPUT CAP IS DERIVED FROM THE MODEL, not taken on trust from the
    # caller. `max_input_tokens` is a request-level policy, but the hard limit
    # is the model's own position count minus the generation budget, and the two
    # disagree the moment a short-context model is used: Phi-4 allows 16384, so
    # a caller asking for 24576 input would sail past `check()` and then fail
    # inside vLLM, which surfaces as `no_judge` (unknown) and looks exactly like
    # a coverage limit. Take the smaller of the two and say so at load.
    _limit = _position_limit(model)
    if _limit:
        room = _limit - max_new_tokens
        if room < max_input_tokens:
            print(f"[hf_judge] {model} allows {_limit} positions; input cap "
                  f"{max_input_tokens} -> {room} (leaving {max_new_tokens} to generate)",
                  flush=True)
            max_input_tokens = room
    if max_input_tokens <= 0:
        raise ValueError(
            f"{model} has no room for input: {_limit} positions against a "
            f"{max_new_tokens}-token generation budget. Lower --max-new-tokens.")

    def check(prompt: str) -> str:
        n = len(tok(prompt, add_special_tokens=False)["input_ids"])
        if n > max_input_tokens:
            # Never truncate: the blocks being judged are at the tail.
            raise ValueError(
                f"prompt is {n} tokens, over max_input_tokens={max_input_tokens}. "
                "Raise --max-input-tokens; truncating would silently drop the "
                "FL blocks and return a verdict about a prompt nobody sent.")
        return prompt

    if backend == "vllm":
        from vllm import LLM, SamplingParams
        # CLAMPED to the model's own position limit. vLLM refuses to start when
        # max_model_len exceeds it, and the two judges differ here:
        # FormalRx-8b allows 40960, Qwen2.5-7B-Instruct only 32768. Asking for
        # the same window from both would start one and hard-fail the other at
        # load time, inside a training job that has already staged Mathlib.
        # Consistent with `check()` above by construction: both are derived from
        # the same clamped input cap, so a prompt that passes the check always
        # fits the window.
        want = max_input_tokens + max_new_tokens
        llm = LLM(model=model, tokenizer=model, dtype="bfloat16",
                  max_model_len=want,
                  gpu_memory_utilization=(gpu_memory_utilization
                                          if gpu_memory_utilization is not None
                                          else float(os.environ.get(
                                              "JUDGE_GPU_MEM_UTIL", "0.85"))),
                  enforce_eager=os.environ.get("JUDGE_ENFORCE_EAGER", "0") == "1",
                  trust_remote_code=True)

        # CONSTRAINED DECODING, and it is the difference between this judge
        # being usable and not. MEASURED on the first validity-gate run: 8 of 12
        # replies were unparseable, every one of them because the model wrote
        # Lean and mathematical notation straight into JSON without quoting it
        # (`#iota < cof (c.ord)` bare inside an array, `"prime_number,`
        # unterminated, `C_0` as a bare identifier). That is not a token-budget
        # problem and no amount of after-the-fact repair fixes it honestly --
        # repairing malformed JSON means guessing what the judge meant. A schema
        # makes the failure mode unrepresentable instead.
        try:
            from vllm.sampling_params import StructuredOutputsParams
        except ImportError:            # older vllm: no structured outputs
            StructuredOutputsParams = None
            print("[hf_judge] WARNING: this vllm exposes no StructuredOutputsParams; "
                  "judge replies are unconstrained and may not parse", flush=True)

        def run(prompts: list[str], temps: list[float],
                schemas: list[dict | None]) -> list[str]:
            # One SamplingParams per prompt: vLLM accepts a list, which keeps a
            # batch from forcing every member onto one temperature or one schema.
            sps = []
            for t, sc in zip(temps, schemas):
                kw = {}
                if sc and StructuredOutputsParams is not None:
                    kw["structured_outputs"] = StructuredOutputsParams(json=sc)
                sps.append(SamplingParams(temperature=(t if sampling else 0.0),
                                          max_tokens=max_new_tokens, **kw))
            out = llm.generate(prompts, sps, use_tqdm=False)
            return [o.outputs[0].text for o in out]

        batcher = _Batcher(run, batch_window_s, max_batch)

        def gen(messages: list[dict], temperature: float = 0.0,
                json_schema: dict | None = None) -> str:
            return batcher.submit(check(_render(tok, messages, enable_thinking)),
                                  float(temperature), json_schema)
        return gen

    import torch
    from transformers import AutoModelForCausalLM
    mdl = AutoModelForCausalLM.from_pretrained(
        model, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    mdl.eval()
    lock = threading.Lock()

    warned = []

    def gen(messages: list[dict], temperature: float = 0.0,
            json_schema: dict | None = None) -> str:
        if json_schema and not warned:
            # Not silently ignored: without constrained decoding this backend
            # reproduces the malformed-JSON failure the vLLM path exists to
            # remove, and the caller would read that as low coverage.
            print("[hf_judge] WARNING: the transformers backend cannot enforce a "
                  "JSON schema; expect unparseable replies", flush=True)
            warned.append(1)
        prompt = check(_render(tok, messages, enable_thinking))
        ids = tok(prompt, return_tensors="pt").to(mdl.device)
        do_sample = sampling and float(temperature) > 0.0
        with lock, torch.no_grad():
            out = mdl.generate(**ids, max_new_tokens=max_new_tokens,
                               do_sample=do_sample,
                               temperature=float(temperature) if do_sample else None,
                               pad_token_id=tok.pad_token_id or tok.eos_token_id)
        return tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)

    return gen
