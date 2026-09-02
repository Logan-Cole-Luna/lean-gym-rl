#!/usr/bin/env python3
"""Score trained checkpoints (and the untrained base model) on the SAME metrics,
so the two ablation arms are actually comparable.

Why this exists: verl's `val-core/<data_source>/acc/mean@1` is just the mean of
whatever `custom_reward_function` that run used. The composite arm's validation
number is therefore `0.1*typecheck + 0.9*beq_plus`, while the typecheck-only
arm's is the raw type-check rate -- two different scales. Comparing them
directly would be meaningless, and the composite number alone cannot be
decomposed (a score of 0.0825 is consistent with anything from
"82.5% type-check, 0% BEq+" to "8.25% both").

This script reports BOTH metrics separately for every checkpoint, which is the
comparison the PoC is actually after:

    checkpoint                 typecheck%   beq_plus%
    base (untrained)              ...         ...
    typecheck_only @ step 30      ...         ...
    composite      @ step 30      ...         ...

Usage:
    python scripts/eval/evaluate_checkpoints.py \
        --checkpoint base \
        --checkpoint checkpoints/beqplus_rl_poc/qwen25_coder_0_5b_compute_score_composite/global_step_30 \
        --n-eval 80
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _paths  # noqa: F401  -- repo root + every stage folder

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASE_MODEL = PROJECT_ROOT / "models" / "qwen2.5-coder-0.5b-instruct"


def resolve_model_path(spec: str) -> tuple[str, str]:
    """Return (label, hf_model_dir) for a checkpoint spec.

    Accepts:
      'base'                      -> the untrained base model
      a merged HF dir             -> used as-is (what `make merge-checkpoints` produces)
      a raw verl checkpoint dir   -> falls back to its actor/huggingface/ subdir
    """
    if spec == "base":
        return "base (untrained)", str(DEFAULT_BASE_MODEL)

    path = Path(spec)
    if not path.exists():
        raise SystemExit(f"No such checkpoint dir: {path}")

    # A merged/plain HF directory already has weights next to its config.
    if has_full_weights(path):
        return path.name, str(path)

    # Otherwise assume verl's checkpoint layout.
    hf_dir = path / "actor" / "huggingface"
    if hf_dir.exists():
        return f"{path.parent.name}@{path.name}", str(hf_dir)

    raise SystemExit(
        f"{path} has neither HF weights nor actor/huggingface/ -- "
        f"run `make merge-checkpoints` first."
    )


def has_full_weights(hf_dir: Path) -> bool:
    """verl saves config/tokenizer into actor/huggingface/ but may keep the
    weights themselves in the sharded model_world_size_*.pt files alongside."""
    return any(hf_dir.glob("*.safetensors")) or any(hf_dir.glob("pytorch_model*.bin"))


def generate(model_dir: str, prompts: list[str], max_new_tokens: int, gpu_frac: float) -> list[str]:
    """Generate completions, applying the model's chat template.

    This MUST match how the prompt is formatted during training, or the model is
    evaluated off-distribution. Both verl's RL rollouts and the SFT trainer apply
    the chat template, so evaluating on raw prompt text silently degrades every
    checkpoint. Observed concretely on the SFT model: with the raw prompt it
    emitted `lean_workbook_plus_11512 : Nat.choose 13 2 = 78` -- no `theorem`
    keyword, so nothing parses and the score is 0/80; with the template applied
    the same model emitted the correct `theorem lean_workbook_plus_11512 : ...`.
    """
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    if tok.chat_template:
        prompts = [
            tok.apply_chat_template(
                [{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True
            )
            for p in prompts
        ]

    # max_model_len was hardcoded to 896 (= the 16GB-era max_prompt_length 768 +
    # max_response_length 128). vLLM REJECTS any prompt over the limit rather
    # than truncating, so a single long example aborts the whole evaluation --
    # which is what happened the first time a 1,000-row validation set was used
    # (one prompt at 1,470 tokens out of 1,000). Derive it from the data, with
    # the old value as a floor so existing runs are unchanged.
    longest = max((len(tok(p)["input_ids"]) for p in prompts), default=0)
    max_model_len = max(896, longest + max_new_tokens + 8)
    if max_model_len > 896:
        print(f"[eval] max_model_len {max_model_len} (longest prompt {longest} tokens); "
              f"note training drops prompts over 768, so anything above that is "
              f"off-distribution for every checkpoint alike", flush=True)

    llm = LLM(
        model=model_dir,
        trust_remote_code=True,
        dtype="bfloat16",
        gpu_memory_utilization=gpu_frac,
        max_model_len=max_model_len,
        enforce_eager=True,
    )
    # Greedy: we're measuring the policy's mode, not sampling diversity.
    out = llm.generate(prompts, SamplingParams(max_tokens=max_new_tokens, temperature=0.0))
    return [o.outputs[0].text for o in out]


_W_SCORER = None


def _init_scoring_worker(probe_stronger: bool, standalone: bool = False) -> None:
    global _W_SCORER
    from reward.beq_plus import BEqPlusScorer
    _W_SCORER = (BEqPlusScorer(), probe_stronger, standalone)


def _score_one(task):
    """(index, completion, gold) -> per-example record. Order is restored by
    imap, so per_example stays index-aligned with the validation slice -- the
    paired McNemar tests depend on that."""
    i, comp, gold = task
    from reward.reward_fn import _clean_solution
    scorer, probe_stronger, standalone = _W_SCORER
    pred = _clean_solution(comp)
    r = (scorer.score_standalone(gold, pred, probe_stronger=probe_stronger) if standalone
         else scorer.score(gold, pred, probe_stronger=probe_stronger))
    return i, pred, {
        "i": i,
        "typecheck": bool(r["typecheck"]),
        "beq_plus": bool(r["beq_plus"]),
        "gold_implies_pred": bool(r.get("gold_implies_pred", False)),
        "pred_implies_gold": bool(r.get("pred_implies_gold", False)),
        "semantic_signal": int(r.get("semantic_signal", 0)),
        "error": r["error"],
        "error_kind": r.get("error_kind"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", action="append", dest="checkpoints", default=[],
                    help="'base' or a verl checkpoint dir (repeatable)")
    ap.add_argument("--val-parquet", default=str(PROJECT_ROOT / "data" / "val.parquet"))
    ap.add_argument("--n-eval", type=int, default=80, help="number of val examples to score")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--gpu-frac", type=float, default=0.35)
    # BEq+ scoring is the wall-clock cost of an eval (the cascade makes up to
    # ~18 Lean calls per example). Each worker builds its OWN BEqPlusScorer /
    # Lean REPL -- no shared state, identical scoring semantics, so results
    # stay comparable with serially-scored evals. Budget ~6GB resident each.
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--out", default=str(PROJECT_ROOT / "results" / "ablation_comparison.json"))
    # Generations were previously produced, scored, and thrown away, which left
    # every "why did this checkpoint get worse?" question unanswerable without a
    # full re-run. They are small; keep them.
    ap.add_argument("--no-save-generations", dest="save_generations", action="store_false",
                    help="skip writing gen_<label>_n<N>.jsonl next to --out")
    ap.add_argument("--probe-stronger", action="store_true",
                    help="also test pred=>gold on its own when gold=>pred fails "
                         "(see reward/beq_plus.py DIRECTION SEMANTICS); slower")
    ap.add_argument("--standalone", action="store_true",
                    help="the model emits a self-contained snippet (own import/open/"
                         "namespace), not a bare theorem for the gold's context -- "
                         "e.g. a CoT-distilled model. Scores via "
                         "BEqPlusScorer.score_standalone (union-context env).")
    args = ap.parse_args()

    if not args.checkpoints:
        raise SystemExit("Pass at least one --checkpoint (use 'base' for the untrained model)")

    import pandas as pd

    _val = pd.read_parquet(args.val_parquet)
    if args.n_eval > len(_val):
        raise SystemExit(
            f"--n-eval {args.n_eval} exceeds {args.val_parquet} ({len(_val)} rows). "
            f"Pass --val-parquet matching the corpus's row count, or lower --n-eval.")
    df = _val.head(args.n_eval)
    # Take the LAST user turn, not element 0.
    def _user_turn(row):
        users = [t for t in row if t.get("role") == "user"]
        if not users:
            raise ValueError(f"prompt has no user turn: {[t.get('role') for t in row]}")
        return users[-1]["content"]

    prompts = [_user_turn(row) for row in df["prompt"]]
    golds = [rm["ground_truth"] for rm in df["reward_model"]]
    print(f"[eval] {len(prompts)} validation examples from {args.val_parquet}")

    # Import the scorer AFTER vllm work, so the Lean REPL isn't held open during
    # generation. (It's CPU-only, but Mathlib's env is memory-hungry.)
    results = {}
    for spec in args.checkpoints:
        label, model_dir = resolve_model_path(spec)
        hf_dir = Path(model_dir)
        if spec != "base" and not has_full_weights(hf_dir):
            print(f"[eval] SKIP {label}: {hf_dir} has no .safetensors/.bin weights "
                  f"(verl kept them sharded; run verl's model merger first, e.g. "
                  f"`python -m verl.model_merger merge --backend fsdp "
                  f"--local_dir {hf_dir.parent} --target_dir <out>`)")
            continue

        print(f"\n[eval] ===== {label} =====")
        print(f"[eval] generating from {model_dir} ...")
        completions = generate(model_dir, prompts, args.max_new_tokens, args.gpu_frac)

        from reward.reward_fn import _clean_solution

        n_tc = n_beq = 0
        per_example = []
        generations = []

        if args.workers > 1:
            import multiprocessing as mp
            ctx = mp.get_context("spawn")
            tasks = [(i, c, g) for i, (c, g) in enumerate(zip(completions, golds))]
            pool = ctx.Pool(args.workers, initializer=_init_scoring_worker,
                            initargs=(args.probe_stronger, args.standalone))
            scored_iter = pool.imap(_score_one, tasks, chunksize=1)
        else:
            from reward.beq_plus import BEqPlusScorer
            scorer = BEqPlusScorer()
            pool = None

            def _serial():
                for i, (comp, gold) in enumerate(zip(completions, golds)):
                    pred = _clean_solution(comp)
                    r = (scorer.score_standalone(gold, pred, probe_stronger=args.probe_stronger)
                         if args.standalone
                         else scorer.score(gold, pred, probe_stronger=args.probe_stronger))
                    yield i, pred, {
                        "i": i,
                        "typecheck": bool(r["typecheck"]),
                        "beq_plus": bool(r["beq_plus"]),
                        "gold_implies_pred": bool(r.get("gold_implies_pred", False)),
                        "pred_implies_gold": bool(r.get("pred_implies_gold", False)),
                        "semantic_signal": int(r.get("semantic_signal", 0)),
                        "error": r["error"],
                        "error_kind": r.get("error_kind"),
                    }
            scored_iter = _serial()

        import time as _time
        _t0 = _time.time()
        for i, pred, rec in scored_iter:
            comp = completions[i]
            n_tc += rec["typecheck"]
            n_beq += rec["beq_plus"]
            # Per-direction flags are recorded because the aggregate rates cannot
            # distinguish "drifted toward weaker statements" from "drifted toward
            # unrelated ones"
            per_example.append(rec)
            if args.save_generations:
                generations.append({"i": i, "prompt": prompts[i], "gold": golds[i],
                                    "completion": comp, "pred": pred, **rec})
            done = len(per_example)
            if done % 10 == 0:
                rate = (_time.time() - _t0) / done
                print(f"  [{done}/{len(completions)}] typecheck={n_tc} beq+={n_beq} "
                      f"({rate:.1f}s/example, eta {rate*(len(completions)-done)/60:.0f}m)",
                      flush=True)
        if pool is not None:
            pool.close(); pool.join()
        # imap yields in task order, but sort defensively: per_example[i] MUST be
        # validation example i for the paired tests to mean anything.
        per_example.sort(key=lambda e: e["i"])
        generations.sort(key=lambda e: e["i"])

        n = len(completions)
        n_err = sum(1 for e in per_example if e["error_kind"])
        results[label] = {
            "model_dir": model_dir,
            "n": n,
            "typecheck_rate": n_tc / n if n else 0.0,
            "beq_plus_rate": n_beq / n if n else 0.0,
            "weaker_only_rate": sum(
                1 for e in per_example if e["gold_implies_pred"] and not e["beq_plus"]
            ) / n if n else 0.0,
            "scorer_error_rate": n_err / n if n else 0.0,
            "per_example": per_example,
        }
        print(f"[eval] {label}: typecheck {n_tc}/{n} ({100*n_tc/n:.1f}%)  "
              f"beq_plus {n_beq}/{n} ({100*n_beq/n:.1f}%)")
        if n_err:
            # These are scored as failures but are not verdicts about the model.
            print(f"[eval] WARNING: {n_err}/{n} examples hit a Lean scorer failure "
                  f"(timeout or dead REPL) and are counted as wrong.")

        if args.save_generations:
            gen_path = Path(args.out).parent / f"gen_{label}_n{n}.jsonl"
            gen_path.parent.mkdir(parents=True, exist_ok=True)
            with gen_path.open("w") as fh:
                for rec in generations:
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(f"[eval] generations -> {gen_path}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))

    print("\n=============== COMPARISON ===============")
    print(f"{'checkpoint':<45} {'typecheck%':>11} {'beq_plus%':>10}")
    for label, r in results.items():
        print(f"{label:<45} {100*r['typecheck_rate']:>10.1f}% {100*r['beq_plus_rate']:>9.1f}%")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
