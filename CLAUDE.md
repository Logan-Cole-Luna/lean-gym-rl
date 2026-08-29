use https://mitcommlab.mit.edu/broad/commkit/coding-and-comment-style/ as a comment guideline

Do not write any — ever

# lean-gym-rl

A **gym for Lean 4 autoformalization RL**: swappable reward functions, a real
Lean/Mathlib scoring backend, and a paired-evaluation harness built so that
reward designs can be compared against each other *and against a calibrated
noise control*.

The flagship experiment is **BEq+** (deterministic, LLM-free bidirectional Lean
tactic equivalence, Poiroux et al. EMNLP'25) as a training-time RL reward — the
paper never got it into the reward, and that gap is what this started as. It is
now one arm among several.

**Proof-pair only.** The model emits a complete theorem AND its own proof, never
a bare signature. A model that only ever writes the statement and leaves `sorry`
isn't doing autoformalization, just NL→Lean-signature translation, so the
signature-only task variant (and every arm, dataset, and checkpoint trained on
it — the original Lean-Workbook line included) was removed. See "LoCoLib
proof-pair task" below for the mechanics.

Qwen2.5-Coder-3B-Instruct, LoCoLib, Lean 4.23.0 + Mathlib4, verl GRPO with
colocated FSDP actor + vLLM rollout.

## Read first

- `README.md` — what the repo is, setup, how to train. **Start here.**
- `arms.md` — every reward as a ladder, the results table, related work.
- `logook.md` — running lab notebook, newest entry first.
- `datasets.md` — the corpora, and the mid-training / OOD case for the second one.
- `hpc/NARVAL_NOTES.md` — cluster gotchas. **Read before running anything on Narval.**

`results/FINDINGS.md` is 0.5B-era and partly superseded: its claim that drift
leaves type-check intact does NOT hold at 3B. Treat it as provenance.

## Environment

Never run bare `python`. Always:

```bash
source hpc/cc_env.sh     # module stack + venv on $SCRATCH + HF/Lean cache paths
```

`pyarrow` comes from the `arrow/23.0.1` **module**, not pip — any check run
without sourcing `cc_env.sh` will wrongly report it missing. The venv lives at
`$SCRATCH/ai4math_training_venv_hpc` and must stay on the **torch 2.11** set
(see `hpc/NARVAL_NOTES.md`; do not "fix" a mismatch by downgrading vllm).

`import vllm` succeeds while vllm is broken — `_C.abi3.so` loads lazily. Test
`import vllm._C`. SFT training never touches vllm and passes regardless.

Anything that touches Lean must run on a compute node with Mathlib staged to
`$SLURM_TMPDIR` from the prebuilt tar (41s vs 1984s; without it `import Mathlib`
exceeds the Lean timeout on compute nodes as well as login nodes).

## Layout

```
hpc/job_prelude.sh       SHARED slurm prelude: env, unset ROCR, PYTHONPATH, and
                         stage_mathlib(). Source it first in any new job -- the
                         preamble used to be copy-pasted into 12 files.
reward/beq_plus.py       BEq+ behind a persistent Lean REPL (~4.3GB Mathlib resident)
reward/reward_fn.py      the reward zoo; compute_score = compute_score_outcome
reward/lean_tool.py      verl function_tool giving the policy Lean's real diagnostics
configs/run_grpo.sh      GRPO entrypoint. Read its header comments before tuning --
                         each constant records the failure that produced it.
scripts/                 dataset prep, rollouts, offline scoring, eval, stats
hpc/*.slurm              SLURM jobs, all Narval fixes baked in
```

**One checkout.** `hpc/grpo_eval.slurm` used to run from a second tree at
`/scratch/logan03/lean-gym-rl-rl_reward`, so eval jobs imported code last touched
2026-08-20 while training used `/home`. Audited (the scratch tree was a strict
subset; the scoring entry points were identical, so no cached number is affected)
and consolidated. If you see that path anywhere, it is stale.

`checkpoints/` and `repos/` are **symlinks into `$SCRATCH`**; model weights go
there too via the Makefile's `MODELS_ROOT` (no symlink). `/home` is 50GB and
cannot hold any of it — a truncated optimizer write killed a run at 46/50GB.

## The reward zoo

| reward | signal | role |
|---|---|---|
| `compute_score_outcome` | six-outcome ladder (`reward/reward.py`) | **default** (`compute_score`); graded below BEq+ too |
| `compute_score_gated` | BEq+ ladder (0 / one direction / both) | the semantic arm; flat floor below BEq+ |
| `compute_score_typecheck_only` | elaborates | loop-health probe, **known exploitable** |
| `compute_score_guided` | gated + type-check-gated similarity | **REMOVED** — regressed BEq+ 38.8%→29.0% over 200 steps; `reward/similarity.py` deleted with it |
| `compute_score_placebo` | deterministic pseudo-random on rollout hash | **RETIRED** — the control, question answered |

**LoCoLib proof-pair task.** Every arm trains on `data_locolib/*_proof.parquet`
(built by `scripts/data/prepare_locolib.py --emit-proof`): the model emits a
complete theorem AND its own proof, never a bare signature. `reward/beq_plus.py`'s
`BEqPlusScorer.check_own_proof()` elaborates that submission AS WRITTEN (never
force-injecting `sorry` the way BEq+'s equivalence cascade does internally) and
additionally checks `#print axioms` against Lean's standard trio, so a candidate
cannot farm `proved` with a smuggled axiom. This is what makes `reward/reward.py`'s
`proved`/`SOLVED` rows reachable — see that file's table. `typecheck_ex`
(force-`sorry`, signature-only elaboration) still exists inside `BEqPlusScorer`
because BEq+'s own equivalence search uses it internally on both sides of the
comparison — that is unrelated to the deleted signature-only *task*.

LoCoLib targets Lean 4.23.0, not the 4.8.0-rc1 this repo used to run
Lean-Workbook against — pass `MATHLIB_TAR=.../mathlib4_v4.23_lake.tar`,
`MATHLIB_TAR_FLAT=1`, and a matching `LEAN_INTERACT_CACHE_DIR` (see "Running
things" below). `hpc/job_prelude.sh`'s `MATHLIB_TAR_FLAT` doc explains why the
flag exists — the two tars are packed differently.

**The placebo is retired. Do not run new placebo arms.** It carried no
information, invoked no Lean, and its constants were fitted per policy to match
the measured informative-group rate, within-group sd, and mean reward. The
fitting script, its constants, and the `rl3b_v2_placebo` checkpoints/data it
produced were removed along with the rest of the signature-only Lean-Workbook
line (see "Proof-pair only" above) — the **+18.2pp, p=7e-16** paired
gated-vs-placebo result stands as recorded provenance even though the run that
produced it no longer exists on disk.

What replaces it as the comparator: **a new arm is paired against SFT and
against its own twin at the previous setting**, which is a tighter control
anyway — one variable changes instead of the whole reward. The `SERIES_TAG`
discipline below still applies; a new series just no longer needs a matched
placebo fit.

The placebo's most useful legacy is a *negative* result about the optimiser: a
zero-information reward should do approximately nothing under correct
regularisation, and instead it took BEq+ 39.4% → 13.0%. That is what points at
the learning rate (see below), not at any reward.

## Conventions that matter

- **Results layout.** `results/eval/<model_label>/` holds every eval/gen/passk
  artifact for one checkpoint series (`eval_<label>-step<N>_n<n>.json`, matching
  `gen_*.jsonl`, `passk_*.json`), `<model_label>` being the label with any
  `-step<N>` suffix stripped (e.g. `rl3b_locolib_proof_gated`,
  `sft3blocolib_proof`). `results/train/`
  holds training-time artifacts (`train_metrics/`, `val_generations_*/`),
  `results/figures/` holds generated figures, and `results/_archive/pre-reorg/`
  holds everything from before this layout existed (untouched, for provenance —
  0.5B-era docs and probes that don't map to a single model). Every writer
  (`hpc/grpo_eval.slurm`, `hpc/eval_sft.slurm`, `hpc/passk.slurm`, `Makefile`'s
  `eval-ckpt`/`eval-sweep`) and reader (`scripts/eval/evalio.py`,
  `compare_arms.py`, `select_checkpoint.py`, `scripts/figures/*.py`) already
  target this layout — do not reintroduce a flat `results/eval_*.json` write.
- **Eval naming**: `results/eval/<model_label>/eval_<label>-step<N>_n<n>.json`,
  with a `per_example` array. `scripts/eval/compare_arms.py` pairs on the
  filename; `select_checkpoint.py` merges records across files **by the label
  inside the JSON**, which `evaluate_checkpoints.py` sets from the *directory
  name*. Point it at a uniquely-named dir — aiming straight at
  `<ckpt>/huggingface` labels everything `"huggingface"` and silently collapses
  all records into one.
- **Always paired.** Same pinned validation slice, McNemar exact. Never
  difference two headline rates — a subset can be easier than the whole.
  `data_locolib/val_proof.parquet` (760 rows) is the pinned proof-pair slice;
  `hpc/grpo_eval.slurm` defaults `N_EVAL` to 1000, so pass the corpus's actual
  row count explicitly rather than relying on that default silently truncating.
- **Report retention vs gains**, not just net. Arms differ mostly in how much
  they *destroy*. The **gain-rate** (converted / previously-wrong) discriminates
  better than the headline: gold-referenced 7.7–10.6%, gold-free 7.2–7.7%,
  exploitable proxy decaying to 2.1%, pure noise 1.7–4.3%.
- **Series scoping.** `TAG` (SFT/eval), `BEST_SFT` (which checkpoint pointer),
  `SERIES_TAG` (GRPO experiment name + placebo constants). Defaults reproduce the
  original paths byte-for-byte; a new series must never share a checkpoint dir,
  eval label, best-SFT pointer, or placebo fit with another.

## Traps that have actually cost us

- **A reward function that raises loses the JOB, not the rollout.** verl marks
  the agent-loop task failed and the GRPO step never closes, so the run sits on a
  GPU until walltime. Tell: `running: N, finished: M, failure: K` static in the
  log for many minutes. Cost ~10 GPU-hours on `rl3b_selfprove`. Cause was a
  recycled Lean REPL raising `AttributeError: 'NoneType' object has no attribute
  'stdin'` past `typecheck_ex`'s narrow `except`. **A dead REPL is normal here** —
  `max_process_memory=0.8` recycles it on purpose. All Lean traffic now goes
  through `BEqPlusScorer._run()`, which recovers via `_restart()`; `_restart`
  clears `_env_cache` **unconditionally**, because cached env ids belong to the
  dead process and keeping them makes a worker score everything 0.0 forever. All
  seven `compute_score_*` entry points carry `@_never_raises`.
- **`multiprocessing.Pool` hangs forever if a worker dies holding a task.** Tell:
  `worker N ready` at the tail and the output file exactly one row short. Cost
  ~9 GPU-hours and cancelled every `afterok` behind it. `score_rollouts.py` now
  drives the iterator with `--result-timeout`.
- **`sbatch` snapshots the script at submit time.** Editing a `.slurm` does not
  change already-queued jobs — which is how the current arm set stayed uniformly
  single-turn after `MULTITURN` defaulted to on.
- **`MULTITURN=1` was a no-op, and now defaults to 0.** Setting
  `multi_turn.enable=True` + `function_tool_path` does NOT route rollouts through
  the tool loop. verl picks the agent loop from a per-row `agent_name`, falling
  back to `rollout.default_agent_loop = "single_turn_agent"`, and our parquet
  rows have no `agent_name` — so `SingleTurnAgentLoop` applies the chat template
  with no `tools=` argument and generates once. `reward/lean_tool.py` never ran
  in any arm. The tool schema *did* reach `RLHFDataset`'s prompt-length filter,
  so leaving it on perturbed row filtering while buying nothing. To really enable
  it: set `rollout.agent.default_agent_loop=tool_agent` (or add `agent_name` in
  the dataset-prep script) **and** raise `MAX_RESPONSE_LENGTH` well past 128 —
  128 tokens is the total across turns and cannot fit a call plus a response
  plus a revised theorem.
- **`max_response_length=128` is set in exactly one place** (`run_grpo.sh`) and
  no `.slurm` overrides it. Measured `response_length/clip_ratio` is 2.3–4.4% per
  step (max 13.3%): a truncated rollout scores 0 under every reward, so this is a
  small standing gradient against long — i.e. semantically richer — statements.
- **Quoted heredocs (`<<'PYX'`) do not expand `${VAR}`** and `bash -n` passes
  anyway. Pass values as argv.
- **verl's SFT loss mask only covers `<|im_start|>system\n`.** A lone assistant
  message trains on Qwen's injected default system prompt (~40% of tokens).
  Include an explicit system message.
- **Two divergent checkouts.** Fixed, but the class of bug is worth remembering:
  a job that `cd`s somewhere else and sets `PYTHONPATH` there will silently run
  different code from the one you are editing.
- **`compare_arms.py` used to require an exact `_n<N>` filename match**, so arms
  evaluated at different `N_EVAL` reported "no common steps" despite being
  perfectly comparable. It now matches any `n` and pairs on the common prefix.
- **`SAVE_FREQ` is in optimizer steps, not prompts.** At batch 256 the batch-16
  value of 10 means the first checkpoint is 20–30h away; `rl3b_bb_*` burned ~28h
  and produced none.

## Running things

```bash
# ARM=gated|typecheck|outcome|gated_edge. LoCoLib needs the 4.23 Mathlib tar --
# defaults in job_prelude.sh point at the wrong (4.8.0-rc1) one otherwise.
sbatch --export=ALL,ARM=gated,SERIES_TAG=locolib_proof,\
TRAIN_FILE=data_locolib/rl_proof.parquet,VAL_FILE=data_locolib/val_proof.parquet,\
BEST_SFT=data_locolib/best_sft_proof.txt,\
MATHLIB_TAR=/scratch/logan03/mathlib4_v4.23_lake.tar,MATHLIB_TAR_FLAT=1,\
LEAN_INTERACT_CACHE_DIR=/scratch/logan03/ai4math_training_lean_interact_cache_v423 \
hpc/grpo.slurm

sbatch --export=ALL,RUN=rl3b_locolib_proof_gated,VAL_PARQUET=data_locolib/val_proof.parquet,N_EVAL=760 hpc/grpo_eval.slurm
sbatch --export=ALL,LABEL=sft3blocolib_proof-step76 hpc/passk.slurm
python scripts/eval/compare_arms.py rl3b_locolib_proof_typecheck rl3b_locolib_proof_outcome rl3b_locolib_proof_gated
python scripts/eval/select_checkpoint.py --baseline sft3blocolib_proof-step76
```

`hpc/submit.sh` wraps `sbatch` and names the job from these same knobs at submit
time rather than at job start, so the name is visible in `squeue` while pending
(`grpo_3b-proof_gated_locolib`, not a placeholder). Prefer it over calling
`sbatch` directly.

Every SLURM job must `unset ROCR_VISIBLE_DEVICES` (Narval sets it alongside
`CUDA_VISIBLE_DEVICES`; verl's worker init hard-errors) and stage Mathlib.

Save checkpoints with `save_contents=[model,extra,hf_model]` (add `optimizer` for
anything that must resume) — `hf_model` writes ready-to-load HF weights, removing
the `verl.model_merger` step.

## Throughput

- BEq+ eval: `--workers 8` → ~1.0–1.7s/example, ~20 min per 400-example checkpoint
- GRPO: `AGENT_LOOP_WORKERS` is the Lean-parallelism knob. `reward.num_workers`
  is **not** — tuning it has no effect on the number of Lean servers.
- **Lean scoring is ~97% of a gated step, and ~1% of a type-check step.** verl
  logs no `timing_s/reward` — the reward runs inside the agent loop, so its cost
  lands in `timing_s/gen`. Measured over 150 logged steps each: placebo `gen`
  4.4s / step 26.1s; typecheck `gen` 6.4s / step 25.7s. Every non-gen term is
  identical. So **the whole type-check bill is ~2 s/step** (~0.36s per rollout at
  24 agent-loop workers, one elaboration against a cached header env), which is
  why `typecheck` and `placebo` have the same step time despite one of them never
  opening Lean. `gated` is ~962 s/step, i.e. ~935 s of Lean = **~175 Lean-seconds
  per rollout** — with `BEQ_TIMEOUT_PER_PROOF=30` and up to 18 cascade calls,
  that is ~6 timed-out proof attempts on each rollout that is not equivalent to
  the gold. The ratio is not a constant factor on "a Lean call"; it is one fast
  elaboration versus six half-minute searches.

## Where things stand

Everything below is the **proof-pair** track (`SERIES_TAG=locolib_proof`), the
only task variant left after the signature-only removal. `sft3blocolib_proof`
(from `checkpoints/sft_3b_locolib_proof`, step 76 pinned as `best_sft_proof.txt`)
is the SFT baseline all three RL arms resume from.

`rl3b_locolib_proof_typecheck` (the exploitable ablation baseline, run at the
old 1e-5 default before it was fixed) ran its full 90 steps and was evaluated
at n=760: BEq+ and type-check both **fell** versus the SFT baseline
(43.9%→34.9% BEq+, 50.5%→43.7% type-check), and `proved` stayed at 0.0%
throughout — expected, since the typecheck reward never requires the proof to
actually close. That run also carried an unrelated reward-side bug (below),
so it never received real gradient from `reward.custom_reward_function` at
all until the fix landed partway through this series -- kept on disk anyway as
the "previous LR / previous reward-bug" reference point, not as a clean result.

**THE LEARNING RATE WAS 1e-5 AND WAS NEVER INTENDED; `configs/run_grpo.sh` NOW
DEFAULTS TO 1e-6**, matching verl's own default, TRL's `GRPOConfig`, and
DeepSeek's GRPO. This was originally diagnosed on the (now-deleted)
signature-only Lean-Workbook line, then found to still apply unfixed on the
proof-pair track (`actor/kl_loss` climbing to 0.27 against a ~0.05 target,
matching the old line's exact signature). Combined with `train_batch_size=16`
(~32x smaller than GRPO is normally run at) and `ppo_epochs=1`, each Adam
update at the old LR was a full-size parameter step driven by a handful of
prompts -- the same mechanism that made the (deleted) placebo control
catastrophic rather than inert on the old line. `rl3b_locolib_proof_typecheck`
above is kept as the explicit "previous LR" pairing target; every arm now
running (`SERIES_TAG=locolib_proof_lr6`: `typecheck`, `gated`, `outcome`, all
at `ACTOR_LR=1e-6,LR_WARMUP_RATIO=0.05`) is the corrected-LR series and is what
future comparisons should use.

**A separate, now-fixed bug**: `compute_score_gated` and `compute_score_outcome`
in `reward/reward_fn.py` built their return dict as
`{"score": <real value>, **_diagnostics(...)}`; `_diagnostics()` used to spread
in `**_SCHEMA`, which itself had a `"score": 0.0` entry, silently overwriting
the real computed reward back to 0.0 on every single call. `compute_score`
returned `0.0` for everything and GRPO's advantage was zero every step --
verified via a compute-node diagnostic (real gold rows and real model
completions through the unwrapped function) before and after the fix.
`compute_score_typecheck_only` was never affected: it never called
`_diagnostics()`. Fixed by dropping `"score"` from `_SCHEMA` and giving
`gated`/`typecheck` their own small hand-built return dicts (matching shape,
so a mid-run scorer failure can't hand verl a different key set than the
happy path). The first `rl3b_locolib_proof_outcome` attempt (50 steps, zero
gradient throughout) was discarded and restarted clean rather than resumed,
to keep step-N comparable across arms.

Settled negatives worth not repeating: type-check-only RL collapsed to 8.0%
BEq+ at 98% type-check on the old signature-only line; mid-training on Mathlib
statements did nothing there, because only ~2.3% of Mathlib declarations are
self-contained enough to elaborate standalone (LoCoLib validates far higher,
~67.5%, which is part of why it replaced Lean-Workbook rather than just adding
proof-pair on top of it).
