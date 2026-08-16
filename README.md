# ai4math_training — BEq+ as an RL Reward for Lean 4 Autoformalization

A proof-of-concept measuring the actual training-time impact of using **BEq+**
(semantic/symbolic statement-equivalence) as an RL reward signal for autoformalization,
per *"Online Reinforcement Learning for Autoformalization"*
([researchgate.net/publication/396825045](https://www.researchgate.net/publication/396825045)).
That paper's core claim: a type-check-only reward is exploitable — the policy learns to
produce syntactically valid but semantically vacuous Lean statements. Composing it with
a BEq+ semantic-equivalence reward against ground truth is meant to close that gap. This
project builds the smallest real pipeline that can measure it: GRPO training with two
swappable reward functions (type-check-only vs. type-check + BEq+), on real data, with a
real Lean toolchain.

## What's here

| Piece | Choice | Why |
|---|---|---|
| RL framework | [verl](https://github.com/volcengine/verl) (`repos/verl`) | Widely-used open RL post-training library for LLMs; native GRPO; pluggable `custom_reward_function`. |
| Algorithm | GRPO, full-parameter (no LoRA) | See *Hardware notes* below for why LoRA was dropped. |
| Base model | `Qwen/Qwen2.5-Coder-0.5B-Instruct` | Small enough to actually train on a single 16GB GPU; code-pretrained, so not starting from zero on Lean syntax. |
| Dataset | [`internlm/Lean-Workbook`](https://huggingface.co/datasets/internlm/Lean-Workbook) | 57k well-known, widely-cited NL↔Lean4 pairs (NeurIPS'24), built specifically for autoformalization training. |
| Lean / Mathlib | Mathlib4 @ `v4.8.0-rc1` (`repos/mathlib4`) | Pinned to match Lean-Workbook's target toolchain — built fresh for this project, not shared with other Lean projects on this machine. |
| Semantic reward | **BEq+** (`reward/beq_plus.py`) | Deterministic, CPU-only, LLM-free symbolic equivalence metric (Poiroux et al., EMNLP'25). Algorithm vendored from `augustepoiroux/RLMEval`; wrapped in a persistent Lean REPL server for online per-rollout scoring. |

## Setup

Requires [`uv`](https://astral.sh/uv), a CUDA-capable GPU, and network access for the
one-time downloads.

```bash
make setup          # env + verl + Mathlib4 + model + dataset (~20-30 min, mostly Mathlib)
source .venv/bin/activate
make smoke           # ~1-2 min sanity check: one GRPO step on 8 examples
```

`make help` lists every target. Each setup step is also independently re-runnable
(`make env`, `make verl`, `make mathlib`, `make model`, `make dataset`) if you only need
to redo one piece.

## Running training

```bash
make train-composite    # GRPO with type-check + BEq+ composite reward
make train-typecheck    # GRPO with type-check-only reward (the exploitable baseline)
make train               # both, sequentially — this is a single-GPU PoC, they can't overlap

make train-shaped        # graded/process-level BEq+ reward, from scratch
make train-curriculum    # two-phase curriculum from scratch (see below)
```

### Reward functions

Selected per run via `REWARD_FN_NAME`; all live in `reward/reward_fn.py`.

| name | definition | purpose |
|---|---|---|
| `compute_score_typecheck_only` | `1.0` iff the statement elaborates | the exploitable baseline |
| `compute_score_composite` | `0.1·typecheck + 0.9·BEq+` | the paper's composite formulation |
| `compute_score_shaped` | `0.15 typecheck + 0.35 one-direction + 0.50 both` | graded — but see the measurement below |
| `compute_score_guided` | `0.20·similarity + 0.15 typecheck + 0.20 one-dir + 0.45 both` | **continuous**; grades the buckets the others collapse |

### Why the shaped reward wasn't enough (measured)

Rung occupancy of `compute_score_shaped` over 40 validation outputs of the SFT
policy (`results/rung_probe.json`):

| rung | share |
|---|---|
| 0.00 — does not type-check | 27.5% |
| 0.15 — type-checks, no BEq+ | 32.5% |
| **0.50 — one BEq+ direction** | **2.5%** |
| 1.00 — full BEq+ | 37.5% |

The intermediate rung fires on **1 example in 40**, because "one direction
proves" still requires a successful Lean tactic search — nearly as rare as the
full proof. So the shaped reward is *cosmetically* denser but functionally close
to binary: 60% of the data sits in two undifferentiated buckets where a
one-coefficient near-miss scores exactly the same as `theorem x : 1 = 1`.

### The guided reward

`compute_score_guided` adds a **continuous** term from `reward/similarity.py`,
which is defined even when the statement fails to elaborate, so it grades both
dead buckets:

- **GTED-style structural similarity** — statements are converted to operator
  trees and compared by normalized tree edit distance. Inspired by
  [GTED (arXiv:2507.07399)](https://arxiv.org/abs/2507.07399), which reports the
  highest accuracy/Kappa on miniF2F and is explicitly complementary to BEq (BEq
  tests logical equivalence; GTED measures structural distance). **Ours is an
  approximation**: we build trees from bracket/operator structure rather than
  from a real Lean-LSP parse, to avoid a second Lean round-trip per rollout.
- **Library-constant overlap** — Jaccard over dotted/capitalised constants
  (`Set.Ioo`, `Nat.choose`). Bare lowercase identifiers are excluded on purpose:
  counting them gave an unrelated prediction 0.667 credit just for also
  mentioning `x`.
- **Embedding similarity** — the continuous, pairing-free reward proposed in the
  source paper itself. Optional (`BEQ_USE_EMBEDDING=1`), CPU-only, off by
  default given this project's host-RAM history.

Effect on the cases that matter:

| case | composite | shaped | **guided** |
|---|---|---|---|
| exact match | 1.00 | 1.00 | **1.00** |
| one BEq+ direction | 0.10 | 0.50 | **0.53** |
| near-miss (one coefficient off) | 0.10 | 0.15 | **0.34** |
| reward-hack boilerplate | 0.10 | 0.15 | **0.24** |

The near-miss and the hack were indistinguishable before; they now differ by
0.10.

**Ordering constraint (important).** Structural similarity cannot detect
semantic *inversion* — flipping `<` to `>` is one token and scores ~0.98, the
same as a harmless typo. The similarity weight is therefore capped so that no
amount of structural resemblance can outscore actually proving equivalence.
Similarity guides the search; BEq+ certifies the answer.

**Expectations.** Denser shaping is well motivated but not guaranteed to pay off:
[FormalRewardBench-adjacent work](https://arxiv.org/abs/2605.10141) finds partial
credit "produces a smoother reward signal but does not improve the final solve
rate". Our own SFT→RL null result is consistent with that. Treat this as a
hypothesis to test (`make train-guided`), not a fix.

The **shaped** reward is a process-level signal derived from BEq+'s own internals.
BEq+ proves equivalence in *two* directions (gold⇒pred and pred⇒gold) and only reports
success when both hold; the per-direction results were previously discarded. One
direction proving means the prediction is a strictly stronger or weaker version of the
gold — genuinely close — so it earns partial credit. The ladder is monotone, so the
argmax is still exact BEq+ equivalence: it shapes the path, not the target.

Concretely (`python -m reward.reward_fn`):

| case | typecheck | composite | shaped |
|---|---|---|---|
| equivalent | 1.00 | 1.00 | 1.00 |
| strictly stronger (one direction) | 1.00 | 0.10 | **0.50** |
| semantically wrong | 1.00 | 0.10 | 0.15 |
| reward-hack boilerplate | 1.00 | 0.10 | 0.15 |

A near-miss gets 5× the gradient it would under the strict composite, while the hack
still bottoms out — shaping without opening a new exploit.

### Curriculum

`make train-curriculum` (→ `scripts/run_curriculum.sh`) trains **from scratch** in two
phases: the first half on the cheap, dense type-check reward to learn Lean syntax, then
a checkpoint handoff, then the second half on BEq+ for semantics.

```bash
make train-curriculum                                   # 15 + 15 steps, shaped phase 2
TOTAL_STEPS=40 SWITCH_AT=20 make train-curriculum       # custom split
PHASE2_REWARD=compute_score_composite make train-curriculum
```

Two implementation notes worth knowing before changing it:

- **Why two runs, not a mid-run reward switch.** verl doesn't pass the global step to
  `custom_reward_function` (`extra_info` carries only dataset fields + `num_turns`), so
  an in-run switch needs a call-counting hack that interleaved validation passes would
  corrupt. A checkpoint handoff is robust and is how curricula are normally built.
- **Why phase 1 stops at the midpoint** rather than converging: by step ~26 the
  type-check objective saturates at 100% and the policy collapses onto trivially-true
  boilerplate with entropy ~0.62. Handing off from a collapsed policy leaves phase 2
  nothing to explore with.

## Plots and per-reward quantification

```bash
make plots     # → results/training_curves.png, training_metrics.csv, reward_impact.md
```

`scripts/plot_results.py` scrapes verl's per-step console metrics out of `logs/`,
groups them by arm (one log may contain several arms back-to-back), and quantifies what
each reward actually did at the training level. The most useful column is **dead
steps** — steps where every rollout in every group earned the same reward, so GRPO's
advantage was zero and the step contributed no gradient — split by cause:

| arm | steps | dead steps | starved | saturated |
|---|---|---|---|---|
| composite (strict BEq+), from scratch | 30 | 18/30 | **18** | 0 |
| type-check-only, from scratch | 30 | 10/30 | 4 | **6** |
| **SFT → RL (shaped)** | 15 | **0/15** | **0** | 0 |

*starved* = the reward never fired for any rollout (signal too sparse to learn from);
*saturated* = every rollout already succeeds (nothing left to optimise). The composite
arm spent **60% of training producing no gradient at all**, entirely from starvation —
the measured mechanism behind its poor result below, and the thing the shaped and
curriculum arms exist to fix.

Both read `configs/run_grpo.sh`, which wires the reward function in via
`reward.custom_reward_function.{path,name}` and reads
`REWARD_FN_NAME=compute_score_composite|compute_score_typecheck_only` from the
environment. Every other knob (batch size, rollout group size, LR, step count) is a
shell variable at the top of that script, overridable via env var
(`TRAIN_BATCH_SIZE=32 make train-composite`, etc.) or extra Hydra overrides passed
straight through (`bash configs/run_grpo.sh trainer.total_epochs=5`).

## Evaluating (the actual comparison)

**Do not compare the two arms using verl's own `val-core/.../acc/mean@1`.** That metric
is just the mean of *whatever reward function that run used* — the composite arm's is
`0.1·typecheck + 0.9·beq_plus`, the baseline's is the raw type-check rate. They are on
different scales, and the composite number can't be decomposed on its own (a score of
0.0825 is consistent with anything from "82.5% type-check, 0% BEq+" to "8.25% both",
since BEq+ implies type-check).

`make evaluate` does the valid comparison: it merges both final checkpoints out of
verl's FSDP-sharded format and re-scores them — plus the untrained base model — with
**both metrics separately**, on the same validation examples.

```bash
make evaluate              # merge-checkpoints + score base/typecheck/composite
STEP=20 make evaluate      # compare a different checkpoint step
N_EVAL=200 make evaluate   # more validation examples
```

## Results

30 training steps per arm (SFT: 2 epochs / 30 steps), `Qwen2.5-Coder-0.5B-Instruct`,
80 validation examples, greedy decoding, both metrics scored identically for every
checkpoint (`results/ablation_comparison_chatfixed.json`):

| checkpoint | type-check % | BEq+ % |
|---|---|---|
| base (untrained) | 0.0% | 0.0% |
| type-check-only RL @ 30 | **100.0%** | 3.8% |
| composite (BEq+) RL @ 30 | 83.8% | 0.0% |
| curriculum phase-1 (type-check) @ 15 | 43.8% | 1.2% |
| **SFT @ 30** | 72.5% | **33.8%** |
| SFT → RL (shaped) @ 15 | 72.5% | 30.0% |

Measurement noise: re-running the SFT evaluation gave 73.8% / 32.5% versus 72.5% /
33.8% — vLLM batching makes even greedy decoding vary by ~1-2 examples out of 80, so
differences smaller than that mean nothing.

**The paper's premise reproduces, emphatically.** The type-check-only arm saturated its
reward completely — 100% of rollouts type-checking by step 26, `actor/pg_loss` exactly
0.0 — while semantic correctness stayed at 3.8%. What it emits shows the hack directly
(`results/typecheck_only_sample_generations.md`): asked to formalize "13 choose 2 = 78",
it outputs `theorem x : x - 1 = -1`. It learned to emit trivial arithmetic identities
that always elaborate, ignoring the input entirely — precisely the "syntactically valid
but semantically vacuous" failure the paper describes. Note the diagnostic shape:
**highest type-check, near-zero BEq+**.

**Neither RL arm learned semantics from scratch.** Both sit at ~0-4% BEq+. The mechanism
is reward sparsity interacting with GRPO: advantages are computed *within* a rollout
group, so if no sample in a group earns the semantic reward the gradient is exactly
zero. Measured, the composite arm spent **18 of 30 steps producing no gradient at all**,
every one from starvation (see the dead-step table above).

**SFT solves what RL could not — by ~9×.** Plain supervised fine-tuning on
(informal → gold formal) pairs reaches **33.8% BEq+** in ~6 minutes of training, versus
3.8% for the best RL arm after 25+ minutes. It also inverts the reward-hacking
signature: SFT has *lower* type-check than the hacked baseline (72.5% vs 100%) but ~9×
the semantic accuracy, because it is solving the task rather than gaming the checker.

The reason is structural. BEq+ as a *reward* only informs the policy when it happens to
fire, which for a weak policy is almost never. The same BEq+ signal as a *supervised
target* is dense — every example teaches the exact reference formalization.

**SFT does fix RL's cold start — but 15 steps of RL on top added nothing measurable.**
Starting RL from the SFT policy, the shaped reward fires immediately (0.49 at step 1,
versus 0.0 for the first ~19 steps from scratch) and no step is starved, so the
mechanism works exactly as intended. Yet the resulting checkpoint scores 30.0% BEq+
against SFT's 32.5% — indistinguishable given ~±2 examples of measurement noise. The
reward oscillated between 0.19 and 0.51 throughout with no trend, while entropy fell
monotonically (0.42 → 0.29): the policy sharpened without getting better on held-out
data.

Read honestly, that is a *null result at this scale*, not evidence that RL cannot help:
15 steps × batch 8 is only ~120 prompts of experience on a 0.5B model. What it does
establish is that the expensive part (BEq+ in the loop) is now unblocked and correctly
instrumented, so the interesting question — does RL add anything over SFT given a real
step budget — is a compute question rather than an engineering one.

BEq+ remains an excellent *metric* throughout — it is what exposed the baseline's reward
hacking in the first place.

### Does RL add anything on top of SFT? (paired tests)

Two attempts, both evaluated against the SFT policy on the *same* 80 examples, so
McNemar's paired test applies:

| comparison | metric | change | discordant | p |
|---|---|---|---|---|
| SFT→RL shaped | BEq+ | 26→24 | lost 7, gained 5 | 0.774 |
| SFT→RL shaped | type-check | 59→58 | lost 9, gained 8 | 1.000 |
| SFT→RL guided | BEq+ | 26→19 | lost 10, gained 3 | 0.092 |
| SFT→RL guided | type-check | 59→65 | lost 6, gained 12 | 0.238 |

**Nothing is significant.** The eye-catching 8.7pp BEq+ drop for guided is p=0.092 —
suggestive, not established. The directional pattern (type-check up, BEq+ down) is what
you would expect if the reward is easier to satisfy by emitting valid, structurally
similar Lean than genuinely equivalent Lean, but it is not demonstrated.

**These runs were underpowered.** At n=80 with p≈0.3, the minimum detectable paired
difference is ~12.6pp — larger than the effect we were trying to measure:

| eval n | detectable BEq+ difference |
|---|---|
| 80 | ~12.6 pp |
| 200 | ~8.0 pp |
| **400** | **~5.6 pp** |
| 800 | ~4.0 pp |

The validation set has been enlarged to 400 (`scripts/prepare_dataset.py`). The slice is
pinned to a fixed offset so the original 80 remain an exact prefix — cached per-example
results stay valid and comparable rather than being silently re-based.

## Repo layout

```
configs/run_grpo.sh     GRPO launch script (both ablation arms select via REWARD_FN_NAME)
reward/beq_plus.py       BEq+ metric — persistent Lean REPL wrapper (vendored algorithm)
reward/reward_fn.py       verl-facing reward functions (composite / typecheck-only)
scripts/prepare_dataset.py   Lean-Workbook → verl parquet format
scripts/run_curriculum.sh    two-phase from-scratch curriculum (pass@ → BEq+)
scripts/evaluate_checkpoints.py  the valid head-to-head (both metrics, both arms)
scripts/plot_results.py      training curves + per-reward impact quantification
scripts/test_lean_interact.py   standalone Lean/Mathlib sanity check
results/                  ablation_comparison.json + sample generations + plots
repos/verl/               cloned RL framework (editable install)
repos/mathlib4/           Mathlib4 @ v4.8.0-rc1 (built via `lake exe cache get`)
models/                   downloaded base model + HF cache
data/                     prepared train/val parquet (+ data/smoke/ for the smoke test)
hpc/                      SLURM transfer notes + job template (see hpc/README.md)
```