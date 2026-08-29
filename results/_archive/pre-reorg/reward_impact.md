# Per-reward impact at the training level

Parsed from verl's per-step console metrics (`results/training_metrics.csv`)
and the run logs in `logs/`.

**Read the `reward` column with care**: each arm optimises its own reward
function, so these values are on different scales and are NOT comparable to
each other. Only the columns beside them (entropy, pg_loss, response length,
KL) and the separate common-metric eval support cross-arm comparison.

| arm | steps | reward first→last | entropy first→last | dead steps | ‑ starved | ‑ saturated | resp_len first→last |
|---|---|---|---|---|---|---|---|
| SFT → RL (shaped) | 200 | 0.906→0.764 | 0.098→0.022 | 18/200 | 1 | 2 | 65.7→55.5 |
| composite (strict BEq+) | 30 | 0.000→0.075 | 1.109→0.628 | 18/30 | 18 | 0 | 107.3→76.7 |
| rl_scratch gated | 12 | 0.000→0.000 | 3.181→2.163 | 12/12 | 12 | 0 | 103.2→110.5 |
| typecheck-only | 30 | 0.000→1.000 | 1.080→0.623 | 10/30 | 4 | 6 | 115.2→127.5 |
| gated_clean (disjoint) | 100 | 0.358 mean, r=+0.010 | 0.141→0.157 | 16/100 | — | — | 76.3→79.2 |
| **placebo_clean (control)** | 50 | 0.108 mean, r=−0.083 | 0.143→0.190 | 18% | — | — | 75.4→65.1 |

## What each arm's reward measures

- `compute_score_typecheck_only`: type-check pass rate (1.0 = all rollouts elaborate)
- `compute_score_composite`: 0.1*typecheck + 0.9*BEq+ (strict, binary BEq+)
- `compute_score_shaped`: 0.10 typecheck + 0.20 one-direction + 0.70 both (graded)
- `compute_score_gated`: semantic signal only; a flat floor when nothing proves
- `compute_score_placebo`: **no information at all** — a hash of the rollout
  text, tuned to reproduce the gated arm's advantage geometry. The control.

## The placebo control, and why the reward column is a trap

The `reward vs step` correlation is ~0 for the gated arm (r=+0.010) and ~0 for
the placebo (r=−0.083). **On training-reward evidence alone the two arms are
indistinguishable**, which is why "the training reward is flat, so the reward
teaches nothing" was never a safe inference.

Their *validation* trajectories diverge sharply and significantly:

| step | gated (BEq+) | placebo | BEq+ contribution | p |
|---|---|---|---|---|
| 10 | 38.0% | 34.2% | +3.7pp | 4.0e-2 |
| 30 | 34.8% | 30.2% | +4.5pp | 2.7e-2 |
| 50 | 34.2% | 24.5% | **+9.8pp** | 5.3e-6 |

Both arms decline from the 38.8% SFT starting point, because GRPO at this batch
size drifts (§1 of FINDINGS.md). But the *rate* of decline is what the reward
controls, and BEq+ halves it — entirely in the semantic metric, with type-check
statistically identical between arms at every step.

Failure-mode forensics over the same interval (step 10 → 50) show the two kinds
of damage are qualitatively different:

| | gated (BEq+) | placebo |
|---|---|---|
| BEq+ lost / gained | 28 / 13 | 47 / 8 |
| lost → semantically **unrelated** | 10 (36%) | **36 (77%)** |
| lost → weaker (vacuity-ward) | 12 | 4 |
| lost → no longer type-checks | 6 | 7 |
| lost with *identical* text (scorer noise) | 0 | 0 |

Placebo damage is random semantic scrambling. BEq+ damage is smaller and
concentrated on the "weaker" rung — the known artifact of the short-circuiting
cascade (see reward/beq_plus.py DIRECTION SEMANTICS).

## Why `pg_loss == 0` matters

GRPO's advantage is computed *within* each rollout group. If every sample in a
group earns the same reward, the advantage is zero and the step contributes no
gradient — a **dead step**. Two opposite situations produce dead steps and are
indistinguishable in the loss alone, so the table splits them by reward level:

- **starved** (reward ≈ 0): the reward never fired for *any* rollout. The
  signal is too sparse for the policy to get off the ground. This is the
  failure mode of the strict composite reward from a cold start.
- **saturated** (reward ≈ peak): *every* rollout already succeeds, so there is
  nothing further to optimise. This is where the type-check-only arm ends up —
  and, since its objective is trivially satisfiable, where reward hacking lives.

A live gradient is necessary but **not** sufficient: the clean gated run has one
on 84% of steps and still declines, because ~37% group informativeness at batch
4 leaves each Adam update driven by roughly 1.5 prompts. That is the noise the
placebo control quantifies.
