# Per-reward impact at the training level

Parsed from verl's per-step console metrics (`results/training_metrics.csv`).

**Read the `reward` column with care**: each arm optimises its own reward
function, so these values are on different scales and are NOT comparable to
each other. Only the columns below them (entropy, pg_loss, response length)
and the separate common-metric eval (`results/ablation_comparison.json`)
support cross-arm comparison.

| arm | steps | reward first→last | peak | entropy first→last | dead steps | ‑ starved | ‑ saturated | resp_len first→last |
|---|---|---|---|---|---|---|---|---|
| SFT → RL (shaped) | 200 | 0.639→0.764 | 0.954 | 0.109→0.022 | 1/200 | 0 | 0 | 71.625→55.469 |
| composite (strict BEq+) | 30 | 0.000→0.075 | 0.075 | 1.109→0.628 | 18/30 | 18 | 0 | 107.297→76.656 |
| typecheck-only | 30 | 0.000→1.000 | 1.000 | 1.080→0.623 | 10/30 | 4 | 6 | 115.188→127.469 |

## What each arm's reward measures

- `compute_score_typecheck_only`: type-check pass rate (1.0 = all rollouts elaborate)
- `compute_score_composite`: 0.1*typecheck + 0.9*BEq+ (strict, binary BEq+)
- `compute_score_shaped`: 0.15 typecheck + 0.35 one-direction + 0.50 both (graded)

## Why `pg_loss == 0` matters

GRPO's advantage is computed *within* each rollout group. If every sample in a
group earns the same reward, the advantage is zero and the step contributes no
gradient -- a **dead step**. Two opposite situations produce dead steps and are
indistinguishable in the loss alone, so the table splits them by reward level:

- **starved** (reward ~= 0): the reward never fired for *any* rollout. The
  signal is too sparse for the policy to get off the ground. This is the
  failure mode of the strict composite reward.
- **saturated** (reward ~= peak): *every* rollout already succeeds, so there is
  nothing further to optimise. This is where the type-check-only arm ends up --
  and, since its objective is trivially satisfiable, where reward hacking lives.

Starved steps early in training are the problem the shaped/curriculum arms are
designed to fix: they exist to make rollout groups *differ* in reward so that
an advantage, and therefore a gradient, exists at all.
