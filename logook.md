# Done:

## 2026-08-20 — Rejection sampling, the exploitability dose-response, and batch-16 GRPO

### Headline results

**1. Drift was the destroyer, not the reward — confirmed.** Rejection-sampling
fine-tuning (RFT) uses the identical BEq+ verdicts with *no advantage estimate at
all*. Every RFT arm holds SFT parity (±0.5pp) after 3 full epochs, where every
GRPO arm decayed (gated −4.5pp, placebo −14.3pp by step 50). Remove advantage
estimation and the decay vanishes entirely.

**2. But removing drift revealed no gain.** The sharp prediction — that BEq+
would turn net-positive once noise was gone — **failed**. Best arm is +0.5pp
(p=0.83). Most likely a self-distillation ceiling: the arm trains on the model's
own BEq+-certified output over the 631 prompts it already solves.

**3. The exploitability claim, quantified as a dose-response.** Three
size-matched arms (1,203 pairs each, one scoring pass, differing only in
acceptance criterion):

| training data BEq+-certified | mean Δ BEq+ vs SFT | mean Δ type-check |
|---|---|---|
| 100% (`rft_beq`)  | **+0.1pp** | +3.5pp |
| 48%  (`rft_tc`)   | **−1.0pp** | +5.2pp |
| 0%   (`rft_tcnb`) | **−2.1pp** | +5.3pp |

Monotonic in both columns. Linear interpolation at 48% predicts −1.04pp;
observed −1.0pp. Semantic accuracy is a near-linear function of how
BEq+-certified the training data was, in a setting with **zero advantage noise**.
Undiluted contrast (`rft_beq` vs `rft_tcnb`): −2.8 / −3.3 (p=0.047) / −0.5pp.

The arms are structurally indistinguishable (median length 104/109/114, vacuous
`∃a, a=…` at 0.2% in all three), so this is **not** surface-form degeneration —
it is well-formed Lean that means something else. Of the 3,590
type-checks-but-not-BEq+ rollouts: 86.9% semantically unrelated, 13.1% strictly
weaker, 0% stronger.

**4. Retention, not learning, is what BEq+ buys.** Decomposing every arm into
retention (of SFT's 155 correct) vs gains (of SFT's 245 failures):

| arm | kept | lost | gained | net |
|---|---|---|---|---|
| `rft_beq-17` | 145 (93.5%) | 10 | 12 (4.9%) | **+2** |
| `rft_beq-51` | 140 (90.3%) | 15 | 13 (5.3%) | −2 |
| GRPO gated bs4 @50 | 119 (76.8%) | 36 | 18 (7.3%) | −18 |
| GRPO placebo bs4 @50 | 90 (58.1%) | 65 | 8 (3.3%) | −57 |

Of the +9.8pp BEq+-vs-placebo gap at step 50, **29 of 39 examples (74%) are
retention**, only 10 are gains. Revises last session's framing: BEq+ mainly
*protects* capability under noisy updates; its teaching contribution is real
(7.3% vs 3.3% conversion) but small.

**5. The loop is starved, not broken.** Three independent proofs it works:
type-check-only drove type-check 76.2%→100% (saturation); every arm converts
real examples; the dose-response shows BEq+ is causally relevant. The arithmetic:

- group composition: **47.0% starved / 22.1% saturated / 30.9% informative**
- at batch 4: **1.24 informative groups per Adam update**
- gated @50: gains **+4.5pp**, losses **−9.0pp** → net −4.6pp

69% of every batch produces zero gradient. Fix the loss channel and BEq+ RL is
net-positive without one extra gain.

**6. Headroom is real.** SFT-step390 on the 1,191 RL prompts (temp 1.15):
pass@1 38.9% → pass@2 44.9% → pass@4 49.4% → **pass@8 53.0%**. 14.1pp of
prompts the policy *can* solve but doesn't reliably.

**7. Placebo geometry was off by 2×.** Measured against the gated arm's step-0
geometry on the exact RL pool (same policy, temp, k): the placebo at `p=0.30`
runs **+12.8% informative groups and +21.1% within-group σ** — not the ~11%
previously carried. `BEQ_PLACEBO_ROLLOUT_P=0.20` matches both within 1%.
NOT applied to the bs16 run, deliberately: keeping `p=0.30` preserves continuity
with `placebo_bs4` so the batch-size comparison uses an identical reward.

### Runs completed
- 3 RFT arms × 3 epochs = **9 checkpoints trained and evaluated** at n=400
- **`placebo_bs16`: all 80 steps**, 8 checkpoints (1:07:33; no Lean → ~51s/step)
- `gated_bs16`: running, ~10 min/step (Lean-dominated), projects to step 55–60

---

## 2026-08-20 (later) — batch-size confirmed, 3B migration, and two caught bugs

### Item 0 SETTLED: batch 16 halves the pure-noise damage

`placebo_bs4` vs `placebo_bs16` — **identical zero-information reward**, only the
batch differs, so any gap is batch size acting on update noise:

| step | bs4 kept | bs16 kept | bs4 lost | bs16 lost | bs4 gain | bs16 gain |
|---|---|---|---|---|---|---|
| 10 | 82.6% | 86.5% | 27 | 21 | 9 | 12 |
| 20 | 71.6% | **86.5%** | 44 | **21** | 12 | 8 |
| 30 | 71.0% | 77.4% | 45 | 35 | 11 | 12 |
| 40 | 65.2% | 72.3% | 54 | 43 | 11 | 9 |
| **50** | **58.1%** | **72.9%** | **65** | **42** | 8 | 14 |

At step 50: **−14.3pp → −7.0pp** (+7.3pp, p=0.0008). **Losses fall, gains stay
flat** (8–12 in both arms at every step) — a reward with zero information cannot
teach, so the entire improvement is noise-averaging. The gap widens with steps,
which is what accumulating-vs-cancelling noise looks like.

### 3B migration set up (Qwen2.5-Coder-3B-Instruct)

New `data_3b/` split, **verified all three sets disjoint** (val∩SFT, val∩RL,
SFT∩RL all zero):

| slice | size | note |
|---|---|---|
| val | 1,000 | MDE ~3.5pp vs ~5.6pp |
| SFT | 8,000 | |
| RL | **4,300** | **3.4× the old pool** — 268 steps/epoch at batch 16, so a 150-step run never repeats a prompt |

**The first 400 rows of `data_3b/val.parquet` are byte-identical to
`data/val.parquet`** (`VAL_OFFSET=400`, same seeded shuffle). So 3B evals at
n=400 pair directly with all 37 cached 0.5B evals under McNemar — the model-size
effect is measurable as a paired test, contradicting the earlier "never compare
a 3B number to 38.8%" caution.

Chain queued with SLURM dependencies: SFT → eval-and-pick-best → rollouts +
**go/no-go gate** + placebo fit → {gated, placebo, gated_filtered}. The gate
exits non-zero if informative <25% or saturated >45%, so `afterok` blocks all
three GRPO arms automatically if the scale-up lands past the informative peak.

`scripts/calibrate_placebo.py` fits the control's constants to the measured
policy geometry — they are policy-specific, and 0.5B's values would give a
control that is not a control. Validated by reproducing the 0.5B fit exactly
(0.37/0.20, within 1%).

### Two bugs caught before they produced wrong answers

**1. Starved-subset mis-mapping (would have inverted the model-size decision).**
`generate_rollouts.py` dedupes and drops prompts >768 tokens, then re-indexes
from 0 (1,280 rows → 1,191 groups), so `prompt_index` is NOT a parquet row
number. A positional join selected the wrong prompts; the tell was "starved"
prompts scoring 32/32, impossible for something that went 0/8. Left uncaught it
would have reported ~55% of unreachable prompts becoming solvable and pointed at
*exploration* — the opposite of the truth. Now joins on prompt TEXT and verifies
the gold (which immediately excluded 1 prompt whose text recurs with a different
gold). **Rule: `prompt_index` is never a parquet row number; join on content.**

**2. 3B SFT OOM at 38.25GB.** Not the offload flags (correctly applied). Two
causes: verl's `model_dtype` defaults to **fp32** (3B ≈ 12GB params + 12GB grads
+ 25GB Adam), and **at world size 1 FSDP falls back to `NO_SHARD`** — no
sharding at all, which no offload setting rescues. Fixed with 2-GPU sharding +
`MICRO_BATCH=1`. **0.5B only ever fit on one GPU because it was small enough for
NO_SHARD.**

### Lean cost (Q3)

Type-check short-circuit implemented and **verified output-identical offline**
against all 2,232 non-type-checking rollouts — the cascade never produced a
non-default result when type-check failed, so skipping it is exact, not an
approximation. Phase timers added. Memo-caching killed by data (0.7% repeats).
Direction short-circuit was already implemented upstream.

---

# TODO

0. **Evaluate `placebo_bs16` vs `placebo_bs4`** (job 1413888 running). Identical
   zero-information reward, only the batch differs — this isolates the batch-size
   effect on update noise with no reward semantics involved. Prediction: losses
   drop sharply from 65 and retention climbs well above 58%. If it does *not*,
   batch size is not the mechanism and the noise story needs rethinking.

1. Fix the training loop — highest confidence, already running. The gain channel is already +4.5pp. Cut losses and BEq+ RL goes net-positive without one extra gain. Batch 16 gives ~5 informative groups/update instead of 1.24. Beyond that: FILTER_GROUPS=1 (DAPO-style) discards degenerate groups and refills — the config calls it strictly dominant, off only on cost, and cost is no longer the constraint on an A100.

2. Don't buy a larger model yet — run the £0 test first. Generate k=32 on the 560 starved prompts and count how many ever succeed. If coverage barely moves, it's capability and scale is the answer. If it jumps, it's exploration and rollout_n/temperature is the answer. One generation job, no training, and it decides a question that otherwise costs weeks. Do this before committing to a bigger model.

3. More eval data beats more training data. Raising n=400 → 1000+ is ~28 min/checkpoint now and would resolve the 1–3pp effects that keep coming out directionally right and statistically mute. More training prompts requires re-splitting and forfeits your baseline — don't, until the loop is fixed.

4. Model size, when you get there, is confounded with the 16GB-era constants (ppo_max_token_len_per_gpu=896, entropy off) and starts a new comparison series against 37 cached 0.5B evals. Worth doing deliberately, not as a reflex.

5. Tighten the placebo to `BEQ_PLACEBO_ROLLOUT_P=0.20` for any *new* placebo
   baseline (not for comparisons against `placebo_bs4`), closing the magnitude
   objection to the +9.8pp result.

The short version: the loop is starved rather than broken, and that's the cheapest thing to fix. The model-size question is real and the evidence genuinely points that way — but it's one job away from being settled instead of assumed.
