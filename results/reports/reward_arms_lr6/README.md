# Does the reward function matter? (LoCoLib proof-pair, in-domain)

Three GRPO arms, identical except `reward.custom_reward_function.name`,
all resuming from the same SFT checkpoint. The question is whether a
semantic reward (BEq+) beats an exploitable proxy (type-check only) on
held-out BEq+, and whether either beats not doing RL at all.

All records re-derived under `leanprover/lean4:v4.23.0` on 2026-09-08; see
`../REWARD_ARMS.md` section 0 for why that matters and what it changed.

## Headline

Each row is that arm's **final** checkpoint (step 90), not a checkpoint
selected on this table's own BEq+ column. Both metrics score the generated
*statement* only, see Caveats.

| arm | step | elaborates | BEq+ | Δ BEq+ vs SFT | McNemar p |
|---|---|---|---|---|---|
| SFT baseline | 76 | 67.0% | 58.2% | -- | -- |
| BEq+ gated | 90 | 70.0% | 61.4% | +3.29 pp | 6.2e-04 |
| Six-outcome ladder | 90 | 69.5% | 60.5% | +2.37 pp | 3.0e-02 |
| Type-check only (exploitable) | 90 | 67.4% | 58.7% | +0.53 pp | 6.6e-01 |

## Trajectories (BEq+ %)

- **BEq+ gated**  10:59.2  20:59.1  30:59.1  40:58.9  50:59.3  60:59.9  70:58.4  80:60.7  90:61.4
- **Six-outcome ladder**  10:58.6  20:59.1  30:59.2  40:59.3  50:59.5  60:59.7  70:60.3  80:61.2  90:60.5
- **Type-check only (exploitable)**  10:58.2  20:58.0  30:57.6  40:57.1  50:58.3  60:59.2  70:59.2  80:58.6  90:58.7

## Read

**Only the semantic rewards beat SFT.** BEq+ gated +3.29 pp (p=6.2e-04) and the six-outcome ladder +2.37 pp (p=3.0e-02); the exploitable type-check proxy is +0.53 pp (p=0.66, not significant).

**Arm against arm at step 90**, paired on the same rows:

- gated vs type-check: +2.76 pp BEq+, McNemar p=0.0025
- outcome vs type-check: +1.84 pp BEq+, McNemar p=0.0704

Read the trajectories, not the final numbers: the arms cross over between
checkpoints and the paired differences are the honest measure.

**Every gain is a reduction in outputs that fail to elaborate.** BEq+
conditional on elaborating is flat at ~88% across every checkpoint of every
arm. GRPO here is not teaching better mathematics, it is teaching the model
to emit something Lean accepts. The reward that directly pays for
elaboration is the worst of the three at achieving it.

## Caveats

- **Proof validity is not measured, by either metric.** Eval scoring goes
  through `BEqPlusScorer.typecheck_ex`, which replaces the proof body with
  `sorry` before Lean is called. *elaborates* means the generated
  **statement** elaborates; *BEq+* means that statement is bidirectionally
  equivalent to the gold **statement**. `proved` is 0.0% throughout.
  Elaboration hard-gates BEq+ (`BEQ_SKIP_CASCADE_ON_TYPECHECK_FAIL=1`), so
  the two columns are nested, not independent.
- **The SFT baseline may be undertrained, and this is load-bearing.**
  Re-scored under the same toolchain: step 38 50.9%, step 76 58.2%, step 114
  60.9%. `best_sft_proof.txt` pins 76. Measured against 114 instead, the
  gain largely disappears (gated +0.53 pp, p=0.79). But step114-step76 is
  itself only +2.76 pp at p=0.092, and picking the highest SFT checkpoint is
  selecting on the statistic. The defensible claim is narrow: **the
  reward-attributable gain is the same order as the variation between
  adjacent SFT checkpoints**, and this setup cannot separate them because it
  has no selection split (`trainer.test_freq: -1`, no dev parquet).
- **No checkpoint is selected on the reported metric.** Every number is each
  arm's final step, fixed by the training schedule.
- **`max_response_length=128`**, a 16GB-card artifact that outlived its
  hardware. A truncated rollout scores 0 under every reward, so this is a
  standing gradient against longer answers, shared by all three arms. Now
  512 repo-wide; the `locolib_proof_c512` series reruns this at the new cap.
- **Cost.** Lean scoring is ~97% of a gated step and ~1% of a type-check
  step (~962 s vs ~26 s). The semantic reward's advantage costs ~37x the
  wall-clock per step.

## Layout

```
eval/<series>/     the corrected eval records, each stamped with `toolchain`
figures/           BEq+ trajectory
tables/            LaTeX: results, per-step, config
training/          verl step metrics per arm
```

Regenerate: `source hpc/cc_env.sh && python scripts/misc/build_reward_arms_report.py`
