# Which corpus, and does the reasoning trace matter? (miniF2F, OOD)

Four SFT runs crossing two factors: the distillation **corpus** (LoCoLib vs
Mizar) and whether the assistant target keeps the teacher's `<think>` block
or only the generated Lean. Same base model, same recipe, same pinned
244-row out-of-domain slice, scored with `score_standalone`.

## Headline

Each row is that run's **final** checkpoint, not a checkpoint selected on this
table's own BEq+ column. Both metrics score the generated *statement* only --
see Caveats.

| series | step | elaborates | BEq+ | Δ BEq+ vs untrained | McNemar p |
|---|---|---|---|---|---|
| untrained base | 0 | 45.5% | 5.3% | — | — |
| LoCoLib distilled (ours) | 160 | 67.2% | 15.2% | +9.8 pp | 1.9e-05 |
| LoCoLib distilled, no CoT | 200 | 70.5% | 16.0% | +10.7 pp | 6.2e-06 |
| Mizar distilled | 240 | 54.9% | 13.5% | +8.2 pp | 8.8e-05 |
| Mizar distilled, no CoT | 225 | 68.9% | 18.9% | +13.5 pp | 2.3e-10 |

## Trajectories (BEq+ %)

- **LoCoLib distilled (ours)**  32:11.9  64:14.3  96:16.0  128:14.8  160:15.2
- **LoCoLib distilled, no CoT**  25:12.3  50:18.0  75:15.6  100:15.2  125:15.6  150:13.5  175:14.8  200:16.0
- **Mizar distilled**  25:10.2  50:14.8  75:16.0  100:13.5  125:11.5  150:16.0  175:13.9  200:12.3  225:12.3  240:13.5
- **Mizar distilled, no CoT**  25:18.4  50:17.6  75:17.6  100:16.0  125:16.0  150:18.0  175:18.4  200:18.4  225:18.9

## Read

**The reasoning trace on LoCoLib:** with `<think>` 15.2% BEq+ / 67.2% elaborates, without it 16.0% / 70.5%.
**The reasoning trace on Mizar:** with `<think>` 13.5% / 54.9%, without it 18.9% / 68.9%.
**The corpus:** LoCoLib-distilled 15.2% BEq+ vs Mizar-distilled 13.5% on the same competition slice. Mizar is a different mathematical register (formalised
mathematics library prose, not contest problems), so this is a transfer question, not a data-quality one.

Read the trajectories, not the final numbers: these runs cross over between
checkpoints and the paired differences below are the honest measure.

- LoCoLib distilled (ours) vs LoCoLib distilled, no CoT: -0.8 pp BEq+, McNemar p=0.86
- Mizar distilled vs Mizar distilled, no CoT: -5.3 pp BEq+, McNemar p=0.011
- LoCoLib distilled (ours) vs Mizar distilled: +1.6 pp BEq+, McNemar p=0.6


## Caveats

- **Proof validity is not measured, by either metric.** Scoring goes through
  `BEqPlusScorer.typecheck_ex`, which calls
  `clean_last_theorem_string(..., add_sorry=True)`: the model's proof body is
  stripped and replaced with `sorry` before Lean is ever called. So
  *elaborates* means the generated **statement** elaborates, and *BEq+* means
  that statement is bidirectionally equivalent to the gold **statement**.
  Neither says whether the model's proof closes the goal.
  `BEqPlusScorer.check_own_proof` would check exactly that and is never called
  by `scripts/eval/evaluate_checkpoints.py`. Elaboration also hard-gates BEq+
  (`BEQ_SKIP_CASCADE_ON_TYPECHECK_FAIL=1`), so the two are nested, not
  independent.
- **No checkpoint is selected on the reported metric.** Every table and figure
  annotation uses each run's *final* checkpoint, fixed by the training
  schedule. There is no separate selection split here (`trainer.test_freq: -1`,
  no dev parquet), so an argmax over the reported BEq+ column would be
  selecting on the statistic it then reports. Read
  `table_eval_by_step.tex` -- the full trajectory -- as the result.
- The untrained number is an upper bound: standalone (union-context) scoring,
  a 1536-token generation budget, and zero-shot with no few-shot exemplars.
- Adjacent checkpoints still disagree on a sizeable minority of the 244 examples
  where the aggregate rate barely moves, so read the curve, not any single
  checkpoint. The LR is annealed over the run, which keeps the trajectory
  monotone; an earlier constant-LR run of the same corpus swung several
  points between neighbouring checkpoints.

## Contents

```
figures/   fig_beq_plus_rate, fig_elaborate_rate, fig_training_loss,
           fig_outcome_breakdown, fig_output_length, fig_target_length,
           fig_domain_distribution            (.png + .pdf)
tables/    table_results, table_eval_by_step, table_outcome_breakdown,
           table_output_lengths, table_config, table_target_lengths,
           table_training_distribution        (.tex, booktabs)
eval/      the per-checkpoint eval JSON (with per_example) and the raw
           generations, copied verbatim from results/eval/<label>/
training/  the SFT job logs these curves were parsed from, plus
           train_loss.csv (series, step, loss)
```

Regenerate: `source hpc/cc_env.sh && python scripts/misc/compare_distilled_baseline.py [--full]`
