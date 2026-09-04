# Does filtering the distilled targets to BEq+-valid help? (miniF2F, OOD)

Two SFT runs, identical except the training data: **all teacher targets
that elaborate** (9,149 rows) vs **only the ~76% BEq+-equivalent to the gold**
(7,181 rows). Both cosine LR from scratch, the LoCoLib pinned-760 slice folded
into training since the eval moved to miniF2F, scored with `score_standalone`
on the same 244-row out-of-domain slice.

## Headline

Each row is that run's **final** checkpoint, not a checkpoint selected on this
table's own BEq+ column. Both metrics score the generated *statement* only --
see Caveats.

| series | step | elaborates | BEq+ | Δ BEq+ vs untrained | McNemar p |
|---|---|---|---|---|---|
| untrained base | 0 | 45.5% | 5.3% | — | — |
| All teacher targets | 200 | 59.8% | 13.5% | +8.2 pp | 0.00032 |
| BEq+-valid targets only | 250 | 52.9% | 12.7% | +7.4 pp | 0.00091 |

## Trajectories (BEq+ %)

- **All teacher targets**  25:7.0  50:11.9  75:15.6  100:15.2  125:13.1  150:15.2  175:13.9  200:13.5
- **BEq+-valid targets only**  25:12.3  50:11.9  75:11.9  100:11.5  125:14.3  150:12.7  175:11.9  200:13.9  225:13.1  250:12.7

## Read

Filtering **does not help**. At the final checkpoint of each run BEq+ is 13.5% (all targets, step 200) against 12.7% (valid only, step 250): +0.8 pp, McNemar p=0.82 -- not significant, and at matched steps 150/175/200 neither run leads either.

**Statement elaboration is clearly worse for the filtered run** (~52-61% vs ~60-70% at every step -- see `fig_elaborate_rate`). Dropping 24% of the training data cost more elaboration skill than the label-cleanliness bought back.

Both plateau by step ~75-125 then wander in a 12-16% band; longer training does nothing on this OOD slice. The earlier hint that filtering helped (+2.9 pp, p=0.09) was a noisier setup (constant LR, no folded val, only to step 175) and does not survive here.

For this transfer target, **data volume beats target-statement purity**.


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
