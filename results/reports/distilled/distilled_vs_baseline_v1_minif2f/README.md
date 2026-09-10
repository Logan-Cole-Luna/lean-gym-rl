# CoT-distilled SFT vs the untrained base model

Same pinned 244-row `data_minif2f/val_proof.parquet` (out-of-domain),
same scoring path
(`score_standalone`) and the same 1536-token generation budget for both, so
nothing here is confounded by eval settings. The untrained model is a level,
not a trajectory, so it is drawn as a dash-dot reference line.

## Headline

| series | step | type-check | BEq+ | Δ BEq+ vs untrained | McNemar p |
|---|---|---|---|---|---|
| untrained base | 0 | 45.5% | 5.3% | — | — |
| Distilled | 96 | 68.9% | 16.0% | +10.7 pp | 1.3e-05 |

## Trajectories (BEq+ %)

- **Distilled**  32:11.9  64:14.3  96:16.0  128:14.8  160:15.2

## Read

SFT does two separable jobs, and `table_outcome_breakdown.tex` splits them:

1. **Make Lean that elaborates at all.** The untrained model compiles 111/244 (45.5%) of the slice; after distilled SFT that is 168/244 (68.9%). This is the bulk of the gain.
2. **Make the *intended* theorem.** Conditional on compiling, the untrained model is right 11.7% of the time and the distilled model 23.2%. The one-directional `weaker_only` bucket is near-empty here (1 vs 2), so on this slice the failures are outright wrong statements rather than drifted ones.

Distilled data trains the model: it moves BEq+ from 5.3% to 16.0% and the compile rate from 45.5% to 68.9%, both with overwhelming paired significance.

## Caveats

- The untrained number is an upper bound: standalone (union-context) scoring,
  a 1536-token generation budget, and zero-shot with no few-shot exemplars.
- Adjacent checkpoints still disagree on a sizeable minority of the 244 examples
  where the aggregate rate barely moves, so read the curve, not any single
  checkpoint. The LR is annealed over the run, which keeps the trajectory
  monotone; an earlier constant-LR run of the same corpus swung several
  points between neighbouring checkpoints.

## Contents

```
figures/   fig_beq_plus_rate, fig_compile_rate, fig_training_loss,
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
