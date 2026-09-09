# Distilled (CoT) vs Baseline SFT -- round 2 (to ~200 steps)

Same base model (Qwen2.5-Coder-3B-Instruct), LR 5e-5 constant, same pinned
760-row `data_locolib/val_proof.parquet`. The baseline here is the **uncapped**
full seed-0 SFT split (11,751 rows; the original had a 1024-token filter that
dropped 2,017). Distilled trains on 8,389 rows with the teacher's `<think>`+proof.

## Headline (best checkpoint)

| series | step | type-check | BEq+ |
|---|---|---|---|
| baseline (uncapped, gold) | 160 | 70.4% | 63.3% |
| distilled (CoT) | 150 | 65.1% | 51.6% |

Δ BEq+ = -11.7 pp, paired McNemar exact p = 2.22e-10 (n=760: both 337, neither 224, baseline-only 144, distilled-only 55).

## Trajectories (BEq+ %)

- **Baseline (uncapped)**  40:52.8  80:58.7  120:62.5  160:63.3  200:61.7  225:62.8
- **Distilled (CoT)**  33:43.3  66:47.8  96:48.9  100:43.4  125:43.4  150:51.6  175:46.7
- **Baseline (capped, orig.)**  38:39.2  76:43.9  114:44.9
- **Matched-gold**  33:50.7  66:57.9  96:59.2

## Read
- The uncapped gold baseline **plateaus ~62-63% BEq+ / ~70% type-check by step 120-160**.
- The distilled CoT model **peaks ~51.6% at step 150, then declines** (step 175: 46.7%) --
  it overfits (train loss < 0.06). Gold beats CoT by ~12 pp at their bests.
- There is a transient BEq+ dip at distilled 100/125 right after the resume; it recovers by 150.

## Caveats
- Not a clean 1-to-1 control (different row set + domain skew; see `table_training_distribution.tex`).
  The instance-matched control is `sft3blocolib_proof_matchedgold` (BEq+ 59.2% @ step 96), shown as a reference line.
- Distilled is scored with `score_standalone` (union gold+prediction context) because it emits
  self-contained snippets; the gold series use plain `score`. If anything this flatters distilled.
- Distilled needs a 1536-token generation budget vs 128 for the gold models (`table_output_lengths.tex`).

## Files
figures/ fig_training_loss, fig_eval_trajectory, fig_outcome_breakdown, fig_output_length,
         fig_target_length, fig_domain_distribution  (.png + .pdf)
tables/  table_config, table_results, table_eval_by_step, table_outcome_breakdown,
         table_output_lengths, table_target_lengths, table_training_distribution  (.tex)

Regenerate: `source hpc/cc_env.sh && python scripts/misc/compare_distilled_baseline.py [--full]`
