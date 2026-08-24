# Archived SLURM jobs

Kept for provenance — every published 0.5B number came from these — but not part
of the current pipeline. The 3B series superseded them:

| archived | superseded by |
|---|---|
| `grpo_bs16.slurm` | `hpc/grpo_3b.slurm` (same arms, 3B, series-scoped) |
| `rft_arms.slurm`, `rft_tcnb.slurm` | the rejection-sampling dose-response is settled; see `logook.md` |
| `rft_eval.slurm` | `hpc/grpo_eval.slurm` |
| `starved_k32.slurm` | `hpc/passk_3b.slurm` answers the same question with a proper pass@k estimator |

They still reference the 0.5B `data/` split and the pre-`job_prelude.sh` inline
preamble. Re-running one means updating both first.
