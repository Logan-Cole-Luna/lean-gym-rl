#!/usr/bin/env python3
"""Prefetch the base model + training dataset for offline compute-node jobs.

Run this on a login node with internet access. Downloads into
PROJECT_ROOT/models/ (model) and the HF datasets cache (dataset) so SLURM jobs
can run with HF_HUB_OFFLINE=1 (see hpc/cc_env.sh).

Mirrors MixtureOfMathExperts/hpc/prefetch_models.py's structure.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

DEFAULT_MODEL = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
DEFAULT_MODEL_DIR = "models/qwen2.5-coder-0.5b-instruct"
DEFAULT_DATASET = "internlm/Lean-Workbook"


def _resolve_project_root(explicit: str | None) -> Path:
    if explicit:
        return Path(explicit).expanduser().resolve()
    return Path(__file__).resolve().parents[1]


def _read_token() -> str | None:
    env_token = (os.environ.get("HF_TOKEN") or "").strip()
    if env_token:
        return env_token
    token_path = Path.home() / ".hf_token"
    if token_path.exists():
        token = token_path.read_text().strip()
        if token:
            return token
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Prefetch model + dataset for offline HPC jobs")
    parser.add_argument("--project-root", type=str, default=None, help="Repo root (default: inferred)")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="HF model repo ID")
    parser.add_argument("--model-dir", type=str, default=None, help="Local dir to save the model into")
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET, help="HF dataset repo ID")
    parser.add_argument("--skip-model", action="store_true")
    parser.add_argument("--skip-dataset", action="store_true")
    args = parser.parse_args()

    project_root = _resolve_project_root(args.project_root)
    hf_home = Path(os.environ.get("HF_HOME", project_root / "models" / ".hf_cache"))
    hf_home.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(hf_home)

    token = _read_token()
    if token:
        os.environ["HF_TOKEN"] = token
    else:
        print("[prefetch] No HF_TOKEN / ~/.hf_token found — proceeding unauthenticated "
              "(fine for public repos, but rate-limited).")

    if not args.skip_model:
        from huggingface_hub import snapshot_download

        model_dir = args.model_dir or str(project_root / DEFAULT_MODEL_DIR)
        print(f"[prefetch] Downloading model {args.model} -> {model_dir}")
        snapshot_download(args.model, local_dir=model_dir)
        print(f"[prefetch] Model ready: {model_dir}")

    if not args.skip_dataset:
        from datasets import load_dataset

        print(f"[prefetch] Downloading dataset {args.dataset} (HF_HOME={hf_home})")
        load_dataset(args.dataset, split="train")
        print("[prefetch] Dataset cached.")

    print("")
    print("Done. Both are now cached locally — compute-node jobs can run with")
    print("HF_HUB_OFFLINE=1 (set automatically by hpc/cc_env.sh).")


if __name__ == "__main__":
    main()
