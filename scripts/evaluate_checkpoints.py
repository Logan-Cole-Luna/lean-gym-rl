#!/usr/bin/env python3
"""Forwarder to scripts/eval/evaluate_checkpoints.py.

TEMPORARY. Eval jobs 1877647 and 1877650 were submitted before the reorganisation
and their snapshots call this path. Delete this file once both have run.
"""
import runpy
import sys
from pathlib import Path

target = Path(__file__).resolve().parent / "eval" / "evaluate_checkpoints.py"
sys.argv[0] = str(target)
runpy.run_path(str(target), run_name="__main__")
