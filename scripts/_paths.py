"""Put the repo root and every stage folder on sys.path.

Scripts are grouped by pipeline stage (data, train, pool, eval, figures, misc)
and a few import a sibling from another stage, which plain sibling imports do
not reach. Importing this module fixes that in one place:

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import _paths  # noqa: F401

Kept as sys.path manipulation rather than a package so the scripts stay directly
runnable as `python3 scripts/<stage>/<name>.py`.
"""
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
REPO = SCRIPTS.parent
STAGES = ("data", "train", "pool", "eval", "figures", "misc")

for _p in (REPO, SCRIPTS, *(SCRIPTS / s for s in STAGES)):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)
