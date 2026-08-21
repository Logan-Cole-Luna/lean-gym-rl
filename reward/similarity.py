#!/usr/bin/env python3
"""Continuous similarity signals between a predicted and a gold Lean statement.

WHY THIS EXISTS
---------------
BEq+ is a hard binary: it either proves equivalence in both directions or it
does not. Measured on our SFT policy's 80 validation outputs, that leaves most
of the data carrying no directional gradient:

    26%  does not type-check          -> reward 0.00, no information
    41%  type-checks but no BEq+      -> reward 0.15, ALL LUMPED TOGETHER
    32%  full BEq+                    -> reward 1.00

A prediction that differs from the gold by one coefficient scores exactly the
same as `theorem x : 1 = 1`. The "one BEq+ direction proves" rung added in
reward_fn.compute_score_shaped does not fix this, because it still requires a
successful Lean tactic search -- it moves the cliff, it does not interpolate it.

The signals here are *continuous* and are defined even when the statement fails
to elaborate, so they grade both dead buckets.

WHAT IS IMPLEMENTED
-------------------
1. `structural_similarity` -- GTED-*style*. The published Generalized Tree Edit
   Distance (Xu et al., arXiv:2507.07399) standardizes both statements, parses
   them into operator trees via the Lean language server, and returns a
   normalized tree-edit cost in [0,1]. It reports the highest accuracy/Kappa on
   miniF2F and joint-highest on ProofNet, and is explicitly complementary to
   BEq: BEq tests logical equivalence, GTED measures structural distance.

   THIS IS AN APPROXIMATION, NOT THE PUBLISHED METRIC. We build the tree from
   Lean's bracket structure and operator tokens rather than from a real parse,
   because routing every rollout through the Lean LSP would add a second
   Lean round-trip to a reward that is already the training bottleneck. It
   captures the same intuition (structural distance to the reference) at
   roughly zero cost. If you want the real thing, swap `operator_tree` for an
   LSP-derived parse; the rest of the pipeline is unchanged.

2. `symbol_similarity` -- Jaccard overlap of the identifiers/constants used
   (`Nat.choose`, `Real.sqrt`, ...). Cheap, and directly reflects "is the
   prediction talking about the same mathematical objects".

3. `embedding_similarity` -- the continuous embedding reward proposed in
   "Online Reinforcement Learning for Autoformalization" itself (the paper this
   project implements), which it motivates as a semantic measure requiring no
   paired data. Optional and lazily loaded: it pulls a small sentence-embedding
   model, and this project has repeatedly hit host-RAM limits, so it is OFF
   unless BEQ_USE_EMBEDDING=1.

All functions return a float in [0,1], higher = more similar, and never raise:
a reward function must not crash training on a malformed generation.
"""
from __future__ import annotations

import functools
import os
import re

# ── tokenisation ──────────────────────────────────────────────────────────────

# Lean identifiers may be dotted (Nat.choose) and contain unicode (ℝ, ∀).
_TOKEN_RE = re.compile(
    r"""
      (?P<ident>[A-Za-z_Ͱ-Ͽ℀-⅏][A-Za-z0-9_.'Ͱ-Ͽ℀-⅏]*)
    | (?P<num>\d+(?:\.\d+)?)
    | (?P<open>[\(\[\{])
    | (?P<close>[\)\]\}])
    | (?P<op>[^\sA-Za-z0-9_\(\)\[\]\{\}]+)
    """,
    re.VERBOSE,
)

# Binder/keyword noise that carries little discriminative signal.
_STOP = {"theorem", "lemma", "example", "by", "sorry", "fun", "have", "show"}


def tokenize(stmt: str) -> list[tuple[str, str]]:
    """(kind, text) tokens. Never raises."""
    try:
        return [(m.lastgroup, m.group()) for m in _TOKEN_RE.finditer(stmt or "")]
    except Exception:
        return []


def strip_theorem_name(stmt: str) -> str:
    """Drop `theorem <name>` so two statements are not judged different purely
    because the model invented a different (irrelevant) theorem name."""
    return re.sub(r"\b(theorem|lemma)\s+\S+", r"\1", stmt or "", count=1)


# ── GTED-style structural similarity ──────────────────────────────────────────


class _Node:
    __slots__ = ("label", "children")

    def __init__(self, label: str):
        self.label = label
        self.children = []

    @staticmethod
    def get_children(n):
        return n.children

    @staticmethod
    def get_label(n):
        return n.label

    @staticmethod
    def label_dist(a, b):
        return 0 if a == b else 1


def operator_tree(stmt: str) -> _Node:
    """Build a tree from Lean's bracket structure, with operators/identifiers as
    leaves. An approximation of GTED's operator tree (see module docstring)."""
    root = _Node("root")
    stack = [root]
    for kind, text in tokenize(strip_theorem_name(stmt)):
        if kind == "open":
            node = _Node("()")
            stack[-1].children.append(node)
            stack.append(node)
        elif kind == "close":
            if len(stack) > 1:
                stack.pop()
        else:
            if text in _STOP:
                continue
            stack[-1].children.append(_Node(text))
    return root


def _tree_size(n: _Node) -> int:
    return 1 + sum(_tree_size(c) for c in n.children)


_ZSS_WARNED = False


def _zss_missing() -> None:
    """Warn ONCE that the structural term is inert.

    This function exists because the silent version of this failure already cost
    a 200-step training run. `zss` was in no dependency list, the ImportError was
    caught by the broad `except Exception: return 0.0` below, and so
    `structural_similarity` returned 0.0 for every pair -- indistinguishable from
    "these statements are completely unrelated". The guided reward's headline
    continuous term was therefore contributing nothing, leaving library-constant
    Jaccard as the only non-binary signal, and nothing anywhere said so.
    """
    global _ZSS_WARNED
    if not _ZSS_WARNED:
        _ZSS_WARNED = True
        print(
            "[similarity] WARNING: `zss` is not installed, so GTED-style structural "
            "similarity is DISABLED and scores 0.0 for every pair. The guided reward "
            "is running on library-constant overlap alone. Install it: pip install zss",
            flush=True,
        )


def structural_similarity(pred: str, gold: str) -> float:
    """1 - normalized tree edit distance, in [0,1]."""
    try:
        try:
            from zss import simple_distance
        except ImportError:
            _zss_missing()
            return 0.0

        a, b = operator_tree(pred), operator_tree(gold)
        na, nb = _tree_size(a), _tree_size(b)
        if na <= 1 and nb <= 1:
            return 0.0
        d = simple_distance(a, b, _Node.get_children, _Node.get_label, _Node.label_dist)
        # Normalise by max(size), NOT by (na+nb). The latter is far too lenient:
        # it halves every distance, so two structurally unrelated statements
        # still scored ~0.6. Dividing by the larger tree means "replace every
        # node" scores 0, which is the behaviour a reward needs.
        return max(0.0, 1.0 - d / max(na, nb, 1))
    except Exception:
        return 0.0


# ── symbolic overlap ──────────────────────────────────────────────────────────


def library_constants(stmt: str) -> set[str]:
    """Mathlib-ish constants only: dotted (`Set.Ioo`, `Nat.choose`) or
    capitalised (`Real`, `Finset`).

    Deliberately EXCLUDES bare lowercase identifiers, which are almost always
    locally-bound variables (`x`, `a`, `hx`) and are shared by essentially every
    statement. Counting them inflated the score of unrelated predictions: the
    degenerate `theorem foo (x : ℝ) : x - 1 = -1 + x` scored 0.667 symbol
    overlap against a completely different gold purely because both mention `x`.
    A reward must not hand out credit for that.
    """
    out = set()
    for kind, text in tokenize(strip_theorem_name(stmt)):
        if kind != "ident" or text in _STOP:
            continue
        if "." in text or (text[:1].isupper()):
            out.add(text)
    return out


def symbol_similarity(pred: str, gold: str) -> float:
    """Jaccard over library constants -- 'is this about the same objects?'

    Returns 0.0 when the gold references no library constants at all, since
    there is then nothing to be right or wrong about on this axis.
    """
    try:
        a, b = library_constants(pred), library_constants(gold)
        if not b:
            return 0.0
        return len(a & b) / max(len(a | b), 1)
    except Exception:
        return 0.0


# ── optional embedding similarity ─────────────────────────────────────────────

USE_EMBEDDING = os.environ.get("BEQ_USE_EMBEDDING", "0") == "1"
EMBEDDING_MODEL = os.environ.get("BEQ_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")


@functools.lru_cache(maxsize=1)
def _embedder():
    """CPU-only, lazily loaded. Kept off the GPU deliberately: the reward loop
    already competes with vLLM and the actor for VRAM."""
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)
    mod = AutoModel.from_pretrained(EMBEDDING_MODEL).eval()
    return tok, mod


def embedding_similarity(pred: str, gold: str) -> float:
    """Cosine similarity of mean-pooled embeddings, rescaled to [0,1]."""
    if not USE_EMBEDDING:
        return 0.0
    try:
        import torch

        tok, mod = _embedder()
        batch = tok([pred or "", gold or ""], return_tensors="pt", padding=True, truncation=True, max_length=256)
        with torch.no_grad():
            out = mod(**batch).last_hidden_state
        mask = batch["attention_mask"].unsqueeze(-1).float()
        emb = (out * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        cos = torch.nn.functional.cosine_similarity(emb[0:1], emb[1:2]).item()
        return max(0.0, min(1.0, (cos + 1.0) / 2.0))
    except Exception:
        return 0.0


# ── combined ──────────────────────────────────────────────────────────────────

W_STRUCT = float(os.environ.get("BEQ_W_STRUCT", "0.6"))
W_SYMBOL = float(os.environ.get("BEQ_W_SYMBOL", "0.4"))
W_EMBED = float(os.environ.get("BEQ_W_EMBED", "0.0"))


def similarity(pred: str, gold: str) -> float:
    """Weighted blend of the available continuous signals, in [0,1]."""
    total = W_STRUCT + W_SYMBOL + (W_EMBED if USE_EMBEDDING else 0.0)
    if total <= 0:
        return 0.0
    s = W_STRUCT * structural_similarity(pred, gold) + W_SYMBOL * symbol_similarity(pred, gold)
    if USE_EMBEDDING:
        s += W_EMBED * embedding_similarity(pred, gold)
    return s / total


if __name__ == "__main__":
    gold = "theorem lean_workbook_plus_2 (x : ℝ) : x^2 - 2*x - 24 < 0 ↔ x ∈ Set.Ioo (-4) 6"
    cases = {
        "identical":        gold,
        "renamed only":     "theorem foo (x : ℝ) : x^2 - 2*x - 24 < 0 ↔ x ∈ Set.Ioo (-4) 6",
        "one coeff wrong":  "theorem foo (x : ℝ) : x^2 - 2*x - 25 < 0 ↔ x ∈ Set.Ioo (-4) 6",
        "flipped relation": "theorem foo (x : ℝ) : x^2 - 2*x - 24 > 0 ↔ x ∈ Set.Ioo (-4) 6",
        "same objects, wrong shape": "theorem foo (x : ℝ) : x ∈ Set.Ioo (-4) 6",
        "unrelated (hack)": "theorem foo (x : ℝ) : x - 1 = -1 + x",
        "garbage":          "not lean at all",
    }
    print(f"{'case':28} {'struct':>7} {'symbol':>7} {'blend':>7}")
    for name, pred in cases.items():
        print(f"{name:28} {structural_similarity(pred,gold):7.3f} "
              f"{symbol_similarity(pred,gold):7.3f} {similarity(pred,gold):7.3f}")
