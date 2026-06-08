# -*- coding: utf-8 -*-
"""Answer-quality and answer-similarity metrics (SQuAD-style EM/F1) plus a
lightweight semantic-similarity helper used by the counterfactual attribution.
"""
from __future__ import annotations

import re
import string
from collections import Counter

_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCT = str.maketrans("", "", string.punctuation)


def normalize_answer(s: str) -> str:
    s = s.lower()
    s = s.translate(_PUNCT)
    s = _ARTICLES.sub(" ", s)
    return " ".join(s.split())


def exact_match(pred: str, gold: str) -> float:
    return float(normalize_answer(pred) == normalize_answer(gold))


def f1_score(pred: str, gold: str) -> float:
    p, g = normalize_answer(pred).split(), normalize_answer(gold).split()
    if not p or not g:
        return float(p == g)
    common = Counter(p) & Counter(g)
    n = sum(common.values())
    if n == 0:
        return 0.0
    prec, rec = n / len(p), n / len(g)
    return 2 * prec * rec / (prec + rec)


def best_em(pred: str, golds) -> float:
    golds = golds if isinstance(golds, (list, tuple)) else [golds]
    return max((exact_match(pred, g) for g in golds), default=0.0)


def best_f1(pred: str, golds) -> float:
    golds = golds if isinstance(golds, (list, tuple)) else [golds]
    return max((f1_score(pred, g) for g in golds), default=0.0)


def answer_contains_gold(pred: str, golds) -> float:
    """Looser correctness: gold string appears in the prediction (handles models
    that wrap the short answer in a sentence)."""
    golds = golds if isinstance(golds, (list, tuple)) else [golds]
    np_ = normalize_answer(pred)
    return float(any(normalize_answer(g) and normalize_answer(g) in np_ for g in golds))


# --- semantic similarity between two answer strings (for attribution shift) ---
_EMBEDDER = None


def _get_embedder():
    global _EMBEDDER
    if _EMBEDDER is None:
        from sentence_transformers import SentenceTransformer
        _EMBEDDER = SentenceTransformer("all-MiniLM-L6-v2")
    return _EMBEDDER


def semantic_sim(a: str, b: str) -> float:
    if not a.strip() or not b.strip():
        return float(a.strip() == b.strip())
    from sentence_transformers import util
    m = _get_embedder()
    e = m.encode([a, b], convert_to_tensor=True, normalize_embeddings=True)
    return float(util.cos_sim(e[0], e[1]).item())


def token_jaccard(a: str, b: str) -> float:
    pa, pb = set(normalize_answer(a).split()), set(normalize_answer(b).split())
    if not pa and not pb:
        return 1.0
    if not pa or not pb:
        return 0.0
    return len(pa & pb) / len(pa | pb)
