# -*- coding: utf-8 -*-
"""Unified context-selection methods with explicit cost accounting.

Each method returns (selected_idxs, probe_calls), where probe_calls is the
number of *extra* generator calls the method spends to decide the selection
(beyond the single final answer generation). This puts every method on a common
accuracy-vs-cost axis:

  full / random / top-k relevance        -- 0 probe calls
  amortized causal (ours, learned)       -- 0 generator probe calls (cheap features + model)
  ContextCite-style surrogate            -- n_masks probe calls
  CFA pruning (ours, exact)              -- m+1 probe calls (leave-one-out)
  oracle gold                            -- 0 (uses labels; ceiling)
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from exp import metrics
from exp.rag import answer
from exp.selection import relevance_scores, select_topk

RES = Path(__file__).resolve().parent.parent / "results"
_MODEL = None


def _model():
    global _MODEL
    if _MODEL is None:
        _MODEL = pickle.load(open(RES / "causal_selector.pkl", "rb"))
    return _MODEL


def m_full(sample, **kw):
    return list(range(len(sample["fragments"]))), 0


def m_random_k(sample, k=4, rng=None, **kw):
    rng = rng or np.random.default_rng(0)
    n = len(sample["fragments"])
    return sorted(rng.choice(n, size=min(k, n), replace=False).tolist()), 0


def m_topk_relevance(sample, k=4, **kw):
    return select_topk(sample["_rel"], min(k, len(sample["fragments"]))), 0


def m_oracle(sample, **kw):
    g = [i for i, f in enumerate(sample["fragments"]) if f.is_gold]
    return (g or list(range(len(sample["fragments"])))), 0


def m_amortized(sample, thresh=0.5, min_keep=1, **kw):
    """Learned causal selector: predict load-bearing prob from cheap features."""
    from exp.selector_data import cell_features
    F = cell_features(sample, sample["_rel"])
    prob = _model().predict_proba(F)[:, 1]
    keep = [i for i in range(len(prob)) if prob[i] >= thresh]
    if len(keep) < min_keep:
        keep = (-prob).argsort()[:min_keep].tolist()
    return sorted(keep), 0


def m_contextcite(generator, sample, n_masks=16, k=2, rng=None, max_tokens=48, **kw):
    """ContextCite-style: fit a linear surrogate over random ablation masks
    predicting answer agreement with the full-context answer; attribute by
    coefficients; keep the top-k attributed fragments."""
    rng = rng or np.random.default_rng(0)
    frags, q = sample["fragments"], sample["question"]
    n = len(frags)
    ans_full = answer(generator, q, frags, list(range(n)), max_tokens=max_tokens)
    masks, scores = [], []
    for _ in range(n_masks):
        mask = rng.random(n) < 0.5
        if not mask.any():
            mask[rng.integers(n)] = True
        idxs = sorted(np.where(mask)[0].tolist())
        a = answer(generator, q, frags, idxs, max_tokens=max_tokens)
        masks.append(mask.astype(float))
        scores.append(metrics.f1_score(a, ans_full))   # agreement with full answer
    from sklearn.linear_model import Ridge
    coef = Ridge(alpha=1.0).fit(np.array(masks), np.array(scores)).coef_
    top = sorted((-coef).argsort()[:k].tolist())
    return top, n_masks + 1   # +1 for the full-context probe


def m_cfa_prune(generator, sample, tau=0.5, max_tokens=48, **kw):
    """Exact counterfactual pruning: leave-one-out, keep fragments whose removal
    flips the answer."""
    frags, q = sample["fragments"], sample["question"]
    n = len(frags)
    ans_full = answer(generator, q, frags, list(range(n)), max_tokens=max_tokens)
    causal = []
    for i in range(n):
        a = answer(generator, q, frags, [j for j in range(n) if j != i], max_tokens=max_tokens)
        if (1.0 - metrics.exact_match(a, ans_full)) > tau:
            causal.append(i)
    if not causal:
        causal = select_topk(sample["_rel"], 1)
    return sorted(causal), n + 1


METHODS = {
    "full": m_full,
    "random_k": m_random_k,
    "topk_relevance": m_topk_relevance,
    "amortized_causal": m_amortized,
    "contextcite": m_contextcite,
    "cfa_prune": m_cfa_prune,
    "oracle_gold": m_oracle,
}
NEEDS_GEN = {"contextcite", "cfa_prune"}   # methods that call the generator to probe
