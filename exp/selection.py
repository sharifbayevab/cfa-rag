# -*- coding: utf-8 -*-
"""Context-selection strategies evaluated against the causal-attribution method.

Given a candidate fragment pool, each strategy chooses a subset to feed the
generator. We compare:
  * full            -- keep all candidates (standard "stuff everything" RAG)
  * topk_relevance  -- top-k by dense similarity to the question (standard RAG)
  * random_k        -- random k (control)
  * oracle_gold     -- only the gold supporting fragments (quality ceiling)
  * causal_prune    -- OURS: keep only causally load-bearing fragments found by
                       counterfactual attribution (falls back to top-1 relevance
                       if no fragment is causal)

The causal_prune strategy needs an attribution pass first; this lets us study
the relevance-vs-causality gap and whether pruning to causal evidence improves
faithfulness and accuracy while cutting context tokens.
"""
from __future__ import annotations

import numpy as np

from exp import metrics
from exp.rag import answer, context_tokens

_EMB = None


def _embedder():
    global _EMB
    if _EMB is None:
        from sentence_transformers import SentenceTransformer
        _EMB = SentenceTransformer("all-MiniLM-L6-v2")
    return _EMB


def relevance_scores(question, fragments):
    m = _embedder()
    texts = [f"{f.title}. {f.text}" for f in fragments]
    q = m.encode([question], convert_to_tensor=True, normalize_embeddings=True)
    e = m.encode(texts, convert_to_tensor=True, normalize_embeddings=True)
    from sentence_transformers import util
    return util.cos_sim(q, e)[0].cpu().numpy()


def select_topk(scores, k):
    return sorted(np.argsort(-scores)[:k].tolist())


def select_indices(strategy, sample, attribution=None, *, k=4, rng=None,
                   causal_threshold=0.5):
    frags = sample["fragments"]
    n = len(frags)
    alli = list(range(n))
    if strategy == "full":
        return alli
    if strategy == "oracle_gold":
        g = [i for i, f in enumerate(frags) if f.is_gold]
        return g or alli
    if strategy == "random_k":
        rng = rng or np.random.default_rng(0)
        return sorted(rng.choice(alli, size=min(k, n), replace=False).tolist())
    if strategy == "topk_relevance":
        return select_topk(sample["_rel"], min(k, n))
    if strategy == "causal_prune":
        assert attribution is not None
        causal = [fa.idx for fa in attribution.fragments if fa.em_flip > causal_threshold]
        if not causal:                       # fall back to top-1 relevance
            return select_topk(sample["_rel"], 1)
        return sorted(causal)
    raise ValueError(strategy)


def evaluate_selection(generator, sample, idxs, max_tokens=48):
    frags, golds, q = sample["fragments"], sample["golds"], sample["question"]
    ans = answer(generator, q, frags, idxs, max_tokens=max_tokens)
    return {
        "idxs": idxs,
        "answer": ans,
        "em": metrics.best_em(ans, golds),
        "f1": metrics.best_f1(ans, golds),
        "cover": metrics.answer_contains_gold(ans, golds),
        "n_frags": len(idxs),
        "n_tokens": context_tokens(frags, idxs),
        "gold_recall": (len(set(idxs) & {i for i, f in enumerate(frags) if f.is_gold}) /
                        max(1, sum(f.is_gold for f in frags))),
    }
