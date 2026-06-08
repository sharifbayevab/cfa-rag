# -*- coding: utf-8 -*-
"""Counterfactual fragment attribution (CFA) -- the core of the method.

Given a question, a candidate fragment pool, and a frozen generator, we:
  1. generate the baseline answer with all fragments;
  2. generate the closed-book answer (no context) -- a parametric-leakage probe;
  3. for each fragment i, ablate it and regenerate; the *causal contribution* of
     fragment i is how much the answer shifts when i is removed.

The shift is measured with black-box signals that work for any generator
(answer change / token-F1 drop / semantic drift) and an optional correctness
drop against the gold answer. A fragment is "load-bearing" if its removal
changes the answer. This separates causally grounding evidence from merely
co-retrieved or cited passages.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from exp import metrics
from exp.rag import answer


@dataclass
class FragmentAttribution:
    idx: int
    is_gold: bool
    ans_ablated: str
    em_flip: float          # 1 if answer changed (normalized EM) when removed
    f1_drop: float          # 1 - F1(ans_full, ans_ablated)
    semantic_shift: float   # 1 - cos(ans_full, ans_ablated)
    correctness_drop: float # best_f1(full,gold) - best_f1(ablated,gold)
    causal_score: float     # combined load-bearing score in [0,1]


@dataclass
class SampleAttribution:
    qid: str
    question: str
    golds: list
    ans_full: str
    ans_closedbook: str
    full_correct_em: float
    full_correct_f1: float
    full_correct_cover: float       # gold string contained in answer (lenient)
    closedbook_correct_em: float
    closedbook_correct_cover: float
    n_fragments: int
    gold_idxs: list
    fragments: list = field(default_factory=list)  # list[FragmentAttribution]

    # ---- derived faithfulness quantities ----
    @property
    def causal_idxs(self):
        return [fa.idx for fa in self.fragments if fa.em_flip > 0.5]

    @property
    def any_causal(self) -> bool:
        return len(self.causal_idxs) > 0

    @property
    def parametric(self) -> bool:
        """Correct even closed-book -> the answer is not grounded in context."""
        return self.closedbook_correct_cover > 0.5

    @property
    def correct(self) -> bool:
        """Lenient correctness (gold contained in answer), standard for
        generative multi-hop QA where models give the short entity form."""
        return self.full_correct_cover > 0.5


def _causal_score(em_flip, f1_drop, semantic_shift, correctness_drop,
                  w=(0.4, 0.25, 0.2, 0.15)):
    cd = max(correctness_drop, 0.0)
    return float(w[0] * em_flip + w[1] * f1_drop + w[2] * semantic_shift + w[3] * cd)


def attribute_sample(generator, sample, *, use_semantic=True, max_tokens=48,
                     workers=8) -> SampleAttribution:
    q, golds, frags = sample["question"], sample["golds"], sample["fragments"]
    all_idxs = list(range(len(frags)))

    # baseline + closed-book + all k ablations issued concurrently
    def gen_ctx(idxs):
        return answer(generator, q, frags, idxs, max_tokens=max_tokens)

    jobs = {"full": all_idxs, "cb": []}
    for i in all_idxs:
        jobs[i] = [j for j in all_idxs if j != i]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        results = dict(zip(jobs.keys(), ex.map(gen_ctx, jobs.values())))

    ans_full, ans_cb = results["full"], results["cb"]
    sa = SampleAttribution(
        qid=sample.get("qid", ""), question=q, golds=golds,
        ans_full=ans_full, ans_closedbook=ans_cb,
        full_correct_em=metrics.best_em(ans_full, golds),
        full_correct_f1=metrics.best_f1(ans_full, golds),
        full_correct_cover=metrics.answer_contains_gold(ans_full, golds),
        closedbook_correct_em=metrics.best_em(ans_cb, golds),
        closedbook_correct_cover=metrics.answer_contains_gold(ans_cb, golds),
        n_fragments=len(frags),
        gold_idxs=[i for i, f in enumerate(frags) if f.is_gold],
    )
    base_f1_gold = sa.full_correct_f1
    for i in all_idxs:
        ans_i = results[i]
        em_flip = 1.0 - metrics.exact_match(ans_i, ans_full)
        f1_drop = 1.0 - metrics.f1_score(ans_i, ans_full)
        sem_shift = (1.0 - metrics.semantic_sim(ans_full, ans_i)) if use_semantic else f1_drop
        corr_drop = base_f1_gold - metrics.best_f1(ans_i, golds)
        sa.fragments.append(FragmentAttribution(
            idx=i, is_gold=frags[i].is_gold, ans_ablated=ans_i,
            em_flip=em_flip, f1_drop=f1_drop, semantic_shift=sem_shift,
            correctness_drop=corr_drop,
            causal_score=_causal_score(em_flip, f1_drop, sem_shift, corr_drop),
        ))
    return sa
