# -*- coding: utf-8 -*-
"""LLM-as-judge correctness, with an on-disk cache.

Standard automatic metrics (EM/F1/cover) under-credit generative answers that
give a correct entity in a different surface form (e.g. "CMS" vs "Centers for
Medicare and Medicaid Services"). A cheap LLM judge gives the headline
correctness (ACC_LLM) used across recent RAG papers. Per-fragment causal shifts
remain automatic (answer-vs-answer), so the judge is only for final answers.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from exp.generators import OpenAIGenerator

CACHE = Path(__file__).resolve().parent.parent / "results" / "judge_cache.json"
_cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
_judge = None


def _get_judge(model="gpt-4.1-nano"):
    global _judge
    if _judge is None:
        _judge = OpenAIGenerator(model=model, logprobs=False)
    return _judge


def _key(q, gold, pred):
    return hashlib.md5(f"{q}|||{gold}|||{pred}".encode()).hexdigest()


def judge_correct(question, golds, pred, model="gpt-4.1-nano") -> float:
    golds = golds if isinstance(golds, (list, tuple)) else [golds]
    gold = golds[0]
    if not pred.strip():
        return 0.0
    k = _key(question, gold, pred)
    if k in _cache:
        return _cache[k]
    prompt = (
        "You are grading a question-answering system. Decide if the PREDICTION "
        "is correct given the reference answer. Minor paraphrases, abbreviations, "
        "or extra words are fine as long as the core answer matches. "
        "Reply with exactly 'yes' or 'no'.\n\n"
        f"Question: {question}\nReference answer: {gold}\nPrediction: {pred}\n\nCorrect?"
    )
    out = _get_judge(model).generate(prompt, max_tokens=3, temperature=0.0).text.strip().lower()
    val = 1.0 if out.startswith("y") else 0.0
    _cache[k] = val
    return val


def flush_cache():
    CACHE.write_text(json.dumps(_cache), encoding="utf-8")
