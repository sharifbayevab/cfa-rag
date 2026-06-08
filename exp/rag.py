# -*- coding: utf-8 -*-
"""Minimal RAG answer pipeline over a fixed set of candidate fragments.

The multi-hop datasets ship each question with a pool of candidate paragraphs
(a few gold + several distractors), so no retriever training is needed: context
selection operates over this provided pool, which isolates the *selection*
effect from retriever noise.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Fragment:
    idx: int
    title: str
    text: str
    is_gold: bool = False


QA_INSTRUCTION = (
    "Answer the question using the numbered context passages below. "
    "Reply with ONLY the short answer (a few words, a name, a number, or yes/no). "
    "Do not explain.\n\n"
)
CLOSED_BOOK_INSTRUCTION = (
    "Answer the question with ONLY the short answer (a few words, a name, a "
    "number, or yes/no). Do not explain.\n\n"
)


def render_context(fragments, idxs) -> str:
    lines = []
    for n, i in enumerate(idxs, 1):
        f = fragments[i]
        lines.append(f"[{n}] {f.title}: {f.text}")
    return "\n".join(lines)


def build_prompt(question: str, fragments, idxs) -> str:
    if not idxs:
        return f"{CLOSED_BOOK_INSTRUCTION}Question: {question}\nAnswer:"
    ctx = render_context(fragments, idxs)
    return f"{QA_INSTRUCTION}Context:\n{ctx}\n\nQuestion: {question}\nAnswer:"


def answer(generator, question: str, fragments, idxs=None, max_tokens: int = 48) -> str:
    if idxs is None:
        idxs = list(range(len(fragments)))
    prompt = build_prompt(question, fragments, idxs)
    return generator.generate(prompt, max_tokens=max_tokens, temperature=0.0).text


def context_tokens(fragments, idxs) -> int:
    """Whitespace-token proxy for the context length (for the efficiency axis)."""
    return sum(len(fragments[i].text.split()) + len(fragments[i].title.split()) for i in idxs)
