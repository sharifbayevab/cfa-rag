"""Round-2 helpers (E6-E9): answer-containment features and shared ranking definitions.

Containment (reviewer's `anscont`): the normalized answer string, padded with
spaces, is a substring of the normalized passage (title + text), using
cfa_lib.normalize.  Variants differ only in whose answer is used:
  own    - the target reader's own full-context answer (1 reader call)
  gemma  - the full-context answer of Gemma-3-4B from the main grid (1 cheap call)
  gold   - any gold answer alias (oracle diagnostic, not deployable)
  overlap- share of the own answer's tokens found in the passage (soft variant)
"""
import sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from cfa_lib import jl, normalize, rank

READERS6 = ["openai-gpt-4.1-nano", "openai-gpt-4o-mini", "openai-gpt-4.1-mini", "openai-gpt-4.1", "gemini-gemini-3.5-flash-lite", "ollama-qwen3-4b-instruct"]
NICE = {"openai-gpt-4.1-nano": "GPT-4.1-nano", "openai-gpt-4o-mini": "GPT-4o-mini", "openai-gpt-4.1-mini": "GPT-4.1-mini", "openai-gpt-4.1": "GPT-4.1",
        "gemini-gemini-3.5-flash-lite": "Gemini-3.5-flash-lite", "ollama-qwen3-4b-instruct": "Qwen3-4B (Ollama)", "hf-qwen3-4b": "Qwen3-4B (HF)"}
FAMILY = lambda r: r.split("-")[0]


def passage_norm(f): return " " + normalize(f["title"] + " " + f["text"]) + " "


def contains(answer, fragments):
    a = normalize(answer)
    if not a: return np.zeros(len(fragments))
    return np.array([float(f" {a} " in passage_norm(f)) for f in fragments])


def contains_any(answers, fragments):
    return np.max([contains(a, fragments) for a in answers], axis=0) if answers else np.zeros(len(fragments))


def overlap(answer, fragments):
    toks = normalize(answer).split()
    if not toks: return np.zeros(len(fragments))
    return np.array([np.mean([t in set(passage_norm(f).split()) for t in toks]) for f in fragments])


def load_gemma4b_answers():
    out = {}
    for d in ("hotpotqa", "2wiki", "musique"):
        for r in jl(HERE / f"inputs/raw_{d}_ollama-gemma3-4b_n150.jsonl"): out[f"{d}:{r['qid']}"] = r["ans_full"]
    return out


def load_grid_answers():
    """Full-context answers of the five generators of the main grid (donors of the selection arms)."""
    from cfa_lib import GRID_GENS
    out = {g: {} for g in GRID_GENS}
    for g in GRID_GENS:
        for d in ("hotpotqa", "2wiki", "musique"):
            for r in jl(HERE / f"inputs/raw_{d}_{g}_n150.jsonl"): out[g][f"{d}:{r['qid']}"] = r["ans_full"]
    return out


def e6_rankings(ac, ac_gemma, ac_gold, ov, tr, exact, bge, ce):
    """New round-2 rankings plus the three round-1 reference arms (identical definitions)."""
    fl = (np.asarray(exact) > 0).astype(float)
    return {
        "bge": rank(bge, ce), "cfa_transfer_bge": rank(tr, bge), "cfa_exact_bge": rank(exact, bge),
        "anscont_bge": rank(ac, bge), "anscont_gemma4b_bge": rank(ac_gemma, bge), "ansoverlap_bge": rank(ov, bge),
        "anscont_transfer_bge": rank(ac * 10 + tr, bge), "transfer_anscont_bge": rank(tr * 10 + ac, bge), "sum_anscont_transfer_bge": rank(ac + tr, bge),
        "anscont_exact_bge": rank(ac * 10 + exact, bge), "sum_anscont_exact_bge": rank(ac + fl, bge),
        "goldans_oracle_bge": rank(ac_gold, bge),
    }
