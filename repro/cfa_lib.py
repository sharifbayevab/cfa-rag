"""Shared helpers for the 2026-11 experiments (E1-E4): frozen inputs, rankings,
answer metrics and clustered statistics.  No generator calls."""
import json, re, string, sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
GRID_GENS = ["ollama-gemma3-12b", "ollama-gemma3-4b", "openai-gpt-4.1", "openai-gpt-4.1-mini", "openai-gpt-4.1-nano"]
SIGNED_API_TEACHERS = ["openai-gpt-4.1-nano", "openai-gpt-4o-mini", "openai-gpt-4.1-mini", "openai-gpt-4.1", "gemini-gemini-3.5-flash-lite"]
BUDGETS = [128, 256, 384, 512, 768, 1024]
ARTICLES = re.compile(r"\b(a|an|the)\b"); PUNCT = str.maketrans("", "", string.punctuation)


def normalize(t): return " ".join(ARTICLES.sub(" ", t.lower().translate(PUNCT)).split())


def token_f1(pred, gold):
    p, g = normalize(pred).split(), normalize(gold).split()
    if not p or not g: return float(p == g)
    ov = sum((Counter(p) & Counter(g)).values())
    if not ov: return 0.0
    pr, rc = ov / len(p), ov / len(g)
    return 2 * pr * rc / (pr + rc)


def metrics(pred, golds):
    n = normalize(pred)
    return {"cover": float(any(normalize(g) and normalize(g) in n for g in golds)),
            "em": max(float(normalize(pred) == normalize(g)) for g in golds), "f1": max(token_f1(pred, g) for g in golds)}


def jl(path): return [json.loads(x) for x in open(path) if x.strip()]


def load_pools(): return {r["question_group"]: r for r in jl(HERE / "inputs_e1/pools_all.jsonl")}


def load_features():
    """Per question: 9 retrieval features (identical across generators) and the grid CFA labels of each generator."""
    from selector_oof import load_aligned
    X, y, ds, gen, qid, pidx, names = load_aligned()
    feats, labels = {}, {g: {} for g in GRID_GENS}
    for g in GRID_GENS:
        m = gen == g
        for q in np.unique(qid[m]):
            t = np.where(m & (qid == q))[0]; t = t[np.argsort(pidx[t])]
            assert pidx[t].tolist() == list(range(len(t)))
            labels[g][q] = y[t].astype(float)
            if g == GRID_GENS[0]: feats[q] = X[t]
            else: assert np.allclose(feats[q][:, names.index("ce")], X[t][:, names.index("ce")], atol=1e-4)
    return feats, labels, names


def load_signed(reader):
    out = {}
    for d in ("hotpotqa", "2wiki", "musique"):
        for r in jl(HERE / f"inputs_signed/signed_{d}_{reader}_n150_r0.jsonl"):
            out[f"{d}:{r['qid']}"] = r
    return out


def rank(primary, tiebreak):
    """Indices sorted by primary desc, then tiebreak desc, then index."""
    primary = np.asarray(primary, float); tiebreak = np.asarray(tiebreak, float)
    return sorted(range(len(primary)), key=lambda i: (-round(primary[i], 9), -tiebreak[i], i))


def exact_cfa_score(full_answer, delete_answers):
    """Label-free deletion sensitivity: answer-change flag plus 1-F1 to the full answer."""
    return [float(normalize(a) != normalize(full_answer)) + (1 - token_f1(a, full_answer)) if normalize(a) != normalize(full_answer) else 0.0
            for a in delete_answers]


def budget_select(ranking, count_tokens, budget):
    """Greedy in rank order, skipping candidates that would exceed the budget; source order restored."""
    sel = []
    for i in ranking:
        if count_tokens(sorted(sel + [i])) <= budget: sel.append(i)
    if not sel: sel = [min(ranking, key=lambda i: count_tokens([i]))]
    return sorted(sel)


def holm(pvals):
    names = list(pvals); p = np.array([pvals[k] for k in names]); order = np.argsort(p)
    adj = np.empty(len(p)); run = 0
    for r, i in enumerate(order):
        run = max(run, (len(p) - r) * p[i]); adj[i] = min(1.0, run)
    return dict(zip(names, adj.tolist()))


def boot_indices(datasets, B=2000, seed=0):
    """Dataset-stratified question bootstrap index matrix (B x n)."""
    rng = np.random.default_rng(seed); datasets = np.asarray(datasets); idx = np.arange(len(datasets))
    parts = [idx[datasets == d] for d in sorted(set(datasets))]
    return np.stack([np.concatenate([rng.choice(p, len(p)) for p in parts]) for _ in range(B)])


def paired(a, b, bi):
    """Paired difference a-b (per-question arrays): mean, 95% CI, two-sided bootstrap p."""
    d = np.asarray(a, float) - np.asarray(b, float); boots = d[bi].mean(axis=1)
    p = 2 * min(np.mean(boots <= 0), np.mean(boots >= 0))
    return {"diff": float(d.mean()), "ci95": [float(x) for x in np.percentile(boots, [2.5, 97.5])], "p": float(max(p, 1 / len(bi)))}


def mean_ci(a, bi):
    a = np.asarray(a, float); boots = a[bi].mean(axis=1)
    return {"mean": float(a.mean()), "ci95": [float(x) for x in np.percentile(boots, [2.5, 97.5])]}
