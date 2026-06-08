# -*- coding: utf-8 -*-
"""Build a fragment-level training table for the amortized causal selector.

For every (question, fragment) in the attributed cells we compute cheap features
and the CFA label (load-bearing = the answer flips when the fragment is
removed). The predictor learns to recover the expensive counterfactual causal
signal from features that cost no extra generation, enabling causal context
selection in ~2 generator calls instead of m+1.
"""
from __future__ import annotations

import glob
import json
import re
from pathlib import Path

import numpy as np

from exp.data import load_samples
from exp.run_experiment import RESULTS

_EMB = None
_CE = None


def _emb():
    global _EMB
    if _EMB is None:
        from sentence_transformers import SentenceTransformer
        _EMB = SentenceTransformer("all-MiniLM-L6-v2")
    return _EMB


def _ce():
    global _CE
    if _CE is None:
        from sentence_transformers import CrossEncoder
        _CE = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _CE


def _tok(s):
    return set(re.findall(r"[a-z0-9]+", s.lower()))


def cell_features(sample, rel_scores):
    """Return an (m, d) feature matrix for the fragments of one question."""
    q = sample["question"]
    frags = sample["fragments"]
    texts = [f"{f.title}. {f.text}" for f in frags]
    m = len(frags)

    # dense embeddings (for relevance + redundancy)
    from sentence_transformers import util
    E = _emb().encode(texts, convert_to_tensor=True, normalize_embeddings=True)
    qv = _emb().encode([q], convert_to_tensor=True, normalize_embeddings=True)
    dense = util.cos_sim(qv, E)[0].cpu().numpy()
    sim_mat = util.cos_sim(E, E).cpu().numpy()
    np.fill_diagonal(sim_mat, 0.0)
    redundancy = sim_mat.max(axis=1)         # most-similar other fragment
    mean_sim = sim_mat.mean(axis=1)

    # cross-encoder relevance
    ce = np.array(_ce().predict([(q, t) for t in texts]))

    qtok = _tok(q)
    feats = []
    rel = np.array(rel_scores) if rel_scores is not None else dense
    rel_rank = (-rel).argsort().argsort()    # 0 = most relevant
    ce_rank = (-ce).argsort().argsort()
    for i, f in enumerate(frags):
        ftok = _tok(f.text + " " + f.title)
        overlap = len(qtok & ftok) / (len(qtok) + 1e-9)
        feats.append([
            float(rel[i]),                       # dense relevance
            float(rel_rank[i]) / max(m - 1, 1),  # normalized relevance rank
            float(ce[i]),                        # cross-encoder score
            float(ce_rank[i]) / max(m - 1, 1),   # cross-encoder rank
            overlap,                             # lexical overlap with query
            len(_tok(f.text)) / 200.0,           # length (norm)
            i / max(m - 1, 1),                   # position in pool
            float(redundancy[i]),                # max similarity to another fragment
            float(mean_sim[i]),                  # mean similarity to others
        ])
    return np.array(feats, dtype=np.float32)


FEATURE_NAMES = ["rel", "rel_rank", "ce", "ce_rank", "lex_overlap", "length",
                 "position", "redundancy", "mean_sim"]


def build(out_path=RESULTS / "selector_dataset.npz"):
    X, y, ds, gen = [], [], [], []
    files = [f for f in glob.glob(str(RESULTS / "raw_*.jsonl")) if "test_" not in Path(f).name]
    for f in sorted(files):
        name = Path(f).name
        m = re.match(r"raw_(hotpotqa|2wiki|musique)_(.+)_n(\d+)\.jsonl", name)
        if not m:
            continue
        dataset, gname, n = m.group(1), m.group(2), int(m.group(3))
        rows = [json.loads(l) for l in Path(f).read_text().splitlines() if l.strip()]
        if sum(1 for r in rows if not r["ans_full"].strip()) > 0.2 * len(rows):
            print("skip corrupted", name); continue
        samples = load_samples(dataset, n=n, seed=13)
        by_qid = {s["qid"]: s for s in samples}
        print(f"featurizing {name}: {len(rows)} questions")
        for r in rows:
            s = by_qid.get(r["qid"])
            if s is None or len(s["fragments"]) != len(r["rel_scores"]):
                continue
            F = cell_features(s, r["rel_scores"])
            causal = set(r["causal_idxs"])
            for i in range(len(s["fragments"])):
                X.append(F[i]); y.append(1 if i in causal else 0)
                ds.append(dataset); gen.append(gname)
    X = np.array(X, dtype=np.float32); y = np.array(y, dtype=np.int8)
    np.savez(out_path, X=X, y=y, dataset=np.array(ds), gen=np.array(gen),
             feat_names=np.array(FEATURE_NAMES))
    print(f"saved {out_path}: X={X.shape} pos_rate={y.mean():.3f}")


if __name__ == "__main__":
    build()
