# -*- coding: utf-8 -*-
"""GO/NO-GO proxy test for Paper 2 framing: does a CAUSALITY-ALIGNED metric,
used as the RETRIEVER over the open-corpus, surface more gold/load-bearing
passages @k than plain cosine?  (Paper 1 never changed the retriever.)

Train: symmetric Mahalanobis metric d(q,p)=||L(p-q)|| on DISTRACTOR cells'
causal_idxs labels (reuse exp.pilot_finsler). Transfer: rank the FULL rebuilt
open-corpus by the metric, measure gold-passage recall@k vs cosine. Offline,
no generation/API.

GO if causality-metric recall@k clearly > cosine -> distinct Paper 2.
Modest/positive -> fold into Paper 1.  No gain -> drop the metric.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np

from exp.data import load_samples
from exp.opencorpus import build_corpus
from exp.pilot_finsler import build_dataset, train_metric

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"

OC_FILES = {
    "hotpotqa": "raw_oc_hotpotqa_openai-gpt-4.1-mini_n150.jsonl",
    "2wiki": "raw_oc_2wiki_openai-gpt-4.1-mini_n150.jsonl",
    "musique": "raw_oc_musique_openai-gpt-4.1-mini_n150.jsonl",
}
POOL_N, SEED = 1000, 13
KS = [5, 10, 20, 50]


def emb_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")


def recall_at_k(order, gold_pos, ks):
    """order: passage indices sorted best-first; gold_pos: set of gold indices."""
    if not gold_pos:
        return None
    out = {}
    for k in ks:
        hit = len(gold_pos & set(order[:k]))
        out[k] = hit / len(gold_pos)
    return out


def run_dataset(ds, Lf, enc):
    pool, passages = build_corpus(ds, POOL_N, SEED)
    by_qid = {s["qid"]: s for s in pool}
    # title -> list of corpus passage indices
    title2idx = {}
    for j, p in enumerate(passages):
        title2idx.setdefault(p.title, []).append(j)
    texts = [f"{p.title}. {p.text}" for p in passages]
    print(f"  [{ds}] embedding {len(texts)} corpus passages...")
    P = enc.encode(texts, batch_size=256, normalize_embeddings=True,
                   show_progress_bar=False).astype(np.float32)

    # eval queries from stored open-corpus run
    rows = [json.loads(l) for l in (RES / OC_FILES[ds]).read_text().splitlines() if l.strip()]
    qids = [r["qid"] for r in rows if r["qid"] in by_qid]
    qtexts = [by_qid[q]["question"] for q in qids]
    Q = enc.encode(qtexts, batch_size=256, normalize_embeddings=True,
                   show_progress_bar=False).astype(np.float32)

    cos_rec = {k: [] for k in KS}
    met_rec = {k: [] for k in KS}
    for qi, qid in enumerate(qids):
        s = by_qid[qid]
        gold_titles = {f.title for f in s["fragments"] if getattr(f, "is_gold", False)}
        gold_pos = set()
        for t in gold_titles:
            gold_pos.update(title2idx.get(t, []))
        if not gold_pos:
            continue
        q = Q[qi]
        # cosine (dense)
        cos = P @ q
        order_cos = np.argsort(-cos)
        # causality metric: -||L(p-q)||
        diff = P - q
        md = np.linalg.norm(diff @ Lf.T, axis=1)
        order_met = np.argsort(md)   # smaller distance = better
        rc = recall_at_k(order_cos, gold_pos, KS)
        rm = recall_at_k(order_met, gold_pos, KS)
        for k in KS:
            cos_rec[k].append(rc[k]); met_rec[k].append(rm[k])
    n = len(cos_rec[KS[0]])
    return {k: (np.mean(cos_rec[k]), np.mean(met_rec[k])) for k in KS}, n


def main():
    # train causality metric on ALL distractor data (symmetric, omega=0)
    print("training causality-aligned metric on distractor causal labels...")
    data = build_dataset()
    d = data[0]["q"].shape[0]
    scorer = train_metric(data, d, use_omega=False, epochs=60)
    # recover L from the closure (re-train returning L): retrain capturing L
    # train_metric returns a closure; refit here to grab L explicitly
    import torch
    L = torch.zeros(32, d); torch.nn.init.orthogonal_(L); L = L.detach().clone().requires_grad_(True)
    opt = torch.optim.Adam([L], lr=5e-2)
    exs = [(torch.tensor(e["q"]), torch.tensor(e["F"]), torch.tensor(e["y"])) for e in data]
    rng = np.random.default_rng(0)
    for ep in range(60):
        rng.shuffle(exs); opt.zero_grad(); tot = nb = 0
        for q, F, y in exs:
            diff = F - q
            dF = torch.linalg.norm(diff @ L.T, dim=1)
            pos = dF[y > 0.5]; neg = dF[y < 0.5]
            if len(pos) == 0 or len(neg) == 0:
                continue
            loss = torch.clamp(0.2 + pos[:, None] - neg[None, :], min=0).mean()
            loss.backward(); tot += float(loss.detach()); nb += 1
        opt.step()
    Lf = L.detach().numpy()
    print(f"metric trained (L: {Lf.shape}).\n")

    enc = emb_model()
    print(f"{'dataset':<10}{'k':>5}{'cosine':>10}{'causal-metric':>16}{'Δ':>9}")
    grand = {k: [] for k in KS}
    for ds in ["hotpotqa", "2wiki", "musique"]:
        res, n = run_dataset(ds, Lf, enc)
        for k in KS:
            c, m = res[k]
            grand[k].append((c, m))
            print(f"{ds:<10}{k:>5}{c:>10.3f}{m:>16.3f}{m-c:>+9.3f}")
        print(f"  (n={n} eval queries)\n")
    print("=" * 50)
    print(f"{'MEAN':<10}{'k':>5}{'cosine':>10}{'causal-metric':>16}{'Δ':>9}")
    for k in KS:
        c = np.mean([x[0] for x in grand[k]]); m = np.mean([x[1] for x in grand[k]])
        print(f"{'':<10}{k:>5}{c:>10.3f}{m:>16.3f}{m-c:>+9.3f}")
    print("=" * 50)
    print("\nGO if causal-metric recall clearly > cosine at small k; "
          "modest -> fold into Paper 1; none -> drop.")


if __name__ == "__main__":
    main()
