# -*- coding: utf-8 -*-
"""GO/NO-GO pilot for Paper 2 (directed-metric / Finsler retrieval).

Question: does a DIRECTED (Randers) relevance metric recover load-bearing
(causal) fragments better than a SYMMETRIC (Mahalanobis, omega=0) metric or
plain cosine?  d_F(q->f) = ||L(f-q)||_2 + omega . (f-q);  score = -d_F.

Labels = causal_idxs (load-bearing fragments) from Paper-1 results. Fragments
reconstructed via exp.data.load_samples (seed=13). Embeddings = all-MiniLM-L6-v2
(what Paper 1 used). Fully offline, no API.

Compare cosine vs omega=0 vs omega!=0 on held-out queries:
  - ranking AUC (causal vs non-causal within each query)
  - precision@k  (k = #causal for that query)
GO if omega!=0 beats omega=0 by a clear, consistent margin.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
CACHE = RES / "pilot_finsler_emb.npz"
RNG = np.random.default_rng(0)


def cell_specs():
    """(dataset, n) for each distractor cell, dedup."""
    specs = set()
    for p in glob.glob(str(RES / "raw_*.jsonl")):
        name = Path(p).name
        if "raw_oc_" in name or "test_" in name:
            continue
        # raw_<dataset>_<gen>_n<N>.jsonl ; dataset is first token after raw_
        body = name[len("raw_"):-len(".jsonl")]
        ds = body.split("_")[0]
        n = int(body.split("_n")[-1])
        specs.add((ds, n))
    return sorted(specs)


def build_dataset():
    """Return list of per-query dicts: {q_emb, frag_embs (m,d), label (m,), dataset}."""
    if CACHE.exists():
        d = np.load(CACHE, allow_pickle=True)
        return list(d["data"])
    from sentence_transformers import SentenceTransformer
    from exp.data import load_samples

    enc = SentenceTransformer("all-MiniLM-L6-v2")
    # gather unique texts first for batched encoding
    queries = {}        # (ds,n,qid) -> question
    frags = {}          # (ds,n,qid) -> [fragment texts]
    labels = {}         # (ds,n,qid) -> causal_idxs
    dsname = {}
    for ds, n in cell_specs():
        try:
            samples = {s["qid"]: s for s in load_samples(ds, n=n, seed=13)}
        except Exception as e:
            print(f"skip {ds} n{n}: {e}")
            continue
        for p in glob.glob(str(RES / f"raw_{ds}_*_n{n}.jsonl")):
            if "raw_oc_" in p or "test_" in p:
                continue
            for line in Path(p).read_text().splitlines():
                if not line.strip():
                    continue
                r = json.loads(line)
                qid = r["qid"]
                s = samples.get(qid)
                if s is None:
                    continue
                ci = set(r.get("causal_idxs") or [])
                m = len(s["fragments"])
                if not (0 < len(ci) < m):   # need both pos and neg
                    continue
                key = (ds, n, qid)
                queries[key] = s["question"]
                frags[key] = [getattr(f, "text", str(f)) for f in s["fragments"]]
                labels[key] = ci
                dsname[key] = ds
    # batch-encode
    all_texts, spans = [], {}
    for key in queries:
        start = len(all_texts)
        all_texts.append(queries[key])
        all_texts.extend(frags[key])
        spans[key] = (start, len(all_texts))
    print(f"encoding {len(all_texts)} texts over {len(queries)} queries...")
    emb = enc.encode(all_texts, batch_size=256, show_progress_bar=True,
                     normalize_embeddings=True).astype(np.float32)
    data = []
    for key, (a, b) in spans.items():
        q = emb[a]
        F = emb[a + 1:b]
        lab = np.zeros(len(F), dtype=np.float32)
        for i in labels[key]:
            if 0 <= i < len(F):
                lab[i] = 1.0
        data.append({"q": q, "F": F, "y": lab, "ds": dsname[key]})
    np.savez(CACHE, data=np.array(data, dtype=object))
    return data


# ---------- scorers ----------
def cosine_scores(q, F):
    return F @ q  # already normalized -> cosine


def auc_pk(scores, y):
    """ranking AUC (pos>neg) and precision@(#pos)."""
    pos = scores[y > 0.5]; neg = scores[y < 0.5]
    if len(pos) == 0 or len(neg) == 0:
        return None, None
    # AUC = P(score_pos > score_neg)
    auc = (pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean()
    k = int(y.sum())
    topk = np.argsort(-scores)[:k]
    pk = y[topk].mean()
    return float(auc), float(pk)


def eval_scorer(score_fn, data):
    aucs, pks = [], []
    for ex in data:
        s = score_fn(ex["q"], ex["F"])
        a, p = auc_pk(s, ex["y"])
        if a is not None:
            aucs.append(a); pks.append(p)
    return float(np.mean(aucs)), float(np.mean(pks)), len(aucs)


def train_metric(train, d, rank=32, use_omega=True, epochs=60, lr=5e-2, margin=0.2):
    """Learn G=L^T L (L: rank x d) and omega (d,) by margin ranking on d_F(q->f).
    Want d_F(q->causal) < d_F(q->noncausal)."""
    import torch
    L = torch.zeros(rank, d, requires_grad=True)
    torch.nn.init.orthogonal_(L)
    L = L.detach().clone().requires_grad_(True)
    omega = torch.zeros(d, requires_grad=True)
    params = [L, omega] if use_omega else [L]
    opt = torch.optim.Adam(params, lr=lr)
    # pre-build tensors
    exs = [(torch.tensor(e["q"]), torch.tensor(e["F"]), torch.tensor(e["y"])) for e in train]
    for ep in range(epochs):
        RNG.shuffle(exs)
        tot = 0.0
        opt.zero_grad()
        nb = 0
        for q, F, y in exs:
            diff = F - q                       # (m,d)  f - q
            md = torch.linalg.norm(diff @ L.T, dim=1)   # ||L(f-q)||
            dF = md + (diff @ omega if use_omega else 0.0)   # d_F(q->f)
            pos = dF[y > 0.5]; neg = dF[y < 0.5]
            if len(pos) == 0 or len(neg) == 0:
                continue
            # want pos (causal) SMALLER -> hinge(margin + pos - neg) over all pairs (mean)
            loss = torch.clamp(margin + pos[:, None] - neg[None, :], min=0).mean()
            loss.backward(); tot += float(loss); nb += 1
        opt.step(); opt.zero_grad()
        if ep % 20 == 0 or ep == epochs - 1:
            print(f"  ep{ep:>3} loss={tot/max(nb,1):.4f}")
    Lf = L.detach().numpy(); wf = omega.detach().numpy()

    def scorer(q, F):
        diff = F - q
        md = np.linalg.norm(diff @ Lf.T, axis=1)
        dF = md + (diff @ wf if use_omega else 0.0)
        return -dF   # higher = more relevant
    return scorer


def main():
    data = build_dataset()
    d = data[0]["q"].shape[0]
    print(f"\nqueries with usable labels: {len(data)} | emb dim: {d}")
    # split by query 80/20
    idx = np.arange(len(data)); RNG.shuffle(idx)
    cut = int(0.8 * len(idx))
    train = [data[i] for i in idx[:cut]]; test = [data[i] for i in idx[cut:]]
    print(f"train {len(train)} / test {len(test)}\n")

    rows = []
    a, p, n = eval_scorer(cosine_scores, test)
    rows.append(("cosine (baseline)", a, p))
    print(f"[cosine]  AUC={a:.3f}  P@k={p:.3f}  (n={n})\n")

    print("training SYMMETRIC (omega=0)...")
    sym = train_metric(train, d, use_omega=False)
    a, p, _ = eval_scorer(sym, test); rows.append(("symmetric Mahalanobis (omega=0)", a, p))
    print(f"[omega=0] AUC={a:.3f}  P@k={p:.3f}\n")

    print("training DIRECTED (omega!=0)...")
    dirm = train_metric(train, d, use_omega=True)
    a, p, _ = eval_scorer(dirm, test); rows.append(("directed Randers (omega!=0)", a, p))
    print(f"[omega!=0] AUC={a:.3f}  P@k={p:.3f}\n")

    print("=" * 56)
    print(f"{'scorer':<34}{'AUC':>8}{'P@k':>8}")
    for name, a, p in rows:
        print(f"{name:<34}{a:>8.3f}{p:>8.3f}")
    print("=" * 56)
    dgain = rows[2][1] - rows[1][1]
    print(f"\nGO/NO-GO: directed - symmetric AUC gain = {dgain:+.3f}")
    print("GO if clearly positive (>~0.01) and consistent; else NO-GO (pivot hook).")


if __name__ == "__main__":
    main()
