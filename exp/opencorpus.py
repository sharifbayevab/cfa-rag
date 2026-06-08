# -*- coding: utf-8 -*-
"""Open-corpus-lite RAG setting.

Instead of using each question's curated distractor pool, we pool the paragraphs
of MANY questions into a single corpus, index it with a hybrid BM25 + dense
retriever, and retrieve the top-k candidates for each question *from the whole
corpus*. This introduces real retrieval noise (the retriever can miss the gold
passage or pull cross-question passages), so we can test whether the
distractor-pool findings survive an end-to-end retrieve-then-read pipeline.

Gold supporting titles are known from the dataset, so we still measure retrieval
recall and causal-vs-gold. We then run the standard CFA pipeline over the
retrieved candidates.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np

from exp.data import load_samples
from exp.rag import Fragment
from exp.run_experiment import RESULTS, run_cell, summarize

_EMB = None


def _emb():
    global _EMB
    if _EMB is None:
        from sentence_transformers import SentenceTransformer
        _EMB = SentenceTransformer("all-MiniLM-L6-v2")
    return _EMB


def _tok(s):
    return re.findall(r"[a-z0-9]+", s.lower())


def _norm(x):
    x = np.asarray(x, dtype=float)
    lo, hi = x.min(), x.max()
    return (x - lo) / (hi - lo + 1e-9)


class HybridRetriever:
    def __init__(self, passages, alpha=0.5):
        from rank_bm25 import BM25Okapi
        self.passages = passages                      # list[Fragment]
        self.texts = [f"{p.title}. {p.text}" for p in passages]
        self.bm25 = BM25Okapi([_tok(t) for t in self.texts])
        self.emb = _emb().encode(self.texts, convert_to_tensor=True,
                                 normalize_embeddings=True, batch_size=128,
                                 show_progress_bar=False)
        self.alpha = alpha

    def retrieve(self, query, k=10):
        from sentence_transformers import util
        bm = np.array(self.bm25.get_scores(_tok(query)))
        qv = _emb().encode([query], convert_to_tensor=True, normalize_embeddings=True)
        dense = util.cos_sim(qv, self.emb)[0].cpu().numpy()
        score = self.alpha * _norm(dense) + (1 - self.alpha) * _norm(bm)
        top = np.argsort(-score)[:k]
        return [self.passages[i] for i in top]


def build_corpus(dataset, pool_n, seed):
    """Union of unique paragraphs over pool_n questions."""
    pool = load_samples(dataset, n=pool_n, seed=seed)
    seen, passages = set(), []
    for s in pool:
        for f in s["fragments"]:
            key = (f.title, f.text[:120])
            if key in seen:
                continue
            seen.add(key)
            passages.append(Fragment(idx=len(passages), title=f.title, text=f.text, is_gold=False))
    return pool, passages


def make_opencorpus_samples(dataset, pool_n, eval_n, seed, k):
    pool, passages = build_corpus(dataset, pool_n, seed)
    retr = HybridRetriever(passages)
    eval_samples = pool[:eval_n]
    out, recalls = [], []
    for s in eval_samples:
        gold_titles = {f.title for f in s["fragments"] if f.is_gold}
        retrieved = retr.retrieve(s["question"], k=k)
        frags = [Fragment(idx=j, title=p.title, text=p.text, is_gold=(p.title in gold_titles))
                 for j, p in enumerate(retrieved)]
        got = len({f.title for f in frags if f.is_gold})
        recalls.append(got / max(1, len(gold_titles)))
        out.append({"qid": s["qid"], "question": s["question"], "golds": s["golds"],
                    "fragments": frags, "dataset": dataset})
    return out, float(np.mean(recalls)), len(passages)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="hotpotqa")
    ap.add_argument("--generator", default="openai:gpt-4.1-mini")
    ap.add_argument("--pool-n", type=int, default=800)   # questions whose paragraphs form the corpus
    ap.add_argument("--eval-n", type=int, default=150)   # questions evaluated
    ap.add_argument("--k", type=int, default=10)         # retrieved candidates per question
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    print(f"OPEN-CORPUS: {args.dataset} | {args.generator} | corpus from {args.pool_n} q, "
          f"eval {args.eval_n}, k={args.k}", flush=True)
    t0 = time.time()
    samples, recall, corpus_size = make_opencorpus_samples(
        args.dataset, args.pool_n, args.eval_n, args.seed, args.k)
    print(f"corpus={corpus_size} passages | retrieval recall@{args.k}={recall:.3f} "
          f"| built in {time.time()-t0:.0f}s", flush=True)

    rows = run_cell(args.dataset, args.generator, len(samples), args.seed, k=4,
                    use_judge=False, use_semantic=True, samples=samples)
    summ = summarize(rows)
    summ["retrieval_recall_at_k"] = round(recall, 3)
    summ["corpus_size"] = corpus_size
    stem = f"oc_{args.dataset}_{args.generator.replace(':','-')}_n{len(samples)}"
    (RESULTS / f"raw_{stem}.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    (RESULTS / f"summary_{stem}.json").write_text(json.dumps(summ, indent=2), encoding="utf-8")
    print("\n=== OPEN-CORPUS SUMMARY ===")
    print(json.dumps({k: summ[k] for k in ["retrieval_recall_at_k", "corpus_size",
          "acc_full_judge", "acc_closedbook_judge", "faithfulness_gap",
          "relevance_vs_causality", "causal_vs_gold"]}, indent=2))
    print("saved", stem)


if __name__ == "__main__":
    main()
