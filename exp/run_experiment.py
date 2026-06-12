# -*- coding: utf-8 -*-
"""Full experiment runner for one (dataset, generator) cell.

Per sample: relevance scores -> counterfactual attribution -> selection
strategies -> LLM-judge. Aggregates the paper's headline quantities:
  (1) correctness vs faithfulness gap (parametric leakage, no-causal-fragment),
  (2) relevance != causality (overlap of top-relevance vs causal sets),
  (3) causal-vs-gold alignment,
  (4) selection comparison (accuracy / tokens / gold recall per strategy).
Saves per-sample JSONL + a summary JSON.
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from exp.attribution import attribute_sample
from exp.data import load_samples
from exp.generators import make_generator
from exp.judge import flush_cache, judge_correct
from exp.selection import evaluate_selection, relevance_scores, select_indices

RESULTS = Path(__file__).resolve().parent.parent / "results"
RESULTS.mkdir(exist_ok=True)
STRATEGIES = ["full", "topk_relevance", "random_k", "oracle_gold", "causal_prune"]


def run_cell(dataset, generator_spec, n, seed, k, use_judge, use_semantic, samples=None,
             checkpoint_path=None):
    gen = make_generator(generator_spec)
    if samples is None:
        samples = load_samples(dataset, n=n, seed=seed)
    rng = np.random.default_rng(seed)
    rows, done = [], set()
    cp = Path(checkpoint_path) if checkpoint_path else None
    if cp and cp.exists():                       # resume: skip already-done samples
        for line in cp.read_text().splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue                          # drop a truncated trailing line from a crash
            rows.append(r); done.add(r.get("qid"))
        if rows:
            print(f"  resume: {len(rows)}/{len(samples)} samples already done; continuing", flush=True)
    fh = open(cp, "a", encoding="utf-8") if cp else None
    t0 = time.time(); processed = 0
    for ci, s in enumerate(samples):
        if s.get("qid") in done:
            continue
        s["_rel"] = relevance_scores(s["question"], s["fragments"])
        sa = attribute_sample(gen, s, use_semantic=use_semantic)

        # judge correctness of full + closed-book
        jfull = judge_correct(s["question"], s["golds"], sa.ans_full) if use_judge else sa.full_correct_cover
        jcb = judge_correct(s["question"], s["golds"], sa.ans_closedbook) if use_judge else sa.closedbook_correct_cover

        causal = set(sa.causal_idxs)
        gold = set(sa.gold_idxs)
        toprel = set(select_indices("topk_relevance", s, k=max(len(gold), 2)).copy() if False else
                     sorted(np.argsort(-s["_rel"])[:max(len(gold), 2)].tolist()))
        causal_scores = np.array([fa.causal_score for fa in sa.fragments])

        # evaluate the 5 selection strategies' generations in parallel
        strat_idxs = {st: select_indices(st, s, attribution=sa, k=k, rng=rng) for st in STRATEGIES}
        with ThreadPoolExecutor(max_workers=len(STRATEGIES)) as ex:
            evald = dict(zip(strat_idxs.keys(),
                             ex.map(lambda it: evaluate_selection(gen, s, it), strat_idxs.values())))
        sel = {}
        for strat, r in evald.items():
            r["judge"] = (judge_correct(s["question"], s["golds"], r["answer"])
                          if use_judge else r["cover"])
            sel[strat] = r

        row = {
            "qid": sa.qid, "question": s["question"], "golds": s["golds"],
            "ans_full": sa.ans_full, "ans_cb": sa.ans_closedbook,
            "judge_full": jfull, "judge_cb": jcb,
            "cover_full": sa.full_correct_cover, "f1_full": sa.full_correct_f1,
            "gold_idxs": sorted(gold), "causal_idxs": sorted(causal),
            "toprel_idxs": sorted(toprel),
            "rel_scores": [round(float(x), 4) for x in s["_rel"]],
            "causal_scores": [round(float(x), 4) for x in causal_scores],
            "rel_causal_spearman": (float(spearmanr(s["_rel"], causal_scores).correlation)
                                    if len(causal_scores) > 2 else None),
            "selection": sel,
        }
        rows.append(row)
        if fh:                                   # checkpoint every sample (crash-safe)
            fh.write(json.dumps(row, ensure_ascii=False) + "\n"); fh.flush()
        processed += 1
        if processed % 10 == 0:
            print(f"  {len(rows)}/{len(samples)} ({(time.time()-t0)/processed:.1f}s/sample)", flush=True)
            if use_judge:
                flush_cache()
    if fh:
        fh.close()
    if use_judge:
        flush_cache()
    return rows


def summarize(rows):
    n = len(rows)
    corr = [r for r in rows if r["judge_full"] > 0.5]
    nc = len(corr) or 1

    # (1) faithfulness gap among correct
    parametric = sum(1 for r in corr if r["judge_cb"] > 0.5) / nc
    no_causal = sum(1 for r in corr if not r["causal_idxs"]) / nc

    # (2) relevance != causality
    def jacc(a, b):
        a, b = set(a), set(b)
        return len(a & b) / len(a | b) if (a or b) else 1.0
    rel_causal_jacc = np.mean([jacc(r["toprel_idxs"], r["causal_idxs"]) for r in corr]) if corr else 0
    sps = [r["rel_causal_spearman"] for r in rows if r["rel_causal_spearman"] is not None]
    rel_causal_rho = float(np.nanmean(sps)) if sps else None

    # (3) causal vs gold
    def prec(a, b):
        a, b = set(a), set(b)
        return len(a & b) / len(a) if a else 0.0
    def rec(a, b):
        a, b = set(a), set(b)
        return len(a & b) / len(b) if b else 0.0
    cg_prec = np.mean([prec(r["causal_idxs"], r["gold_idxs"]) for r in corr]) if corr else 0
    cg_rec = np.mean([rec(r["causal_idxs"], r["gold_idxs"]) for r in corr]) if corr else 0

    # (4) selection comparison
    sel = {}
    for strat in STRATEGIES:
        rs = [r["selection"][strat] for r in rows]
        sel[strat] = {
            "judge_acc": round(np.mean([x["judge"] for x in rs]), 3),
            "EM": round(np.mean([x["em"] for x in rs]), 3),
            "F1": round(np.mean([x["f1"] for x in rs]), 3),
            "cover": round(np.mean([x["cover"] for x in rs]), 3),
            "avg_frags": round(np.mean([x["n_frags"] for x in rs]), 2),
            "avg_tokens": round(np.mean([x["n_tokens"] for x in rs]), 1),
            "gold_recall": round(np.mean([x["gold_recall"] for x in rs]), 3),
        }
    return {
        "n": n,
        "acc_full_judge": round(np.mean([r["judge_full"] for r in rows]), 3),
        "acc_closedbook_judge": round(np.mean([r["judge_cb"] for r in rows]), 3),
        "n_correct": len(corr),
        "faithfulness_gap": {
            "frac_correct_parametric": round(parametric, 3),
            "frac_correct_no_causal_fragment": round(no_causal, 3),
        },
        "relevance_vs_causality": {
            "top_rel_causal_jaccard": round(float(rel_causal_jacc), 3),
            "rel_causal_spearman": round(rel_causal_rho, 3) if rel_causal_rho is not None else None,
        },
        "causal_vs_gold": {"precision": round(float(cg_prec), 3), "recall": round(float(cg_rec), 3)},
        "selection": sel,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="hotpotqa")
    ap.add_argument("--generator", default="openai:gpt-4.1-mini")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--no-judge", action="store_true")
    ap.add_argument("--no-semantic", action="store_true")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    print(f"RUN: {args.dataset} | {args.generator} | n={args.n} | judge={not args.no_judge}", flush=True)
    tag = (args.tag + "_") if args.tag else ""
    stem = f"{tag}{args.dataset}_{args.generator.replace(':','-')}_n{args.n}"
    raw_path = RESULTS / f"raw_{stem}.jsonl"
    rows = run_cell(args.dataset, args.generator, args.n, args.seed, args.k,
                    use_judge=not args.no_judge, use_semantic=not args.no_semantic,
                    checkpoint_path=raw_path)
    summ = summarize(rows)
    raw_path.write_text(                          # canonical rewrite (ordered, deduped)
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    (RESULTS / f"summary_{stem}.json").write_text(json.dumps(summ, indent=2), encoding="utf-8")
    print("\n=== SUMMARY ===")
    print(json.dumps(summ, indent=2))
    print("saved", RESULTS / f"summary_{stem}.json")


if __name__ == "__main__":
    main()
