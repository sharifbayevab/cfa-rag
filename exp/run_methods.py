# -*- coding: utf-8 -*-
"""Evaluate all selection methods on a (dataset, generator) cell on a common
accuracy-vs-cost axis. Saves a per-method summary.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from exp.data import load_samples
from exp.generators import make_generator
from exp.methods import METHODS, NEEDS_GEN
from exp.rag import answer, context_tokens
from exp.selection import relevance_scores
from exp import metrics

RES = Path(__file__).resolve().parent.parent / "results"


def run(dataset, generator_spec, n, seed, k=4):
    gen = make_generator(generator_spec)
    samples = load_samples(dataset, n=n, seed=seed)
    rng = np.random.default_rng(seed)
    agg = {m: {"cover": [], "f1": [], "tokens": [], "probe": [], "gold_recall": []}
           for m in METHODS}
    for ci, s in enumerate(samples):
        s["_rel"] = relevance_scores(s["question"], s["fragments"])
        gold = {i for i, f in enumerate(s["fragments"]) if f.is_gold}
        n_gold = max(1, len(gold))
        for mname, fn in METHODS.items():
            kw = {"k": k, "rng": rng}
            if mname in NEEDS_GEN:
                idxs, probe = fn(gen, s, **kw)
            else:
                idxs, probe = fn(s, **kw)
            ans = answer(gen, s["question"], s["fragments"], idxs)
            agg[mname]["cover"].append(metrics.answer_contains_gold(ans, s["golds"]))
            agg[mname]["f1"].append(metrics.best_f1(ans, s["golds"]))
            agg[mname]["tokens"].append(context_tokens(s["fragments"], idxs))
            agg[mname]["probe"].append(probe)
            agg[mname]["gold_recall"].append(len(set(idxs) & gold) / n_gold)
        if (ci + 1) % 10 == 0:
            print(f"  {ci+1}/{len(samples)}", flush=True)
    summ = {m: {"acc": round(float(np.mean(v["cover"])), 3),
                "f1": round(float(np.mean(v["f1"])), 3),
                "tokens": round(float(np.mean(v["tokens"])), 1),
                "probe_calls": round(float(np.mean(v["probe"])), 1),
                "gold_recall": round(float(np.mean(v["gold_recall"])), 3)}
            for m, v in agg.items()}
    return {"dataset": dataset, "generator": generator_spec, "n": len(samples), "methods": summ}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="hotpotqa")
    ap.add_argument("--generator", default="ollama:gemma3:4b")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()
    print(f"METHODS RUN: {args.dataset} | {args.generator} | n={args.n}", flush=True)
    out = run(args.dataset, args.generator, args.n, args.seed)
    stem = f"methods_{args.dataset}_{args.generator.replace(':','-')}_n{args.n}"
    (RES / f"{stem}.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out["methods"], indent=2))
    print("saved", RES / f"{stem}.json")


if __name__ == "__main__":
    main()
