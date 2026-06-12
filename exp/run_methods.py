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


def run(dataset, generator_spec, n, seed, k=4, checkpoint_path=None):
    gen = make_generator(generator_spec)
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
    processed = 0
    for ci, s in enumerate(samples):
        if s.get("qid") in done:
            continue
        s["_rel"] = relevance_scores(s["question"], s["fragments"])
        gold = {i for i, f in enumerate(s["fragments"]) if f.is_gold}
        n_gold = max(1, len(gold))
        per = {}
        for mname, fn in METHODS.items():
            kw = {"k": k, "rng": rng}
            if mname in NEEDS_GEN:
                idxs, probe = fn(gen, s, **kw)
            else:
                idxs, probe = fn(s, **kw)
            ans = answer(gen, s["question"], s["fragments"], idxs)
            per[mname] = {
                "cover": float(metrics.answer_contains_gold(ans, s["golds"])),
                "f1": float(metrics.best_f1(ans, s["golds"])),
                "tokens": float(context_tokens(s["fragments"], idxs)),
                "probe": float(probe),
                "gold_recall": len(set(idxs) & gold) / n_gold,
            }
        row = {"qid": s.get("qid"), "methods": per}
        rows.append(row)
        if fh:                                   # checkpoint every sample (crash-safe)
            fh.write(json.dumps(row, ensure_ascii=False) + "\n"); fh.flush()
        processed += 1
        if processed % 10 == 0:
            print(f"  {len(rows)}/{len(samples)}", flush=True)
    if fh:
        fh.close()
    agg = {m: {"cover": [], "f1": [], "tokens": [], "probe": [], "gold_recall": []}
           for m in METHODS}
    for r in rows:
        for mname, v in r["methods"].items():
            if mname in agg:
                for key in agg[mname]:
                    agg[mname][key].append(v[key])
    summ = {m: {"acc": round(float(np.mean(v["cover"])), 3),
                "f1": round(float(np.mean(v["f1"])), 3),
                "tokens": round(float(np.mean(v["tokens"])), 1),
                "probe_calls": round(float(np.mean(v["probe"])), 1),
                "gold_recall": round(float(np.mean(v["gold_recall"])), 3)}
            for m, v in agg.items()}
    return {"dataset": dataset, "generator": generator_spec, "n": len(rows), "methods": summ}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="hotpotqa")
    ap.add_argument("--generator", default="ollama:gemma3:4b")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()
    print(f"METHODS RUN: {args.dataset} | {args.generator} | n={args.n}", flush=True)
    stem = f"methods_{args.dataset}_{args.generator.replace(':','-')}_n{args.n}"
    raw_path = RES / f"methodsraw_{stem.replace('methods_','')}.jsonl"
    out = run(args.dataset, args.generator, args.n, args.seed, checkpoint_path=raw_path)
    (RES / f"{stem}.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out["methods"], indent=2))
    print("saved", RES / f"{stem}.json")


if __name__ == "__main__":
    main()
