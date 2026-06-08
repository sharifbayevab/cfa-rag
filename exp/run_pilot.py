# -*- coding: utf-8 -*-
"""Pilot run: validate the pipeline end-to-end and look at the
correctness-vs-faithfulness gap on a small sample."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from exp.attribution import attribute_sample
from exp.data import load_samples
from exp.generators import make_generator

RESULTS = Path(__file__).resolve().parent.parent / "results"
RESULTS.mkdir(exist_ok=True)


def run(dataset, generator_spec, n, seed, use_semantic=True):
    gen = make_generator(generator_spec)
    samples = load_samples(dataset, n=n, seed=seed)
    rows = []
    t0 = time.time()
    for k, s in enumerate(samples):
        sa = attribute_sample(gen, s, use_semantic=use_semantic)
        rows.append(sa)
        if (k + 1) % 5 == 0:
            print(f"  {k+1}/{len(samples)}  ({(time.time()-t0)/(k+1):.1f}s/sample)")
    return gen, rows


def summarize(rows):
    n = len(rows)
    correct = [r for r in rows if r.correct]
    em = sum(r.full_correct_em for r in rows) / n
    f1 = sum(r.full_correct_f1 for r in rows) / n
    cover = sum(r.full_correct_cover for r in rows) / n
    cb_em = sum(r.closedbook_correct_cover for r in rows) / n
    # faithfulness gap among CORRECT answers
    n_corr = len(correct) or 1
    parametric = sum(1 for r in correct if r.parametric) / n_corr
    no_causal = sum(1 for r in correct if not r.any_causal) / n_corr
    # do causal fragments coincide with gold?
    gold_causal_prec, gold_causal_rec, cnt = 0.0, 0.0, 0
    for r in correct:
        causal = set(r.causal_idxs); gold = set(r.gold_idxs)
        if causal:
            gold_causal_prec += len(causal & gold) / len(causal)
        if gold:
            gold_causal_rec += len(causal & gold) / len(gold)
        cnt += 1
    cnt = cnt or 1
    return {
        "n": n, "EM": round(em, 3), "F1": round(f1, 3), "cover": round(cover, 3),
        "closedbook_cover": round(cb_em, 3),
        "n_correct": len(correct),
        "frac_correct_parametric": round(parametric, 3),
        "frac_correct_no_causal_frag": round(no_causal, 3),
        "causal_vs_gold_precision": round(gold_causal_prec / cnt, 3),
        "causal_vs_gold_recall": round(gold_causal_rec / cnt, 3),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="hotpotqa")
    ap.add_argument("--generator", default="ollama:gemma3:4b")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--no-semantic", action="store_true")
    args = ap.parse_args()

    print(f"PILOT: {args.dataset} | {args.generator} | n={args.n}")
    gen, rows = run(args.dataset, args.generator, args.n, args.seed, use_semantic=not args.no_semantic)
    summ = summarize(rows)
    print("\n=== SUMMARY ===")
    print(json.dumps(summ, indent=2))
    # show a few qualitative cases
    print("\n=== sample cases ===")
    for r in rows[:4]:
        print(f"Q: {r.question[:70]}")
        print(f"   gold={r.golds} | full='{r.ans_full[:40]}'(EM{r.full_correct_em:.0f}) "
              f"closedbook='{r.ans_closedbook[:30]}'(EM{r.closedbook_correct_em:.0f})")
        print(f"   gold_frags={r.gold_idxs} causal_frags={r.causal_idxs}")
    out = RESULTS / f"pilot_{args.dataset}_{args.generator.replace(':','-')}_n{args.n}.json"
    out.write_text(json.dumps(summ, indent=2), encoding="utf-8")
    print("saved", out)


if __name__ == "__main__":
    main()
