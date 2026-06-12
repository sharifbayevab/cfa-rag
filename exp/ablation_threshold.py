# -*- coding: utf-8 -*-
"""Offline ablation of the load-bearing threshold tau on the combined causal
score, computed from saved raw_*.jsonl (no new generation needed).

For each tau we recompute the causal set S_tau = {i : causal_score_i > tau} and
report: average causal-set size, relevance-vs-causality Jaccard, and
causal-vs-gold precision/recall. Shows how the causal definition's strictness
trades set size against gold alignment.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np

RES = Path(__file__).resolve().parent.parent / "results"
OUT = Path(__file__).resolve().parent.parent / "article3"


def load_rows(pattern="raw_*.jsonl"):
    rows = []
    for p in glob.glob(str(RES / pattern)):
        nm = Path(p).name
        if "test_" in nm or nm.startswith("raw_oc_"):
            continue                              # main distractor-pool grid only (15 cells)
        for line in Path(p).read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def sweep(rows, taus=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7)):
    out = []
    for tau in taus:
        sizes, jacc, prec, rec = [], [], [], []
        for r in rows:
            cs = np.array(r["causal_scores"])
            causal = set(np.where(cs > tau)[0].tolist())
            gold = set(r["gold_idxs"])
            toprel = set(r["toprel_idxs"])
            sizes.append(len(causal))
            jacc.append(len(causal & toprel) / len(causal | toprel) if (causal or toprel) else 1.0)
            if causal:
                prec.append(len(causal & gold) / len(causal))
            if gold:
                rec.append(len(causal & gold) / len(gold))
        out.append({
            "tau": tau,
            "avg_causal_size": round(float(np.mean(sizes)), 2),
            "rel_causal_jaccard": round(float(np.mean(jacc)), 3),
            "causal_gold_prec": round(float(np.mean(prec)) if prec else 0.0, 3),
            "causal_gold_rec": round(float(np.mean(rec)) if rec else 0.0, 3),
        })
    return out


def emit_latex(sweep_rows):
    lines = [r"\begin{tabular}{lcccc}", r"\toprule",
             r"$\tau$ & Avg.\ causal size & relCausJ & cgPrec & cgRec\\", r"\midrule"]
    for s in sweep_rows:
        lines.append(f"{s['tau']:.1f} & {s['avg_causal_size']:.2f} & {s['rel_causal_jaccard']:.3f} & "
                     f"{s['causal_gold_prec']:.3f} & {s['causal_gold_rec']:.3f}\\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "tab_threshold.tex").write_text("\n".join(lines), encoding="utf-8")


def main():
    rows = load_rows()
    print(f"loaded {len(rows)} attributed questions")
    if not rows:
        return
    sw = sweep(rows)
    print(f"{'tau':>5}{'size':>8}{'relCausJ':>10}{'cgPrec':>8}{'cgRec':>7}")
    for s in sw:
        print(f"{s['tau']:>5}{s['avg_causal_size']:>8}{s['rel_causal_jaccard']:>10}"
              f"{s['causal_gold_prec']:>8}{s['causal_gold_rec']:>7}")
    emit_latex(sw)
    print("emitted article3/tab_threshold.tex")


if __name__ == "__main__":
    main()
