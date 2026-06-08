# -*- coding: utf-8 -*-
"""Train and evaluate the amortized causal selector.

Predicts whether a fragment is load-bearing (CFA label) from cheap features.
Key questions:
  (1) AUC of the learned predictor;
  (2) does it beat relevance alone (the standard selection signal)?
  (3) does it generalize across held-out datasets and generators?
  (4) which cheap features carry the causal signal?
All offline (no generation). Saves a model, a metrics JSON, and a figure.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
FIG = ROOT / "article3" / "images"
FIG.mkdir(parents=True, exist_ok=True)


def load():
    d = np.load(RES / "selector_dataset.npz", allow_pickle=True)
    return d["X"], d["y"], d["dataset"], d["gen"], list(d["feat_names"])


def cv_auc(X, y, model_fn, k=5):
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=0)
    aucs, aps = [], []
    for tr, te in skf.split(X, y):
        m = model_fn().fit(X[tr], y[tr])
        p = m.predict_proba(X[te])[:, 1]
        aucs.append(roc_auc_score(y[te], p)); aps.append(average_precision_score(y[te], p))
    return float(np.mean(aucs)), float(np.mean(aps))


def group_generalization(X, y, groups, model_fn):
    """Leave-one-group-out AUC (train on all but one dataset/generator)."""
    out = {}
    for g in sorted(set(groups)):
        tr = groups != g; te = groups == g
        if y[te].sum() < 5 or y[tr].sum() < 5:
            continue
        m = model_fn().fit(X[tr], y[tr])
        p = m.predict_proba(X[te])[:, 1]
        out[str(g)] = round(float(roc_auc_score(y[te], p)), 3)
    return out


def main():
    X, y, ds, gen, names = load()
    print(f"X={X.shape} pos_rate={y.mean():.3f} feats={names}")
    gb = lambda: GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.05)
    lr = lambda: LogisticRegression(max_iter=1000, class_weight="balanced")

    res = {}
    res["auc_gb"], res["ap_gb"] = cv_auc(X, y, gb)
    res["auc_lr"], res["ap_lr"] = cv_auc(X, y, lr)
    # relevance-only baseline (feature 0 = dense relevance; also cross-encoder alone)
    res["auc_rel_only"] = round(float(roc_auc_score(y, X[:, 0])), 3)
    res["auc_ce_only"] = round(float(roc_auc_score(y, X[:, names.index("ce")])), 3)
    res["base_rate"] = round(float(y.mean()), 3)

    res["generalize_by_dataset"] = group_generalization(X, y, ds, gb)
    res["generalize_by_generator"] = group_generalization(X, y, gen, gb)

    # feature importance (permutation-free: GB built-in)
    m = gb().fit(X, y)
    imp = dict(sorted(zip(names, [round(float(v), 3) for v in m.feature_importances_]),
                      key=lambda kv: -kv[1]))
    res["feature_importance"] = imp

    print(json.dumps(res, indent=2))
    (RES / "selector_metrics.json").write_text(json.dumps(res, indent=2))

    # save trained model
    import pickle
    pickle.dump(m, open(RES / "causal_selector.pkl", "wb"))

    # figure: AUC comparison
    fig, ax = plt.subplots(figsize=(6, 3.6))
    labels = ["Relevance\nonly", "Cross-enc.\nonly", "Learned\n(LogReg)", "Learned\n(GBoost)"]
    vals = [res["auc_rel_only"], res["auc_ce_only"], res["auc_lr"], res["auc_gb"]]
    bars = ax.bar(labels, vals, color=["#94a3b8", "#60a5fa", "#f59e0b", "#16a34a"])
    ax.axhline(0.5, ls="--", c="gray", lw=0.8)
    ax.set_ylabel("ROC-AUC (predict load-bearing)"); ax.set_ylim(0.4, 1.0)
    for b, v in zip(bars, vals):
        ax.text(b.get_x()+b.get_width()/2, v+0.01, f"{v:.2f}", ha="center", fontsize=9)
    ax.set_title("Cheap features predict causal contribution better than relevance")
    fig.tight_layout(); fig.savefig(FIG/"selector_auc.pdf"); fig.savefig(FIG/"selector_auc.png", dpi=160)

    # LaTeX macros
    macros = [
        f"\\newcommand{{\\selAUC}}{{{res['auc_gb']:.2f}}}",
        f"\\newcommand{{\\selAUCrel}}{{{res['auc_rel_only']:.2f}}}",
        f"\\newcommand{{\\selAUCce}}{{{res['auc_ce_only']:.2f}}}",
        f"\\newcommand{{\\selBaseRate}}{{{res['base_rate']*100:.0f}}}",
    ]
    (ROOT/"article3"/"selmacros.tex").write_text("\n".join(macros)+"\n")
    print("saved model, metrics, figure, selmacros.tex")


if __name__ == "__main__":
    main()
