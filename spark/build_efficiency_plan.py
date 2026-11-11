"""Freeze leakage-free rankings and an inner-CV calibrated CFA threshold."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import fbeta_score
from sklearn.model_selection import GroupKFold

HERE = Path(__file__).resolve().parent
PAPER = HERE.parents[1]
sys.path.insert(0, str(PAPER / "repro"))

from selector_oof import gb, load_aligned, validate_split  # noqa: E402


def ranked(values, indices):
    return [int(indices[i]) for i in np.argsort(-values, kind="stable")]


def choose_threshold(y, predictions):
    candidates = np.arange(0.05, 0.801, 0.025)
    scores = [(fbeta_score(y, predictions >= t, beta=1, zero_division=0), float(t)) for t in candidates]
    # Higher threshold wins an exact F1 tie, reducing unnecessary context.
    return max(scores, key=lambda item: (item[0], item[1]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-plan", type=Path, default=Path("selection_plan.jsonl"))
    parser.add_argument("--fresh-results", type=Path, default=Path("results/fresh_qa.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("efficiency_plan.jsonl"))
    args = parser.parse_args()

    base = {r["question_group"]: r for r in map(json.loads, args.base_plan.read_text().splitlines())}
    fresh = {r["question_group"]: r for r in map(json.loads, args.fresh_results.read_text().splitlines())}
    X, y, datasets, generators, qids, passage_idx, names = load_aligned()
    folds = list(GroupKFold(n_splits=5).split(X, y, qids))
    all9 = np.full(len(y), np.nan)
    ce_len = np.full(len(y), np.nan)
    fold_of = np.full(len(y), -1, dtype=int)
    threshold_of_fold = {}
    ce_len_cols = [names.index("ce_rank"), names.index("length")]

    for fold, (train, test) in enumerate(folds):
        validate_split(qids, train, test)
        inner_predictions = np.full(len(train), np.nan)
        inner_groups = qids[train]
        for inner_train_local, inner_valid_local in GroupKFold(n_splits=4).split(X[train], y[train], inner_groups):
            inner_train, inner_valid = train[inner_train_local], train[inner_valid_local]
            validate_split(qids, inner_train, inner_valid)
            inner_predictions[inner_valid_local] = gb().fit(X[inner_train], y[inner_train]).predict_proba(X[inner_valid])[:, 1]
        score, threshold = choose_threshold(y[train], inner_predictions)
        threshold_of_fold[fold] = {"threshold": threshold, "inner_oof_f1": float(score)}
        model = gb().fit(X[train], y[train])
        all9[test] = model.predict_proba(X[test])[:, 1]
        ce_len[test] = gb().fit(X[train][:, ce_len_cols], y[train]).predict_proba(X[test][:, ce_len_cols])[:, 1]
        fold_of[test] = fold

    rows = []
    for group in sorted(set(qids)):
        members = np.where(qids == group)[0]
        reference_generator = sorted(set(generators[members]))[0]
        take = members[generators[members] == reference_generator]
        take = take[np.argsort(passage_idx[take])]
        idx = passage_idx[take]
        fold = int(fold_of[take][0])
        threshold = threshold_of_fold[fold]["threshold"]
        calibrated = idx[all9[take] >= threshold].astype(int).tolist()
        if not calibrated:
            calibrated = [int(idx[np.argmax(all9[take])])]
        cc = np.asarray(fresh[group]["contextcite"]["coefficients"])
        row = dict(base[group])
        row.update({
            "rankings": {
                "oof": ranked(all9[take], idx),
                "dense": ranked(X[take, names.index("rel")], idx),
                "cross_encoder": ranked(X[take, names.index("ce")], idx),
                "ce_rank_length": ranked(ce_len[take], idx),
                "contextcite": ranked(cc, idx),
            },
            "calibrated_threshold": threshold,
            "calibrated_inner_oof_f1": threshold_of_fold[fold]["inner_oof_f1"],
            "calibrated_selection": sorted(calibrated),
            "threshold_selections": {},
        })
        for threshold_value in [0.1, 0.2, 0.3, 0.4, 0.5]:
            selected = idx[all9[take] >= threshold_value].astype(int).tolist()
            if not selected:
                selected = [int(idx[np.argmax(all9[take])])]
            row["threshold_selections"][f"{threshold_value:.1f}"] = sorted(selected)
        rows.append(row)

    args.output.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    metadata = {
        "n_questions": len(rows),
        "outer_folds": 5,
        "inner_folds": 4,
        "calibration_objective": "passage-level F1 on inner question-held-out predictions",
        "threshold_grid": [float(x) for x in np.arange(0.05, 0.801, 0.025)],
        "thresholds_by_outer_fold": threshold_of_fold,
        "base_plan_sha256": hashlib.sha256(args.base_plan.read_bytes()).hexdigest(),
        "fresh_results_sha256": hashlib.sha256(args.fresh_results.read_bytes()).hexdigest(),
        "output_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
    }
    args.output.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
