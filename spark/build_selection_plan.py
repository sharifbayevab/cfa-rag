"""Build an auditable equal-budget selection plan for fresh downstream QA.

The target selector is trained only on the four question folds that exclude the
test question.  Baselines retain exactly the same number of passages as that
selector for each question.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GroupKFold


HERE = Path(__file__).resolve().parent
PAPER = HERE.parents[1]
sys.path.insert(0, str(PAPER / "repro"))

from selector_oof import load_aligned, load_grid, validate_split  # noqa: E402


def model():
    return GradientBoostingClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.05, random_state=0
    )


def top_k(values: np.ndarray, indices: np.ndarray, k: int) -> list[int]:
    order = np.argsort(-values, kind="stable")[:k]
    return sorted(int(v) for v in indices[order])


def main() -> None:
    X, y, datasets, generators, qids, passage_idx, names = load_aligned()
    raw_by_group = {}
    for (dataset, _generator), raw_rows in load_grid().items():
        for raw in raw_rows:
            group = f"{dataset}:{raw['qid']}"
            evidence = {
                "question": raw["question"],
                "golds": raw["golds"],
                "gold_idxs": raw["gold_idxs"],
            }
            if group in raw_by_group and raw_by_group[group] != evidence:
                raise ValueError(f"Raw question metadata differs across generators for {group}")
            raw_by_group[group] = evidence
    folds = list(GroupKFold(n_splits=5).split(X, y, qids))
    all9 = np.full(len(y), np.nan)
    ce_len = np.full(len(y), np.nan)
    fold_of = np.full(len(y), -1, dtype=int)
    ce_len_cols = [names.index("ce_rank"), names.index("length")]

    for fold, (train, test) in enumerate(folds):
        validate_split(qids, train, test)
        fold_of[test] = fold
        all9[test] = model().fit(X[train], y[train]).predict_proba(X[test])[:, 1]
        ce_len[test] = (
            model()
            .fit(X[train][:, ce_len_cols], y[train])
            .predict_proba(X[test][:, ce_len_cols])[:, 1]
        )

    rel_col = names.index("rel")
    ce_col = names.index("ce")
    rows = []
    for group in sorted(set(qids)):
        members = np.where(qids == group)[0]
        by_generator = []
        for generator in sorted(set(generators[members])):
            take = members[generators[members] == generator]
            take = take[np.argsort(passage_idx[take])]
            by_generator.append(take)
        reference = by_generator[0]
        for take in by_generator[1:]:
            if not np.array_equal(passage_idx[reference], passage_idx[take]):
                raise ValueError(f"Passage index mismatch for {group}")
            if not np.allclose(X[reference], X[take], rtol=0, atol=1e-12):
                raise ValueError(f"Generator-dependent retrieval features for {group}")
            if not np.allclose(all9[reference], all9[take], rtol=0, atol=1e-12):
                raise ValueError(f"Generator-dependent OOF predictions for {group}")

        idx = passage_idx[reference]
        p = all9[reference]
        selected = idx[p >= 0.5]
        if not len(selected):
            selected = np.array([idx[int(np.argmax(p))]])
        selected = sorted(int(v) for v in selected)
        k = len(selected)
        fixed_k = min(4, len(idx))
        dataset, qid = group.split(":", 1)
        evidence = raw_by_group[group]
        rows.append(
            {
                "question_group": group,
                "dataset": dataset,
                "qid": qid,
                "question": evidence["question"],
                "golds": evidence["golds"],
                "gold_idxs": evidence["gold_idxs"],
                "fold": int(fold_of[reference][0]),
                "pool_size": int(len(idx)),
                "budget_k": k,
                "fixed_budget_k": fixed_k,
                "selections": {
                    "oof_all9": selected,
                    "dense_equal_k": top_k(X[reference, rel_col], idx, k),
                    "cross_encoder_equal_k": top_k(X[reference, ce_col], idx, k),
                    "ce_rank_length_equal_k": top_k(ce_len[reference], idx, k),
                    "oof_all9_top4": top_k(p, idx, fixed_k),
                    "dense_top4": top_k(X[reference, rel_col], idx, fixed_k),
                    "cross_encoder_top4": top_k(X[reference, ce_col], idx, fixed_k),
                    "ce_rank_length_top4": top_k(ce_len[reference], idx, fixed_k),
                },
                "oof_all9_probabilities": p.tolist(),
                "oof_ce_rank_length_probabilities": ce_len[reference].tolist(),
            }
        )

    out = HERE / "selection_plan.jsonl"
    out.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    metadata = {
        "n_questions": len(rows),
        "datasets": {d: sum(r["dataset"] == d for r in rows) for d in sorted(set(datasets))},
        "folds": 5,
        "random_state": 0,
        "threshold": 0.5,
        "fallback": "highest predicted probability",
        "equal_budget_definition": "each baseline retains budget_k passages, equal to oof_all9",
        "fixed_budget_sensitivity": "all selectors additionally retain min(4, pool_size) passages",
        "sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
    }
    (HERE / "selection_plan_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
