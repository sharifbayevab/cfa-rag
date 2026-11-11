"""Leakage-free exploratory search for a better CFA passage ranker.

The current selector is a pointwise classifier over duplicated generator rows.
This script instead aggregates the fixed generator panel into one target per
question/passage and evaluates pointwise and pairwise ranking objectives under
the same question-held-out folds. Gold passage labels are evaluation-only.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, ndcg_score, roc_auc_score
from sklearn.model_selection import GroupKFold

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from rebuild import INPUT, OUTPUT, load_grid  # noqa: E402
from selector_oof import gb, load_aligned, validate_split  # noqa: E402


def build_aggregated():
    X, y, datasets, generators, qids, passage_idx, names = load_aligned()
    cover, shift = [], []
    for (_, _), rows in sorted(
        load_grid().items(), key=lambda item: f"raw_{item[0][0]}_{item[0][1]}_n150.jsonl"
    ):
        for row in rows:
            cover.extend([int(row["cover_full"])] * len(row["causal_scores"]))
            shift.extend(row["causal_scores"])
    cover, shift = np.asarray(cover), np.asarray(shift)
    assert len(cover) == len(y)

    records = []
    for question in sorted(set(qids)):
        qrows = np.where(qids == question)[0]
        for pi in sorted(set(passage_idx[qrows])):
            take = qrows[passage_idx[qrows] == pi]
            # Retrieval features are generator-independent. Average only to
            # make the invariant explicit and robust to serialization noise.
            features = X[take].mean(axis=0)
            max_spread = np.max(np.ptp(X[take], axis=0))
            if max_spread > 1e-5:
                raise ValueError(f"Generator-dependent features at {question}/{pi}: {max_spread}")
            correct = cover[take].astype(bool)
            records.append({
                "question": question,
                "dataset": str(datasets[take[0]]),
                "passage_idx": int(pi),
                "X": features,
                "all_target": float(y[take].mean()),
                "correct_target": float(y[take][correct].mean()) if correct.any() else np.nan,
                "correct_shift": float(shift[take][correct].mean()) if correct.any() else np.nan,
                "n_correct": int(correct.sum()),
            })
    return records, list(names)


def make_pairs(X, target, groups, weights=None):
    diffs, labels, pair_weights = [], [], []
    for group in sorted(set(groups)):
        take = np.where(groups == group)[0]
        for a_pos, a in enumerate(take):
            if not np.isfinite(target[a]):
                continue
            for b in take[a_pos + 1:]:
                if not np.isfinite(target[b]) or target[a] == target[b]:
                    continue
                sign = int(target[a] > target[b])
                w = abs(target[a] - target[b])
                if weights is not None:
                    w *= min(weights[a], weights[b])
                diffs.extend([X[a] - X[b], X[b] - X[a]])
                labels.extend([sign, 1 - sign])
                pair_weights.extend([w, w])
    return np.asarray(diffs), np.asarray(labels), np.asarray(pair_weights)


def pairwise_scores(model, X, groups):
    scores = np.zeros(len(X))
    for group in sorted(set(groups)):
        take = np.where(groups == group)[0]
        for a in take:
            others = take[take != a]
            scores[a] = model.predict_proba(X[a][None, :] - X[others])[:, 1].mean()
    return scores


def rank_normalize(scores, groups):
    out = np.zeros(len(scores))
    for group in sorted(set(groups)):
        take = np.where(groups == group)[0]
        order = np.argsort(-scores[take], kind="stable")
        values = np.linspace(1.0, 0.0, len(take)) if len(take) > 1 else np.ones(1)
        out[take[order]] = values
    return out


def evaluate(scores, records, gold_by_question, target_name):
    groups = np.asarray([r["question"] for r in records])
    target = np.asarray([r[target_name] for r in records])
    finite = np.isfinite(target)
    binary = target > 0
    result = {
        "auc_any": float(roc_auc_score(binary[finite], scores[finite])),
        "ap_any": float(average_precision_score(binary[finite], scores[finite])),
    }
    ndcg, recall4, gold_recall4, all_gold4, length4 = [], [], [], [], []
    for group in sorted(set(groups)):
        take = np.where(groups == group)[0]
        valid = np.isfinite(target[take])
        if valid.any() and target[take][valid].max() > 0:
            ndcg.append(ndcg_score(target[take][valid][None, :], scores[take][valid][None, :], k=4))
            positives = target[take] > 0
            local_chosen = np.argsort(-scores[take], kind="stable")[:4]
            recall4.append(float(positives[local_chosen].sum() / positives.sum()))
        chosen = take[np.argsort(-scores[take], kind="stable")[:4]]
        selected = {records[i]["passage_idx"] for i in chosen}
        gold = set(gold_by_question[group])
        gold_recall4.append(len(selected & gold) / len(gold))
        all_gold4.append(float(gold <= selected))
        length4.append(sum(records[i]["X"][5] for i in chosen))
    result.update({
        "ndcg_at_4": float(np.mean(ndcg)),
        "target_recall_at_4": float(np.mean(recall4)),
        "gold_recall_at_4": float(np.mean(gold_recall4)),
        "all_gold_at_4": float(np.mean(all_gold4)),
        "selected_length_feature_at_4": float(np.mean(length4)),
    })
    return result


def main():
    records, names = build_aggregated()
    X = np.asarray([r["X"] for r in records])
    groups = np.asarray([r["question"] for r in records])
    all_target = np.asarray([r["all_target"] for r in records])
    correct_target = np.asarray([r["correct_target"] for r in records])
    correct_shift = np.asarray([r["correct_shift"] for r in records])
    correct_weight = np.asarray([r["n_correct"] / 5 for r in records])
    ce_col = names.index("ce")
    folds = list(GroupKFold(n_splits=5).split(X, all_target, groups))
    predictions = defaultdict(lambda: np.full(len(records), np.nan))

    for fold, (train, test) in enumerate(folds):
        validate_split(groups, train, test)
        # Current duplicated-label pointwise formulation, reproduced on the
        # equivalent aggregate table using fractional sample copies.
        expanded_X, expanded_y, expanded_w = [], [], []
        for i in train:
            p = all_target[i]
            expanded_X.extend([X[i], X[i]])
            expanded_y.extend([0, 1])
            expanded_w.extend([1 - p, p])
        current = gb().fit(np.asarray(expanded_X), np.asarray(expanded_y), sample_weight=np.asarray(expanded_w))
        predictions["pointwise_binary"][test] = current.predict_proba(X[test])[:, 1]

        all_reg = GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=.05,
                                            loss="huber", random_state=0)
        all_reg.fit(X[train], all_target[train])
        predictions["panel_consensus"][test] = all_reg.predict(X[test])

        usable = train[np.isfinite(correct_target[train]) & (correct_weight[train] > 0)]
        correct_reg = GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=.05,
                                                loss="huber", random_state=0)
        correct_reg.fit(X[usable], correct_target[usable], sample_weight=correct_weight[usable])
        predictions["correct_consensus"][test] = correct_reg.predict(X[test])

        shift_reg = GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=.05,
                                              loss="huber", random_state=0)
        shift_reg.fit(X[usable], correct_shift[usable], sample_weight=correct_weight[usable])
        predictions["correct_shift"][test] = shift_reg.predict(X[test])

        pair_X, pair_y, pair_w = make_pairs(X[usable], correct_target[usable], groups[usable],
                                            correct_weight[usable])
        pair = HistGradientBoostingClassifier(max_iter=160, learning_rate=.06, max_leaf_nodes=15,
                                              l2_regularization=1.0, random_state=0)
        pair.fit(pair_X, pair_y, sample_weight=pair_w)
        predictions["pairwise_correct"][test] = pairwise_scores(pair, X[test], groups[test])

        linear_X, linear_y, linear_w = make_pairs(X[usable], correct_target[usable], groups[usable],
                                                  correct_weight[usable])
        linear = LogisticRegression(C=.2, max_iter=2000, random_state=0)
        linear.fit(linear_X, linear_y, sample_weight=linear_w)
        predictions["pairwise_linear"][test] = pairwise_scores(linear, X[test], groups[test])
        print(f"fold {fold + 1}/5", flush=True)

    predictions["cross_encoder"] = X[:, ce_col].copy()
    ce_ranked = rank_normalize(predictions["cross_encoder"], groups)
    # Fixed, transparent fusion: deletion utility remains primary, while the
    # cross-encoder breaks noisy utility ties and protects relevance.
    for base in ["pointwise_binary", "panel_consensus", "correct_consensus",
                 "correct_shift", "pairwise_correct", "pairwise_linear"]:
        utility_ranked = rank_normalize(predictions[base], groups)
        predictions[f"{base}_fusion75"] = .75 * utility_ranked + .25 * ce_ranked

    plans = [json.loads(x) for x in (INPUT / "efficiency_plan.jsonl").read_text().splitlines() if x.strip()]
    gold_by_question = {p["question_group"]: p["gold_idxs"] for p in plans}
    output = {
        "n_questions": len(set(groups)),
        "n_question_passages": len(records),
        "features": names,
        "protocol": "Five question-held-out folds; gold passage labels used only for evaluation.",
        "methods": {},
    }
    for name, scores in predictions.items():
        if not np.all(np.isfinite(scores)):
            raise ValueError(f"Missing scores for {name}")
        output["methods"][name] = {
            "all_generator_target": evaluate(scores, records, gold_by_question, "all_target"),
            "correct_generator_target": evaluate(scores, records, gold_by_question, "correct_target"),
        }
    OUTPUT.mkdir(exist_ok=True)
    (OUTPUT / "selector_methodology_search.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
