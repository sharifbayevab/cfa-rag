"""Build a leakage-free selector from control-confirmed supporting labels.

This is an exploratory replacement for the unsigned pointwise CFA selector.
It separates supporting interventions from harmful/brittle answer changes,
aggregates the supporting label across teacher readers, and chooses both the
cheap-feature model and its fusion with frozen CE scores by inner grouped CV.
The held-out Qwen labels and dataset gold annotations are evaluation-only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
)
from sklearn.metrics import average_precision_score, ndcg_score, roc_auc_score
from sklearn.model_selection import GroupKFold

HERE = Path(__file__).resolve().parent
PAPER = HERE.parent
sys.path.insert(0, str(HERE))
from selector_oof import load_aligned, validate_split  # noqa: E402

TEACHERS = (
    "openai-gpt-4.1-nano",
    "openai-gpt-4o-mini",
    "openai-gpt-4.1-mini",
    "openai-gpt-4.1",
    "gemini-gemini-3.5-flash-lite",
)
TARGET_READER = "ollama-qwen3-4b-instruct"
DATASETS = ("hotpotqa", "2wiki", "musique")
ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def estimator(name: str):
    if name == "gbr_d2":
        return GradientBoostingRegressor(
            n_estimators=250, max_depth=2, learning_rate=0.04,
            loss="huber", random_state=0,
        )
    if name == "gbr_d3":
        return GradientBoostingRegressor(
            n_estimators=200, max_depth=3, learning_rate=0.05,
            loss="huber", random_state=0,
        )
    if name == "hist_7":
        return HistGradientBoostingRegressor(
            max_iter=200, learning_rate=0.05, max_leaf_nodes=7,
            l2_regularization=1.0, random_state=0,
        )
    if name == "extra_5":
        return ExtraTreesRegressor(
            n_estimators=300, min_samples_leaf=5, max_features=0.8,
            n_jobs=-1, random_state=0,
        )
    raise KeyError(name)


MODELS = ("gbr_d2", "gbr_d3", "hist_7", "extra_5")


def rank_normalize(values: np.ndarray, groups: np.ndarray) -> np.ndarray:
    out = np.zeros(len(values), dtype=float)
    for group in sorted(set(groups)):
        take = np.where(groups == group)[0]
        order = np.argsort(-values[take], kind="stable")
        scale = np.linspace(1.0, 0.0, len(take)) if len(take) > 1 else np.ones(1)
        out[take[order]] = scale
    return out


def support_ndcg(target: np.ndarray, scores: np.ndarray, groups: np.ndarray, k: int = 4) -> float:
    values = []
    for group in sorted(set(groups)):
        take = np.where(groups == group)[0]
        if target[take].max() <= 0:
            continue
        values.append(ndcg_score(target[take][None, :], scores[take][None, :], k=k))
    return float(np.mean(values))


def aggregate_features():
    X, _, datasets, generators, qids, passage_idx, names = load_aligned()
    rows = []
    for question in sorted(set(qids)):
        members = np.where(qids == question)[0]
        for index in sorted(set(passage_idx[members])):
            take = members[passage_idx[members] == index]
            feature = X[take].mean(axis=0)
            if np.max(np.ptp(X[take], axis=0)) > 1e-5:
                raise ValueError(f"Generator-dependent feature at {question}/{index}")
            rows.append({
                "question_group": question,
                "dataset": str(datasets[take[0]]),
                "qid": question.split(":", 1)[1],
                "passage_idx": int(index),
                "X": feature,
            })
    return rows, list(names)


def load_reader(signed_root: Path, reader: str):
    labels, files = {}, []
    for dataset in DATASETS:
        path = signed_root / f"signed_{dataset}_{reader}_n150_r0.jsonl"
        if not path.exists():
            raise FileNotFoundError(path)
        files.append(path)
        for line in path.read_text().splitlines():
            row = json.loads(line)
            key = f"{dataset}:{row['qid']}"
            for fragment in row["fragments"]:
                # A supporting label is retained only when the position-
                # preserving replacement independently gives the same role.
                confirmed = (
                    fragment["del_role"] == "supporting"
                    and fragment["rep_role"] == "supporting"
                )
                labels[(key, int(fragment["idx"]))] = float(confirmed)
    return labels, files


def ranking_metrics(scores, target, groups, records, gold_by_question):
    binary = target > 0
    result = {
        "auc": float(roc_auc_score(binary, scores)),
        "ap": float(average_precision_score(binary, scores)),
        "ndcg_at_4": support_ndcg(target, scores, groups),
    }
    support_recall, gold_recall, all_gold, selected_length = [], [], [], []
    for group in sorted(set(groups)):
        take = np.where(groups == group)[0]
        chosen = take[np.argsort(-scores[take], kind="stable")[:4]]
        positives = target[take] > 0
        if positives.any():
            support_recall.append(float(binary[chosen].sum() / positives.sum()))
        selected = {records[i]["passage_idx"] for i in chosen}
        gold = set(gold_by_question[group])
        gold_recall.append(len(selected & gold) / max(1, len(gold)))
        all_gold.append(float(gold <= selected))
        selected_length.append(sum(records[i]["X"][5] for i in chosen))
    result.update({
        "support_recall_at_4": float(np.mean(support_recall)),
        "gold_recall_at_4": float(np.mean(gold_recall)),
        "all_gold_at_4": float(np.mean(all_gold)),
        "selected_length_feature_at_4": float(np.mean(selected_length)),
    })
    return result


def paired_bootstrap(scores_a, scores_b, target, groups, records, gold_by_question, n_boot=5000):
    by_dataset = defaultdict(list)
    for group in sorted(set(groups)):
        dataset = group.split(":", 1)[0]
        by_dataset[dataset].append(group)
    rng = np.random.default_rng(0)

    def per_question(scores, group):
        take = np.where(groups == group)[0]
        chosen = take[np.argsort(-scores[take], kind="stable")[:4]]
        positives = target[take] > 0
        sr = float((target[chosen] > 0).sum() / positives.sum()) if positives.any() else np.nan
        selected = {records[i]["passage_idx"] for i in chosen}
        gold = set(gold_by_question[group])
        return sr, len(selected & gold) / max(1, len(gold))

    cache = {name: {g: per_question(scores, g) for g in sorted(set(groups))}
             for name, scores in (("a", scores_a), ("b", scores_b))}
    draws_support, draws_gold = [], []
    for _ in range(n_boot):
        sample = []
        for population in by_dataset.values():
            sample.extend(population[i] for i in rng.integers(0, len(population), len(population)))
        support = [cache["a"][g][0] - cache["b"][g][0] for g in sample
                   if np.isfinite(cache["a"][g][0])]
        gold = [cache["a"][g][1] - cache["b"][g][1] for g in sample]
        draws_support.append(np.mean(support))
        draws_gold.append(np.mean(gold))
    return {
        "support_recall_at_4_difference": float(np.nanmean([
            cache["a"][g][0] - cache["b"][g][0] for g in cache["a"]
        ])),
        "support_recall_at_4_ci95": [float(x) for x in np.percentile(draws_support, [2.5, 97.5])],
        "gold_recall_at_4_difference": float(np.mean([
            cache["a"][g][1] - cache["b"][g][1] for g in cache["a"]
        ])),
        "gold_recall_at_4_ci95": [float(x) for x in np.percentile(draws_gold, [2.5, 97.5])],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--signed-root", type=Path, required=True)
    parser.add_argument(
        "--base-plan", type=Path,
        default=PAPER / "experiments/spark/efficiency_plan.jsonl",
    )
    parser.add_argument(
        "--output", type=Path,
        default=PAPER / "experiments/spark/signed_support_plan.jsonl",
    )
    args = parser.parse_args()

    records, names = aggregate_features()
    X = np.asarray([row["X"] for row in records])
    groups = np.asarray([row["question_group"] for row in records])
    ce = X[:, names.index("ce")]
    ce_rank = rank_normalize(ce, groups)

    teacher_columns, source_files = [], []
    for reader in TEACHERS:
        labels, files = load_reader(args.signed_root, reader)
        source_files.extend(files)
        teacher_columns.append(np.asarray([
            labels[(row["question_group"], row["passage_idx"])] for row in records
        ]))
    teacher_target = np.mean(np.stack(teacher_columns, axis=1), axis=1)
    qwen_labels, qwen_files = load_reader(args.signed_root, TARGET_READER)
    source_files.extend(qwen_files)
    qwen_target = np.asarray([
        qwen_labels[(row["question_group"], row["passage_idx"])] for row in records
    ])

    outer = list(GroupKFold(n_splits=5).split(X, teacher_target, groups))
    utility_oof = np.full(len(records), np.nan)
    fusion_oof = np.full(len(records), np.nan)
    fold_of = np.full(len(records), -1, dtype=int)
    choices = []
    for fold, (train, test) in enumerate(outer):
        validate_split(groups, train, test)
        inner_groups = groups[train]
        inner = list(GroupKFold(n_splits=4).split(X[train], teacher_target[train], inner_groups))
        candidates = []
        inner_predictions = {}
        for model_name in MODELS:
            pred = np.full(len(train), np.nan)
            for inner_train_local, inner_valid_local in inner:
                inner_train, inner_valid = train[inner_train_local], train[inner_valid_local]
                validate_split(groups, inner_train, inner_valid)
                fitted = estimator(model_name).fit(X[inner_train], teacher_target[inner_train])
                pred[inner_valid_local] = fitted.predict(X[inner_valid])
            inner_predictions[model_name] = pred
            utility_rank = rank_normalize(pred, inner_groups)
            ce_inner = rank_normalize(ce[train], inner_groups)
            for alpha in ALPHAS:
                fused = alpha * utility_rank + (1.0 - alpha) * ce_inner
                score = support_ndcg(teacher_target[train], fused, inner_groups)
                candidates.append((score, alpha, model_name))
        best_score, best_alpha, best_model = max(candidates, key=lambda x: (x[0], x[1], x[2]))
        fitted = estimator(best_model).fit(X[train], teacher_target[train])
        raw = fitted.predict(X[test])
        utility_oof[test] = raw
        fusion_oof[test] = (
            best_alpha * rank_normalize(raw, groups[test])
            + (1.0 - best_alpha) * rank_normalize(ce[test], groups[test])
        )
        fold_of[test] = fold
        choices.append({
            "fold": fold, "model": best_model, "utility_weight": best_alpha,
            "inner_teacher_ndcg_at_4": best_score,
            "train_questions": len(set(groups[train])),
            "test_questions": len(set(groups[test])),
        })
        print(f"fold {fold + 1}/5: {best_model}, alpha={best_alpha:.2f}, ndcg={best_score:.4f}", flush=True)

    if not np.all(np.isfinite(utility_oof)) or not np.all(np.isfinite(fusion_oof)):
        raise ValueError("Missing OOF predictions")
    base_plan = {row["question_group"]: row for row in map(json.loads, args.base_plan.read_text().splitlines())}
    gold_by_question = {key: row["gold_idxs"] for key, row in base_plan.items()}
    methods = {
        "cross_encoder": ce,
        "signed_support": utility_oof,
        "signed_support_ce": fusion_oof,
    }
    evaluation = {
        name: ranking_metrics(score, qwen_target, groups, records, gold_by_question)
        for name, score in methods.items()
    }
    evaluation["signed_support_ce_vs_cross_encoder_bootstrap"] = paired_bootstrap(
        fusion_oof, ce, qwen_target, groups, records, gold_by_question
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out_rows = []
    for group in sorted(set(groups)):
        take = np.where(groups == group)[0]
        indices = np.asarray([records[i]["passage_idx"] for i in take])
        order = np.argsort(indices)
        take, indices = take[order], indices[order]
        base = dict(base_plan[group])
        base["signed_support"] = {
            "outer_fold": int(fold_of[take][0]),
            "teacher_support_probabilities": teacher_target[take].tolist(),
            "heldout_qwen_support_labels_evaluation_only": qwen_target[take].astype(int).tolist(),
            "utility_scores": utility_oof[take].tolist(),
            "fusion_scores": fusion_oof[take].tolist(),
        }
        base["rankings"]["signed_support"] = [
            int(indices[i]) for i in np.argsort(-utility_oof[take], kind="stable")
        ]
        base["rankings"]["signed_support_ce"] = [
            int(indices[i]) for i in np.argsort(-fusion_oof[take], kind="stable")
        ]
        out_rows.append(base)
    args.output.write_text("\n".join(json.dumps(row) for row in out_rows) + "\n")
    metadata = {
        "status": "exploratory; held-out reader labels and gold labels are evaluation-only",
        "n_questions": len(set(groups)),
        "n_question_passages": len(records),
        "teachers": list(TEACHERS),
        "target_reader_evaluation_only": TARGET_READER,
        "label": "deletion supporting AND length-matched replacement supporting",
        "outer_folds": 5,
        "inner_folds": 4,
        "inner_objective": "mean per-question teacher-consensus NDCG@4 over questions with positive support",
        "candidate_models": list(MODELS),
        "candidate_utility_weights": list(ALPHAS),
        "features": names,
        "fold_choices": choices,
        "evaluation": evaluation,
        "base_plan_sha256": sha256(args.base_plan),
        "input_sha256": {str(path): sha256(path) for path in source_files},
        "output_sha256": sha256(args.output),
    }
    args.output.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
