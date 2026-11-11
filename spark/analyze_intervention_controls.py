"""Summarize position-preserving and joint-deletion controls."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def clustered_ci(rows, statistic, rng, n_boot=10000):
    observed = float(statistic(rows))
    groups = defaultdict(list)
    for row in rows:
        groups[row["dataset"]].append(row)
    draws = []
    for _ in range(n_boot):
        sample = []
        for group in groups.values():
            sample.extend(group[i] for i in rng.integers(0, len(group), len(group)))
        value = statistic(sample)
        if np.isfinite(value):
            draws.append(value)
    return {"mean": observed, "ci95": [float(x) for x in np.percentile(draws, [2.5, 97.5])]}


def summarize(rows):
    rng = np.random.default_rng(0)
    result = {"n": len(rows), "datasets": {}, "overall": {}}
    subsets = {"overall": rows, "full_answer_correct": [r for r in rows if r["full_metrics"]["cover"]]}
    subsets.update({d: [r for r in rows if r["dataset"] == d] for d in sorted({r["dataset"] for r in rows})})
    for name, subset in subsets.items():
        fragment_pairs = lambda rs: [(d, m) for r in rs for d, m in zip(r["delete_changes"], r["mask_changes"])]
        def rate(values): return float(np.mean(list(values)))
        stats = {
            "delete_change_rate": lambda rs: rate(float(d) for d, _ in fragment_pairs(rs)),
            "position_preserving_mask_change_rate": lambda rs: rate(float(m) for _, m in fragment_pairs(rs)),
            "delete_mask_label_agreement": lambda rs: rate(float(d == m) for d, m in fragment_pairs(rs)),
            "delete_positive_confirmed_by_mask": lambda rs: rate(float(m) for d, m in fragment_pairs(rs) if d),
            "exact_sensitive_set_match": lambda rs: rate(float(r["delete_changes"] == r["mask_changes"]) for r in rs),
            "single_loo_empty_rate": lambda rs: rate(float(not any(r["delete_changes"])) for r in rs),
            "empty_loo_but_joint_gold_mask_changes": lambda rs: rate(float((not any(r["delete_changes"])) and r["joint_gold_mask_change"]) for r in rs),
            "joint_gold_mask_change_rate": lambda rs: rate(float(r["joint_gold_mask_change"]) for r in rs),
            "joint_gold_delete_change_rate": lambda rs: rate(float(r["joint_gold_delete_change"]) for r in rs),
            "joint_only_beyond_single_gold_mask": lambda rs: rate(float((not any(r["mask_changes"][i] for i in r["gold_idxs"])) and r["joint_gold_mask_change"]) for r in rs),
        }
        result["overall" if name == "overall" else "datasets"].setdefault(name, {})
        target = result["overall"] if name == "overall" else result["datasets"][name]
        target.update({
            "n_questions": len(subset),
            "n_fragment_pairs": len(fragment_pairs(subset)),
            **{key: clustered_ci(subset, statistic, rng) for key, statistic in stats.items()},
        })
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("results/intervention_controls.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("results/intervention_controls.analysis.json"))
    args = parser.parse_args()
    rows = [json.loads(x) for x in args.input.read_text().splitlines() if x.strip()]
    groups = [r["question_group"] for r in rows]
    if len(groups) != len(set(groups)):
        raise ValueError("Duplicate question groups")
    result = summarize(rows)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
