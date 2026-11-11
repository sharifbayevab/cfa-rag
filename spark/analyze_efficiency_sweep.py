"""Analyze equal-token-budget and threshold sweeps with paired intervals."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

METRICS = ["cover", "em", "f1", "n_selected", "context_tokens", "gold_recall", "probe_calls"]


def means(rows, method):
    return {m: float(np.mean([r["methods"][method][m] for r in rows])) for m in METRICS}


def paired_ci(rows, left, right, metric, rng, n_boot=10000):
    groups = defaultdict(list)
    for row in rows: groups[row["dataset"]].append(row)
    values = [r["methods"][left][metric] - r["methods"][right][metric] for r in rows]
    draws = []
    for _ in range(n_boot):
        sample = []
        for group in groups.values(): sample.extend(group[i] for i in rng.integers(0, len(group), len(group)))
        draws.append(np.mean([r["methods"][left][metric] - r["methods"][right][metric] for r in sample]))
    return {"left": left, "right": right, "metric": metric, "difference": float(np.mean(values)),
            "ci95": [float(x) for x in np.percentile(draws, [2.5, 97.5])]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("results/efficiency_sweep.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("results/efficiency_sweep.analysis.json"))
    args = parser.parse_args()
    rows = [json.loads(x) for x in args.input.read_text().splitlines() if x.strip()]
    if len({r["question_group"] for r in rows}) != len(rows): raise ValueError("Duplicate rows")
    if len(rows) != 450: raise ValueError(f"Expected 450 questions, found {len(rows)}")
    methods = list(rows[0]["methods"])
    if any(list(r["methods"]) != methods for r in rows): raise ValueError("Method columns differ across rows")
    for row in rows:
        for method, values in row["methods"].items():
            if not values["selected_idxs"] or len(values["selected_idxs"]) != len(set(values["selected_idxs"])):
                raise ValueError(f"Invalid selection for {row['question_group']}/{method}")
            if "_tok" in method:
                budget = int(method.split("tok")[-1])
                if values["context_tokens"] > budget and values["n_selected"] != 1:
                    raise ValueError(f"Budget exceeded for {row['question_group']}/{method}")
            expected_probes = 33 if method.startswith("contextcite_") else 0
            if values["probe_calls"] != expected_probes:
                raise ValueError(f"Probe count mismatch for {row['question_group']}/{method}")
    result = {"n": len(rows), "overall": {m: means(rows, m) for m in methods}, "by_dataset": {}, "contrasts": []}
    for dataset in sorted({r["dataset"] for r in rows}):
        subset = [r for r in rows if r["dataset"] == dataset]
        result["by_dataset"][dataset] = {m: means(subset, m) for m in methods}
    rng = np.random.default_rng(0)
    budgets = sorted({int(m.split("tok")[-1]) for m in methods if "_tok" in m})
    for budget in budgets:
        left = f"oof_tok{budget}"
        for right in [f"dense_tok{budget}", f"cross_encoder_tok{budget}", f"ce_rank_length_tok{budget}", f"contextcite_tok{budget}"]:
            for metric in ["cover", "f1", "gold_recall"]:
                result["contrasts"].append(paired_ci(rows, left, right, metric, rng))
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
