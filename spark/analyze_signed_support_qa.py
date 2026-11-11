"""Analyze signed-support downstream QA with dataset-stratified paired CIs."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

METRICS = ("cover", "em", "f1", "context_words", "gold_recall")


def mean(rows, getter):
    return float(np.mean([getter(row) for row in rows]))


def paired_ci(rows, left, right, metric, rng, n_boot=10000):
    by_dataset = defaultdict(list)
    for row in rows:
        by_dataset[row["dataset"]].append(row)
    observed = mean(rows, lambda r: left(r)[metric] - right(r)[metric])
    draws = []
    for _ in range(n_boot):
        sample = []
        for population in by_dataset.values():
            sample.extend(population[i] for i in rng.integers(0, len(population), len(population)))
        draws.append(mean(sample, lambda r: left(r)[metric] - right(r)[metric]))
    return {"difference": observed, "ci95": [float(x) for x in np.percentile(draws, [2.5, 97.5])]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("results/signed_support_qa.jsonl"))
    parser.add_argument("--fresh-results", type=Path, default=Path("results/fresh_qa.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("results/signed_support_qa.analysis.json"))
    args = parser.parse_args()
    signed = [json.loads(x) for x in args.input.read_text().splitlines() if x.strip()]
    fresh = {r["question_group"]: r for r in map(json.loads, args.fresh_results.read_text().splitlines())}
    if len(signed) != 450 or len({r["question_group"] for r in signed}) != 450:
        raise ValueError(f"Expected 450 unique rows, found {len(signed)}")

    getters = {
        "signed_support": lambda r: r["methods"]["signed_support"],
        "signed_support_ce": lambda r: r["methods"]["signed_support_ce"],
        "cross_encoder": lambda r: fresh[r["question_group"]]["methods"]["cross_encoder_top4"],
        "unsigned_cfa": lambda r: fresh[r["question_group"]]["methods"]["oof_all9_top4"],
        "contextcite": lambda r: fresh[r["question_group"]]["methods"]["contextcite_top4"],
        "full": lambda r: fresh[r["question_group"]]["methods"]["full"],
    }
    result = {"n": len(signed), "overall": {}, "by_dataset": {}, "contrasts": {}}
    for name, getter in getters.items():
        result["overall"][name] = {m: mean(signed, lambda r, m=m: getter(r)[m]) for m in METRICS}
    for dataset in sorted({r["dataset"] for r in signed}):
        subset = [r for r in signed if r["dataset"] == dataset]
        result["by_dataset"][dataset] = {
            name: {m: mean(subset, lambda r, m=m: getter(r)[m]) for m in METRICS}
            for name, getter in getters.items()
        }
    rng = np.random.default_rng(0)
    for left in ("signed_support", "signed_support_ce"):
        for right in ("cross_encoder", "unsigned_cfa", "contextcite", "full"):
            key = f"{left}_minus_{right}"
            result["contrasts"][key] = {
                metric: paired_ci(signed, getters[left], getters[right], metric, rng)
                for metric in ("cover", "f1", "gold_recall")
            }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
