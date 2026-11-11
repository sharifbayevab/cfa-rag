"""Validate the fresh QA run and emit auditable summaries and LaTeX rows."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


ADAPTIVE = [
    "full",
    "oof_all9",
    "dense_equal_k",
    "cross_encoder_equal_k",
    "ce_rank_length_equal_k",
    "contextcite_equal_k",
]
TOP4 = [
    "full",
    "oof_all9_top4",
    "dense_top4",
    "cross_encoder_top4",
    "ce_rank_length_top4",
    "contextcite_top4",
]
LABELS = {
    "full": "Full context",
    "dense_equal_k": "Dense",
    "cross_encoder_equal_k": "Cross-encoder",
    "ce_rank_length_equal_k": "CE rank + length",
    "oof_all9": "OOF CFA selector",
    "contextcite_equal_k": "ContextCite",
    "dense_top4": "Dense",
    "cross_encoder_top4": "Cross-encoder",
    "ce_rank_length_top4": "CE rank + length",
    "oof_all9_top4": "OOF CFA selector",
    "contextcite_top4": "ContextCite",
}
METRICS = ["cover", "em", "f1", "n_selected", "context_words", "gold_recall", "probe_calls"]


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    for line_no, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Malformed JSON at {path}:{line_no}") from exc
    return rows


def mean_metrics(rows: list[dict], method: str) -> dict:
    return {key: float(np.mean([row["methods"][method][key] for row in rows])) for key in METRICS}


def stratified_paired_ci(rows: list[dict], left: str, right: str, metric: str, rng, n_boot: int = 10000) -> dict:
    groups = defaultdict(list)
    for row in rows:
        groups[row["dataset"]].append(row)
    observed = float(np.mean([
        row["methods"][left][metric] - row["methods"][right][metric] for row in rows
    ]))
    draws = np.empty(n_boot)
    for draw in range(n_boot):
        differences = []
        for group in groups.values():
            indices = rng.integers(0, len(group), len(group))
            differences.extend(
                group[i]["methods"][left][metric] - group[i]["methods"][right][metric]
                for i in indices
            )
        draws[draw] = np.mean(differences)
    return {
        "left": left,
        "right": right,
        "metric": metric,
        "difference": observed,
        "ci95": [float(x) for x in np.percentile(draws, [2.5, 97.5])],
    }


def validate(plan: list[dict], rows: list[dict]) -> None:
    plan_by_group = {row["question_group"]: row for row in plan}
    counts = Counter(row["question_group"] for row in rows)
    duplicates = sorted(group for group, count in counts.items() if count != 1)
    if duplicates:
        raise ValueError(f"Duplicate result groups: {duplicates[:5]}")
    if set(counts) != set(plan_by_group):
        missing = sorted(set(plan_by_group) - set(counts))
        extra = sorted(set(counts) - set(plan_by_group))
        raise ValueError(f"Result/plan mismatch: missing={missing[:5]}, extra={extra[:5]}")

    for row in rows:
        group = row["question_group"]
        planned = plan_by_group[group]
        if row["dataset"] != planned["dataset"] or row["qid"] != planned["qid"]:
            raise ValueError(f"Identity mismatch for {group}")
        if row["question"] != planned["question"] or row["golds"] != planned["golds"]:
            raise ValueError(f"Frozen text mismatch for {group}")
        if row["contextcite"]["num_masks"] != 32 or row["contextcite"]["lasso_alpha"] != 0.01:
            raise ValueError(f"ContextCite protocol mismatch for {group}")
        if len(row["contextcite"]["coefficients"]) != planned["pool_size"]:
            raise ValueError(f"ContextCite coefficient count mismatch for {group}")
        for method, indices in planned["selections"].items():
            if row["methods"][method]["selected_idxs"] != indices:
                raise ValueError(f"Frozen selection mismatch for {group}/{method}")
        for method in ["oof_all9", "dense_equal_k", "cross_encoder_equal_k", "ce_rank_length_equal_k", "contextcite_equal_k"]:
            if row["methods"][method]["n_selected"] != planned["budget_k"]:
                raise ValueError(f"Adaptive budget mismatch for {group}/{method}")
        for method in ["oof_all9_top4", "dense_top4", "cross_encoder_top4", "ce_rank_length_top4", "contextcite_top4"]:
            if row["methods"][method]["n_selected"] != planned["fixed_budget_k"]:
                raise ValueError(f"Top-four budget mismatch for {group}/{method}")


def latex_panel(title: str, methods: list[str], summary: dict) -> list[str]:
    lines = [f"\\multicolumn{{8}}{{l}}{{\\textit{{{title}}}}}\\\\", "\\addlinespace[2pt]"]
    for method in methods:
        m = summary["overall"][method]
        probes = "--" if method == "full" else f"{m['probe_calls']:.0f}"
        label = f"\\textbf{{{LABELS[method]}}}" if method.startswith("oof_all9") else LABELS[method]
        gold_recall = f"{100*m['gold_recall']:.1f}"
        if method == "oof_all9_top4":
            gold_recall = f"\\textbf{{{gold_recall}}}"
        lines.append(
            f"{label} & {m['n_selected']:.2f} & {100*m['cover']:.1f} & "
            f"{100*m['em']:.1f} & {100*m['f1']:.1f} & {m['context_words']:.0f} & "
            f"{gold_recall} & {probes} \\\\"
        )
    return lines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, default=Path("selection_plan.jsonl"))
    parser.add_argument("--results", type=Path, default=Path("results/fresh_qa.jsonl"))
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    parser.add_argument(
        "--latex-dir",
        type=Path,
        help="Optional manuscript directory for tab_fresh_qa.tex and freshmacros.tex",
    )
    parser.add_argument(
        "--batch16-rows",
        type=int,
        default=0,
        help="Number of initial checkpoint rows executed with mask batch 16",
    )
    parser.add_argument(
        "--full-logit-rows",
        type=int,
        default=0,
        help="Initial rows that materialized prompt-wide logits before final-state optimization",
    )
    parser.add_argument("--cache-disabled-start", type=int, default=0)
    parser.add_argument("--cache-disabled-end", type=int, default=0)
    args = parser.parse_args()

    plan, rows = read_jsonl(args.plan), read_jsonl(args.results)
    validate(plan, rows)
    rows.sort(key=lambda row: row["question_group"])
    methods = list(rows[0]["methods"])
    summary = {
        "n_questions": len(rows),
        "dataset_counts": dict(sorted(Counter(row["dataset"] for row in rows).items())),
        "plan_sha256": hashlib.sha256(args.plan.read_bytes()).hexdigest(),
        "results_sha256": hashlib.sha256(args.results.read_bytes()).hexdigest(),
        "overall": {method: mean_metrics(rows, method) for method in methods},
        "by_dataset": {},
    }
    for dataset in sorted({row["dataset"] for row in rows}):
        subset = [row for row in rows if row["dataset"] == dataset]
        summary["by_dataset"][dataset] = {method: mean_metrics(subset, method) for method in methods}

    rng = np.random.default_rng(0)
    contrasts = []
    for metric in ["cover", "f1"]:
        for left, right in [
            ("oof_all9", "full"),
            ("oof_all9", "dense_equal_k"),
            ("oof_all9", "cross_encoder_equal_k"),
            ("oof_all9", "ce_rank_length_equal_k"),
            ("oof_all9", "contextcite_equal_k"),
            ("contextcite_equal_k", "full"),
            ("oof_all9_top4", "full"),
            ("oof_all9_top4", "dense_top4"),
            ("oof_all9_top4", "cross_encoder_top4"),
            ("oof_all9_top4", "ce_rank_length_top4"),
            ("oof_all9_top4", "contextcite_top4"),
            ("contextcite_top4", "full"),
        ]:
            contrasts.append(stratified_paired_ci(rows, left, right, metric, rng))
    summary["paired_stratified_bootstrap"] = contrasts
    summary["execution"] = {
        "reader": "Qwen/Qwen3-4B-Instruct-2507",
        "reader_commit": "cdbee75f17c01a7cc42f958dc650907174af0554",
        "gpu": "NVIDIA GB10",
        "container": "nvcr.io/nvidia/pytorch:25.09-py3@sha256:f3b4a33fd60d5e0358287bb44d57e91eda45e8f1681fef1bfd6c969371417130",
        "software": {
            "torch": "2.9.0a0+50eac811a6.nv25.09",
            "transformers": "4.57.6",
            "datasets": "4.8.5",
            "scikit_learn": "1.7.1",
        },
        "mask_batches": (
            [
                {"rows": [1, args.batch16_rows], "batch_size": 16},
                {"rows": [args.batch16_rows + 1, len(rows)], "batch_size": 32},
            ]
            if args.batch16_rows
            else [{"rows": [1, len(rows)], "batch_size": 32}]
        ),
        "lm_head_execution": (
            [
                {"rows": [1, args.full_logit_rows], "mode": "prompt-wide logits, then response slice"},
                {"rows": [args.full_logit_rows + 1, len(rows)], "mode": "identical final-state logits only"},
            ]
            if args.full_logit_rows
            else [{"rows": [1, len(rows)], "mode": "identical final-state logits only"}]
        ),
        "teacher_forcing_cache": (
            [
                {"rows": [1, args.cache_disabled_start - 1], "use_cache": True},
                {"rows": [args.cache_disabled_start, args.cache_disabled_end], "use_cache": False},
                {"rows": [args.cache_disabled_end + 1, len(rows)], "use_cache": True},
            ]
            if args.cache_disabled_start and args.cache_disabled_end
            else [{"rows": [1, len(rows)], "use_cache": True}]
        ),
        "note": (
            "Batching, final-state LM-head evaluation, and the unused KV cache change execution only; "
            "all rows use the same model, prompts, fixed responses, and 32 deterministic masks."
        ),
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    latex_dir = args.latex_dir or args.out_dir
    latex_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "fresh_qa_analysis.json").write_text(json.dumps(summary, indent=2) + "\n")

    table = [
        "\\begin{tabular}{lrrrrrrr}",
        "\\toprule",
        "Method & Ret. & Cover & EM & $F_1$ & W & GR & Probes \\\\",
        "\\midrule",
        *latex_panel("Adaptive OOF-selector budget", ADAPTIVE, summary),
        "\\midrule",
        *latex_panel("Fixed four-passage budget", TOP4, summary),
        "\\bottomrule",
        "\\end{tabular}",
    ]
    (latex_dir / "tab_fresh_qa.tex").write_text("\n".join(table) + "\n")

    lookup = {(c["left"], c["right"], c["metric"]): c for c in contrasts}
    adaptive = lookup[("oof_all9", "cross_encoder_equal_k", "cover")]
    top4 = lookup[("oof_all9_top4", "cross_encoder_top4", "cover")]
    adaptive_dense = lookup[("oof_all9", "dense_equal_k", "cover")]
    top4_dense = lookup[("oof_all9_top4", "dense_top4", "cover")]
    top4_ce_length = lookup[("oof_all9_top4", "ce_rank_length_top4", "cover")]
    adaptive_cc = lookup[("oof_all9", "contextcite_equal_k", "cover")]
    top4_cc = lookup[("oof_all9_top4", "contextcite_top4", "cover")]
    macros = {
        "freshAdaptiveBudget": f"{summary['overall']['oof_all9']['n_selected']:.2f}",
        "freshOOFCover": f"{100*summary['overall']['oof_all9']['cover']:.1f}",
        "freshOOFTopCover": f"{100*summary['overall']['oof_all9_top4']['cover']:.1f}",
        "freshOOFTopGoldRecall": f"{100*summary['overall']['oof_all9_top4']['gold_recall']:.1f}",
        "freshCETopCover": f"{100*summary['overall']['cross_encoder_top4']['cover']:.1f}",
        "freshCCCover": f"{100*summary['overall']['contextcite_equal_k']['cover']:.1f}",
        "freshCCTopCover": f"{100*summary['overall']['contextcite_top4']['cover']:.1f}",
        "freshAdaptiveDeltaDense": f"{100*adaptive_dense['difference']:+.1f}",
        "freshAdaptiveDeltaDenseLo": f"{100*adaptive_dense['ci95'][0]:+.1f}",
        "freshAdaptiveDeltaDenseHi": f"{100*adaptive_dense['ci95'][1]:+.1f}",
        "freshAdaptiveDeltaCE": f"{100*adaptive['difference']:+.1f}",
        "freshAdaptiveDeltaCELo": f"{100*adaptive['ci95'][0]:+.1f}",
        "freshAdaptiveDeltaCEHi": f"{100*adaptive['ci95'][1]:+.1f}",
        "freshTopDeltaCE": f"{100*top4['difference']:+.1f}",
        "freshTopDeltaCELo": f"{100*top4['ci95'][0]:+.1f}",
        "freshTopDeltaCEHi": f"{100*top4['ci95'][1]:+.1f}",
        "freshTopDeltaDense": f"{100*top4_dense['difference']:+.1f}",
        "freshTopDeltaDenseLo": f"{100*top4_dense['ci95'][0]:+.1f}",
        "freshTopDeltaDenseHi": f"{100*top4_dense['ci95'][1]:+.1f}",
        "freshTopDeltaCELength": f"{100*top4_ce_length['difference']:+.1f}",
        "freshTopDeltaCELengthLo": f"{100*top4_ce_length['ci95'][0]:+.1f}",
        "freshTopDeltaCELengthHi": f"{100*top4_ce_length['ci95'][1]:+.1f}",
        "freshAdaptiveDeltaCC": f"{100*adaptive_cc['difference']:+.1f}",
        "freshAdaptiveDeltaCCLo": f"{100*adaptive_cc['ci95'][0]:+.1f}",
        "freshAdaptiveDeltaCCHi": f"{100*adaptive_cc['ci95'][1]:+.1f}",
        "freshTopDeltaCC": f"{100*top4_cc['difference']:+.1f}",
        "freshTopDeltaCCLo": f"{100*top4_cc['ci95'][0]:+.1f}",
        "freshTopDeltaCCHi": f"{100*top4_cc['ci95'][1]:+.1f}",
    }
    (latex_dir / "freshmacros.tex").write_text(
        "% Generated by experiments/spark/analyze_results.py\n"
        + "\n".join(f"\\newcommand{{\\{key}}}{{{value}}}" for key, value in macros.items())
        + "\n"
    )
    print(json.dumps({"status": "ok", **summary, "macros": macros}, indent=2))


if __name__ == "__main__":
    main()
