# -*- coding: utf-8 -*-
"""Run a matrix of (dataset, generator, n) cells sequentially, saving each
summary. API cells are fast; local Ollama cells run one model at a time.
"""
from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path

from exp.run_experiment import RESULTS, run_cell, summarize


def cell(dataset, generator, n, seed=13, k=4, use_judge=False, tag=""):
    t0 = time.time()
    try:
        rows = run_cell(dataset, generator, n, seed, k, use_judge=use_judge, use_semantic=True)
    except Exception as e:
        print(f"!! FAILED {dataset}/{generator}: {e}")
        traceback.print_exc()
        return
    summ = summarize(rows)
    stem = f"{(tag+'_') if tag else ''}{dataset}_{generator.replace(':','-')}_n{n}"
    (RESULTS / f"raw_{stem}.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    (RESULTS / f"summary_{stem}.json").write_text(json.dumps(summ, indent=2), encoding="utf-8")
    fg = summ["faithfulness_gap"]; rc = summ["relevance_vs_causality"]
    print(f"DONE {stem} in {time.time()-t0:.0f}s | acc_full={summ['acc_full_judge']} "
          f"acc_cb={summ['acc_closedbook_judge']} parametric={fg['frac_correct_parametric']} "
          f"rel~causal_jacc={rc['top_rel_causal_jaccard']}", flush=True)


# default matrix
MATRIX = {
    "api": [
        ("hotpotqa", "openai:gpt-4.1-mini", 200),
        ("2wiki", "openai:gpt-4.1-mini", 200),
        ("musique", "openai:gpt-4.1-mini", 200),
        ("hotpotqa", "openai:gpt-4.1-nano", 200),
        ("2wiki", "openai:gpt-4.1-nano", 200),
        ("musique", "openai:gpt-4.1-nano", 200),
    ],
    "local": [
        ("hotpotqa", "ollama:gemma3:4b", 100),
        ("2wiki", "ollama:gemma3:4b", 100),
        ("musique", "ollama:gemma3:4b", 100),
        ("hotpotqa", "ollama:gemma3:12b", 80),
        ("2wiki", "ollama:gemma3:12b", 80),
    ],
    # trimmed: complete the capability gradient (run alone, no API contention)
    "local2": [
        ("hotpotqa", "ollama:gemma3:12b", 60),
        ("2wiki", "ollama:gemma3:12b", 60),
        ("musique", "ollama:gemma3:4b", 80),
    ],
    # fully-local gradient extension (cover-EM correctness, no API/judge)
    "local3": [
        ("hotpotqa", "ollama:gemma3:12b", 80),
        ("2wiki", "ollama:gemma3:12b", 80),
        ("musique", "ollama:gemma3:12b", 80),
        ("hotpotqa", "ollama:gemma3:27b", 60),
        ("2wiki", "ollama:gemma3:27b", 60),
        ("musique", "ollama:gemma3:4b", 100),
    ],
}


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "api"
    cells = MATRIX[which]
    print(f"=== MATRIX [{which}]: {len(cells)} cells ===", flush=True)
    for ds, gen, n in cells:
        print(f"\n--- {ds} | {gen} | n={n} ---", flush=True)
        cell(ds, gen, n)
    print("\n=== MATRIX DONE ===", flush=True)
