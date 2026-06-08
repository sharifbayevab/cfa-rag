# -*- coding: utf-8 -*-
"""Re-summarize saved raw_*.jsonl cells with a consistent, fully-local
correctness metric (cover-EM: gold contained in the answer), recomputed from the
stored answers. This makes all cells comparable without the OpenAI judge.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

from exp import metrics
from exp.run_experiment import RESULTS, summarize


def rescore_row(r):
    golds = r["golds"]
    r["judge_full"] = metrics.answer_contains_gold(r["ans_full"], golds)
    r["judge_cb"] = metrics.answer_contains_gold(r["ans_cb"], golds)
    for st, v in r["selection"].items():
        v["judge"] = metrics.answer_contains_gold(v.get("answer", ""), golds)
    return r


def main():
    files = [f for f in glob.glob(str(RESULTS / "raw_*.jsonl")) if "test_" not in Path(f).name]
    for f in sorted(files):
        rows = [json.loads(l) for l in Path(f).read_text().splitlines() if l.strip()]
        # skip corrupted cells (many empty answers)
        empty = sum(1 for r in rows if not r["ans_full"].strip())
        if empty > 0.2 * len(rows):
            print(f"SKIP {Path(f).name}: {empty}/{len(rows)} empty (corrupted)")
            continue
        rows = [rescore_row(r) for r in rows]
        summ = summarize(rows)
        stem = Path(f).name[len("raw_"):-len(".jsonl")]
        (RESULTS / f"summary_{stem}.json").write_text(json.dumps(summ, indent=2), encoding="utf-8")
        # rewrite the raw file so the persisted judge_* fields equal cover-EM
        # (the *_judge keys are historical; no LLM judge is used)
        Path(f).write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows),
                           encoding="utf-8")
        print(f"rescored {stem}: acc_full={summ['acc_full_judge']} "
              f"acc_cb={summ['acc_closedbook_judge']} "
              f"param={summ['faithfulness_gap']['frac_correct_parametric']}")


if __name__ == "__main__":
    main()
