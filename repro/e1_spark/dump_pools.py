"""Dump the frozen candidate pools (texts + Qwen tokenizer lengths) for offline use. CPU only."""
import json
from pathlib import Path
from transformers import AutoTokenizer
from run_intervention_controls import stable_context
from probe_loo import load_samples

plan = [json.loads(x) for x in Path("fresh_selection_plan.jsonl").read_text().splitlines() if x.strip()]
samples = load_samples(plan)
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct-2507")
with open("results/pools_all.jsonl", "w") as out:
    for p in plan:
        s = samples[p["dataset"]][p["qid"]]; fr = s["fragments"]
        out.write(json.dumps({"question_group": p["question_group"], "dataset": p["dataset"], "qid": p["qid"], "question": s["question"],
                              "golds": s["golds"], "gold_idxs": p["gold_idxs"], "fold": p["fold"],
                              "fragments": [{"idx": f["idx"], "title": f["title"], "text": f["text"], "is_gold": f["is_gold"]} for f in fr],
                              "passage_tokens": [len(tok.encode(stable_context(fr, {i}), add_special_tokens=False)) for i in range(len(fr))]},
                             ensure_ascii=False) + "\n")
print("ok", len(plan))
