"""E5 on all 450 questions (Spark): position-preserving mask of every passage and of the
joint gold set, in the same stack as probe_loo.py.  Deletion answers are read from
results/e1_loo.jsonl; the full-context answer is regenerated and compared as a check."""
from __future__ import annotations
import json, random, time
from pathlib import Path
import numpy as np, torch
from run_fresh_qa import normalize, token_f1
from run_intervention_controls import find_neutral_id, full_ids_and_spans, masked_ids
from probe_loo import gen_ids, load_model, load_samples

torch.manual_seed(0); np.random.seed(0); random.seed(0)
plan = [json.loads(x) for x in Path("fresh_selection_plan.jsonl").read_text().splitlines() if x.strip()]
loo = {r["question_group"]: r for r in map(json.loads, Path("results/e1_loo.jsonl").read_text().splitlines())}
samples = load_samples(plan); tokenizer, model = load_model("Qwen/Qwen3-4B-Instruct-2507")
neutral_id, _ = find_neutral_id(tokenizer)
out = Path("results/e1_masks.jsonl"); done = set()
if out.exists():
    for line in out.read_text().splitlines():
        try: done.add(json.loads(line)["question_group"])
        except Exception: pass
t0 = time.time()
with out.open("a") as stream:
    for pos, p in enumerate(plan, 1):
        g = p["question_group"]
        if g in done: continue
        s = samples[p["dataset"]][p["qid"]]; fr = s["fragments"]; m = len(fr); L = loo[g]
        full_ids, spans = full_ids_and_spans(tokenizer, s["question"], fr)
        seqs = [full_ids] + [masked_ids(full_ids, spans, [i], neutral_id) for i in range(m)] + [masked_ids(full_ids, spans, p["gold_idxs"], neutral_id)]
        ans, _ = gen_ids(model, tokenizer, seqs, 24, 48)
        ref = L["full_answer"]
        row = {"question_group": g, "dataset": p["dataset"], "gold_idxs": p["gold_idxs"], "full_answer": ref, "full_answer_regenerated": ans[0],
               "delete_changes": L["delete_changes"], "mask_answers": ans[1:1 + m], "mask_changes": [normalize(x) != normalize(ref) for x in ans[1:1 + m]],
               "mask_f1_to_full": [token_f1(x, ref) for x in ans[1:1 + m]], "joint_gold_mask_answer": ans[1 + m],
               "joint_gold_delete_change": normalize(L["joint_gold_delete_answer"]) != normalize(ref), "joint_gold_mask_change": normalize(ans[1 + m]) != normalize(ref)}
        stream.write(json.dumps(row, ensure_ascii=False) + "\n"); stream.flush()
        print(f"[{pos}/{len(plan)}] {g} {time.time()-t0:.0f}s", flush=True)
print("DONE", flush=True)
