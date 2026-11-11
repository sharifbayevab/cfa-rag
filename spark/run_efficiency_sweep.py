"""Equal-token-budget and CFA-threshold sweep on the frozen fresh reader."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import datasets
import numpy as np
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from run_fresh_qa import LOADERS, QA_INSTRUCTION, metrics


def stable_context(fragments, indices):
    chosen = set(indices)
    return "\n".join(
        f"[{i + 1}] {fragment['title']}: {fragment['text']}"
        for i, fragment in enumerate(fragments) if i in chosen
    )


def prompt(tokenizer, question, fragments, indices):
    user = f"{QA_INSTRUCTION}Context:\n{stable_context(fragments, indices)}\n\nQuestion: {question}\nAnswer:"
    return tokenizer.apply_chat_template([{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True)


def content_tokens(tokenizer, fragments, indices):
    return len(tokenizer.encode(stable_context(fragments, indices), add_special_tokens=False))


def budget_select(tokenizer, fragments, ranking, budget):
    selected = []
    for index in ranking:
        candidate = sorted(selected + [index])
        if content_tokens(tokenizer, fragments, candidate) <= budget:
            selected.append(index)
    if not selected:
        selected = [min(ranking, key=lambda i: content_tokens(tokenizer, fragments, [i]))]
    return sorted(selected)


def generate(model, tokenizer, prompts, batch_size, max_new_tokens):
    answers = []
    tokenizer.padding_side = "left"
    for start in range(0, len(prompts), batch_size):
        encoded = tokenizer(prompts[start:start + batch_size], return_tensors="pt", padding=True, add_special_tokens=False)
        encoded = {k: v.to(model.device) for k, v in encoded.items()}
        width = encoded["input_ids"].shape[1]
        with torch.inference_mode():
            output = model.generate(
                **encoded, max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
            )
        answers.extend(tokenizer.batch_decode(output[:, width:], skip_special_tokens=True))
    return [x.strip() for x in answers]


def scored(answer, sample, indices, tokenizer, probes=0):
    gold = {f["idx"] for f in sample["fragments"] if f["is_gold"]}
    return {
        **metrics(answer, sample["golds"]), "answer": answer, "selected_idxs": indices,
        "n_selected": len(indices), "context_tokens": content_tokens(tokenizer, sample["fragments"], indices),
        "gold_recall": len(set(indices) & gold) / max(1, len(gold)), "probe_calls": probes,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, default=Path("efficiency_plan.jsonl"))
    parser.add_argument("--fresh-results", type=Path, default=Path("results/fresh_qa.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("results/efficiency_sweep.jsonl"))
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--budgets", default="256,512,768")
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    args = parser.parse_args()
    budgets = [int(x) for x in args.budgets.split(",")]
    torch.manual_seed(0); np.random.seed(0); random.seed(0)
    plan = [json.loads(x) for x in args.plan.read_text().splitlines() if x.strip()]
    fresh = {r["question_group"]: r for r in map(json.loads, args.fresh_results.read_text().splitlines())}
    required = {d: {r["qid"] for r in plan if r["dataset"] == d} for d in LOADERS}
    samples = {}
    for dataset, qids in required.items():
        if qids:
            loaded = LOADERS[dataset]()
            samples[dataset] = {qid: loaded[qid] for qid in qids}

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None: tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto", low_cpu_mem_usage=True
    ).eval()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    completed = set()
    if args.output.exists():
        for line in args.output.read_text().splitlines():
            try: completed.add(json.loads(line)["question_group"])
            except (json.JSONDecodeError, KeyError): pass

    with args.output.open("a") as stream:
        for position, planned in enumerate(plan, 1):
            group = planned["question_group"]
            if group in completed: continue
            sample = samples[planned["dataset"]][planned["qid"]]
            fragments = sample["fragments"]
            gold = [f["idx"] for f in fragments if f["is_gold"]]
            if sample["question"] != planned["question"] or sample["golds"] != planned["golds"] or gold != planned["gold_idxs"]:
                raise ValueError(f"Frozen data mismatch for {group}")

            selections = {"oof_calibrated": planned["calibrated_selection"]}
            for threshold, indices in planned["threshold_selections"].items():
                selections[f"oof_threshold_{threshold}"] = indices
            for budget in budgets:
                for method, ranking in planned["rankings"].items():
                    selections[f"{method}_tok{budget}"] = budget_select(tokenizer, fragments, ranking, budget)

            # Deduplicate identical selected contexts before generation.
            unique = {}
            for indices in selections.values(): unique.setdefault(tuple(indices), len(unique))
            ordered = [None] * len(unique)
            for indices, index in unique.items(): ordered[index] = list(indices)
            answers = generate(
                model, tokenizer,
                [prompt(tokenizer, sample["question"], fragments, indices) for indices in ordered],
                args.batch_size, args.max_new_tokens,
            )
            answer_of = {tuple(indices): answer for indices, answer in zip(ordered, answers)}
            methods = {}
            for name, indices in selections.items():
                probes = 33 if name.startswith("contextcite_") else 0
                methods[name] = scored(answer_of[tuple(indices)], sample, indices, tokenizer, probes)
            full = fresh[group]["methods"]["full"]
            row = {
                "question_group": group, "dataset": planned["dataset"], "qid": planned["qid"],
                "question": sample["question"], "golds": sample["golds"],
                "full": {k: full[k] for k in ["cover", "em", "f1", "answer"]},
                "calibrated_threshold": planned["calibrated_threshold"], "methods": methods,
            }
            stream.write(json.dumps(row, ensure_ascii=False) + "\n"); stream.flush()
            print(f"[{position}/{len(plan)}] {group} unique={len(unique)}", flush=True)

    metadata = {
        "model": args.model, "model_commit": getattr(model.config, "_commit_hash", None),
        "budgets": budgets, "budget_unit": "tokenizer tokens in rendered context only",
        "selection_rule": "greedy rank order, skip candidates that exceed the budget, restore source order for reading",
        "stable_passage_ids": True, "plan_sha256": hashlib.sha256(args.plan.read_bytes()).hexdigest(),
        "fresh_results_sha256": hashlib.sha256(args.fresh_results.read_bytes()).hexdigest(),
        "seed": 0, "torch": torch.__version__, "transformers": transformers.__version__,
        "datasets": datasets.__version__, "gpu": torch.cuda.get_device_name(0), "finished_unix": time.time(),
    }
    args.output.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
