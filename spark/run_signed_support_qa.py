"""Evaluate signed-support rankings on the frozen fresh reader.

Answers are reused from the original fresh-reader artifact whenever the exact
selected context is already present. Only previously unseen contexts are sent
to the reader. This is exact memoization because generation is deterministic
and the prompt is a function of the selected indices.
"""
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

from run_fresh_qa import LOADERS, chat_prompt, generate_batch, metrics


def scored(answer, sample, indices):
    gold = {f["idx"] for f in sample["fragments"] if f["is_gold"]}
    words = sum(len(sample["fragments"][i]["text"].split()) for i in indices)
    return {
        **metrics(answer, sample["golds"]),
        "answer": answer,
        "selected_idxs": indices,
        "n_selected": len(indices),
        "context_words": words,
        "gold_recall": len(set(indices) & gold) / max(1, len(gold)),
        "probe_calls": 0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, default=Path("signed_support_plan.jsonl"))
    parser.add_argument("--fresh-results", type=Path, default=Path("results/fresh_qa.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("results/signed_support_qa.jsonl"))
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    args = parser.parse_args()

    torch.manual_seed(0); np.random.seed(0); random.seed(0)
    plan = [json.loads(x) for x in args.plan.read_text().splitlines() if x.strip()]
    fresh = {r["question_group"]: r for r in map(json.loads, args.fresh_results.read_text().splitlines())}
    required = {d: {r["qid"] for r in plan if r["dataset"] == d} for d in LOADERS}
    samples = {}
    for dataset, qids in required.items():
        if qids:
            loaded = LOADERS[dataset]()
            samples[dataset] = {qid: loaded[qid] for qid in qids}

    # Determine whether any unseen prompt remains before loading the model.
    work = {}
    unseen_total = 0
    for row in plan:
        group = row["question_group"]
        known = {tuple(v["selected_idxs"]): v for v in fresh[group]["methods"].values()}
        selections = {
            name: sorted(row["rankings"][name][:4])
            for name in ("signed_support", "signed_support_ce")
        }
        unseen = {tuple(v) for v in selections.values() if tuple(v) not in known}
        unseen_total += len(unseen)
        work[group] = (known, selections)
    print(f"unseen deterministic contexts: {unseen_total}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = None
    if unseen_total:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.bfloat16, device_map="auto", low_cpu_mem_usage=True
        ).eval()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    completed = set()
    if args.output.exists():
        for line in args.output.read_text().splitlines():
            try: completed.add(json.loads(line)["question_group"])
            except (json.JSONDecodeError, KeyError): pass

    generated, reused = 0, 0
    with args.output.open("a") as stream:
        for position, row in enumerate(plan, 1):
            group = row["question_group"]
            if group in completed:
                continue
            sample = samples[row["dataset"]][row["qid"]]
            gold = [f["idx"] for f in sample["fragments"] if f["is_gold"]]
            if sample["question"] != row["question"] or sample["golds"] != row["golds"] or gold != row["gold_idxs"]:
                raise ValueError(f"Frozen data mismatch for {group}")
            known, selections = work[group]
            unseen = sorted({tuple(v) for v in selections.values() if tuple(v) not in known})
            answers = {}
            if unseen:
                generated_answers, _ = generate_batch(
                    model, tokenizer,
                    [chat_prompt(tokenizer, sample["question"], sample["fragments"], list(indices)) for indices in unseen],
                    args.max_new_tokens,
                )
                answers = dict(zip(unseen, generated_answers))
                generated += len(unseen)
            methods = {}
            for name, indices in selections.items():
                key = tuple(indices)
                if key in known:
                    methods[name] = dict(known[key])
                    methods[name]["reused"] = True
                    reused += 1
                else:
                    methods[name] = scored(answers[key], sample, indices)
                    methods[name]["reused"] = False
            output = {
                "question_group": group, "dataset": row["dataset"], "qid": row["qid"],
                "question": sample["question"], "golds": sample["golds"], "methods": methods,
            }
            stream.write(json.dumps(output, ensure_ascii=False) + "\n"); stream.flush()
            print(f"[{position}/{len(plan)}] {group} new={len(unseen)}", flush=True)

    metadata = {
        "model": args.model,
        "model_commit": getattr(model.config, "_commit_hash", None) if model is not None else None,
        "selection": "top four; restored to source order before rendering",
        "memoization": "exact selected-index match against frozen deterministic fresh-reader artifact",
        "new_generations_this_invocation": generated,
        "reused_method_results_this_invocation": reused,
        "plan_sha256": hashlib.sha256(args.plan.read_bytes()).hexdigest(),
        "fresh_results_sha256": hashlib.sha256(args.fresh_results.read_bytes()).hexdigest(),
        "seed": 0, "torch": torch.__version__, "transformers": transformers.__version__,
        "datasets": datasets.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "finished_unix": time.time(),
    }
    args.output.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
