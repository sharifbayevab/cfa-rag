"""Position-preserving and joint-deletion controls on a frozen sample.

The sample is selected before generation by SHA-256 order, independently in
each benchmark.  Every passage is evaluated twice: by deletion with stable
passage IDs and by replacing its exact prompt-token span with one repeated
neutral token.  The latter preserves prompt length and all later token
positions exactly.  The annotated supporting set is also intervened on jointly.
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

from run_fresh_qa import LOADERS, QA_INSTRUCTION, metrics, normalize, token_f1


def stable_context(fragments: list[dict], keep: set[int] | None = None) -> str:
    if keep is None:
        keep = set(range(len(fragments)))
    return "\n".join(
        f"[{i + 1}] {fragment['title']}: {fragment['text']}"
        for i, fragment in enumerate(fragments)
        if i in keep
    )


def rendered_prompt(tokenizer, question: str, fragments: list[dict], keep: set[int] | None = None) -> str:
    user = f"{QA_INSTRUCTION}Context:\n{stable_context(fragments, keep)}\n\nQuestion: {question}\nAnswer:"
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True
    )


def prompt_ids(tokenizer, prompt: str) -> list[int]:
    return tokenizer.encode(prompt, add_special_tokens=False)


def find_neutral_id(tokenizer) -> tuple[int, str]:
    for text in [" the", " x", "_", "."]:
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) == 1 and ids[0] not in tokenizer.all_special_ids:
            return ids[0], text
    raise RuntimeError("Could not find a one-token neutral replacement")


def full_ids_and_spans(tokenizer, question: str, fragments: list[dict]):
    prompt = rendered_prompt(tokenizer, question, fragments)
    encoded = tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)
    ids = list(encoded["input_ids"])
    offsets = encoded["offset_mapping"]
    spans = []
    cursor = 0
    for i, fragment in enumerate(fragments):
        segment = f"[{i + 1}] {fragment['title']}: {fragment['text']}"
        start = prompt.find(segment, cursor)
        if start < 0:
            raise ValueError(f"Could not locate rendered passage {i}")
        # Preserve the stable numeric passage identifier; mask title and text.
        content_start = start + len(f"[{i + 1}] ")
        end = start + len(segment)
        token_positions = [
            j for j, (left, right) in enumerate(offsets)
            if right > content_start and left < end
        ]
        if not token_positions:
            raise ValueError(f"Empty token span for passage {i}")
        spans.append(token_positions)
        cursor = end
    return ids, spans


def masked_ids(full_ids: list[int], spans: list[list[int]], indices: list[int], neutral_id: int) -> list[int]:
    out = list(full_ids)
    for index in indices:
        for position in spans[index]:
            out[position] = neutral_id
    return out


def generate_id_batches(model, tokenizer, sequences: list[list[int]], batch_size: int, max_new_tokens: int):
    answers = []
    tokenizer.padding_side = "left"
    for start in range(0, len(sequences), batch_size):
        batch = sequences[start : start + batch_size]
        width = max(map(len, batch))
        input_ids = [[tokenizer.pad_token_id] * (width - len(x)) + x for x in batch]
        attention = [[0] * (width - len(x)) + [1] * len(x) for x in batch]
        ids = torch.tensor(input_ids, device=model.device)
        mask = torch.tensor(attention, device=model.device)
        with torch.inference_mode():
            output = model.generate(
                input_ids=ids,
                attention_mask=mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        answers.extend(tokenizer.batch_decode(output[:, width:], skip_special_tokens=True))
    return [answer.strip() for answer in answers]


def select_plan(plan: list[dict], per_dataset: int) -> list[dict]:
    selected = []
    for dataset in sorted({row["dataset"] for row in plan}):
        rows = [row for row in plan if row["dataset"] == dataset]
        rows.sort(key=lambda row: hashlib.sha256(row["question_group"].encode()).hexdigest())
        selected.extend(rows[:per_dataset])
    return sorted(selected, key=lambda row: row["question_group"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", default="selection_plan.jsonl")
    parser.add_argument("--output", default="results/intervention_controls.jsonl")
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--per-dataset", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    args = parser.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    plan_path = Path(args.plan)
    plan = select_plan(
        [json.loads(line) for line in plan_path.read_text().splitlines() if line.strip()],
        args.per_dataset,
    )
    required = {d: {row["qid"] for row in plan if row["dataset"] == d} for d in LOADERS}
    samples = {}
    for dataset, qids in required.items():
        if qids:
            loaded = LOADERS[dataset]()
            samples[dataset] = {qid: loaded[qid] for qid in qids}

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    neutral_id, neutral_text = find_neutral_id(tokenizer)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto", low_cpu_mem_usage=True
    ).eval()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    completed = set()
    if output.exists():
        for line in output.read_text().splitlines():
            try:
                completed.add(json.loads(line)["question_group"])
            except (json.JSONDecodeError, KeyError):
                pass

    with output.open("a") as stream:
        for position, planned in enumerate(plan, 1):
            group = planned["question_group"]
            if group in completed:
                continue
            sample = samples[planned["dataset"]][planned["qid"]]
            fragments = sample["fragments"]
            loaded_gold = [f["idx"] for f in fragments if f["is_gold"]]
            if (sample["question"] != planned["question"] or sample["golds"] != planned["golds"]
                    or loaded_gold != planned["gold_idxs"] or len(fragments) != planned["pool_size"]):
                raise ValueError(f"Frozen data mismatch for {group}")

            full_ids, spans = full_ids_and_spans(tokenizer, sample["question"], fragments)
            all_indices = set(range(len(fragments)))
            deletion_ids = [
                prompt_ids(tokenizer, rendered_prompt(tokenizer, sample["question"], fragments, all_indices - {i}))
                for i in range(len(fragments))
            ]
            mask_ids = [masked_ids(full_ids, spans, [i], neutral_id) for i in range(len(fragments))]
            joint_delete = prompt_ids(
                tokenizer,
                rendered_prompt(tokenizer, sample["question"], fragments, all_indices - set(loaded_gold)),
            )
            joint_mask = masked_ids(full_ids, spans, loaded_gold, neutral_id)
            sequences = [full_ids] + deletion_ids + mask_ids + [joint_delete, joint_mask]
            answers = generate_id_batches(model, tokenizer, sequences, args.batch_size, args.max_new_tokens)
            full_answer = answers[0]
            m = len(fragments)
            delete_answers = answers[1 : 1 + m]
            mask_answers = answers[1 + m : 1 + 2 * m]
            row = {
                "question_group": group,
                "dataset": planned["dataset"],
                "qid": planned["qid"],
                "question": sample["question"],
                "golds": sample["golds"],
                "gold_idxs": loaded_gold,
                "full_answer": full_answer,
                "full_metrics": metrics(full_answer, sample["golds"]),
                "delete_answers": delete_answers,
                "mask_answers": mask_answers,
                "joint_gold_delete_answer": answers[-2],
                "joint_gold_mask_answer": answers[-1],
                "delete_changes": [normalize(x) != normalize(full_answer) for x in delete_answers],
                "mask_changes": [normalize(x) != normalize(full_answer) for x in mask_answers],
                "delete_f1_to_full": [token_f1(x, full_answer) for x in delete_answers],
                "mask_f1_to_full": [token_f1(x, full_answer) for x in mask_answers],
                "joint_gold_delete_change": normalize(answers[-2]) != normalize(full_answer),
                "joint_gold_mask_change": normalize(answers[-1]) != normalize(full_answer),
                "prompt_tokens": len(full_ids),
                "masked_span_tokens": [len(span) for span in spans],
            }
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
            print(f"[{position}/{len(plan)}] {group} full={full_answer!r}", flush=True)

    metadata = {
        "model": args.model,
        "model_commit": getattr(model.config, "_commit_hash", None),
        "sample_rule": f"first {args.per_dataset} SHA-256-sorted question IDs per dataset",
        "sample_size": len(plan),
        "plan_sha256": hashlib.sha256(plan_path.read_bytes()).hexdigest(),
        "neutral_token_id": neutral_id,
        "neutral_token_probe": neutral_text,
        "position_control": "replace exact title+text token positions; preserve passage IDs and prompt length",
        "seed": 0,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "datasets": datasets.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "finished_unix": time.time(),
    }
    output.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
