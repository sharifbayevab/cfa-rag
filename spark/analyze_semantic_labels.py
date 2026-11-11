"""Repeat intervention-label analysis with a pinned embedding-equivalence rule."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from run_fresh_qa import normalize


def embed(texts, tokenizer, model, batch_size):
    vectors = []
    for start in range(0, len(texts), batch_size):
        encoded = tokenizer(
            texts[start:start + batch_size], padding=True, truncation=True,
            max_length=128, return_tensors="pt",
        )
        encoded = {k: v.to(model.device) for k, v in encoded.items()}
        with torch.inference_mode(): hidden = model(**encoded).last_hidden_state
        mask = encoded["attention_mask"].unsqueeze(-1)
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
        vectors.append(torch.nn.functional.normalize(pooled.float(), dim=1).cpu().numpy())
    return np.concatenate(vectors)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("results/intervention_controls.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("results/semantic_label_validation.json"))
    parser.add_argument("--model", default="BAAI/bge-base-en-v1.5")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    rows = [json.loads(x) for x in args.input.read_text().splitlines() if x.strip()]
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map="auto").eval()

    pair_records, texts = [], []
    for row in rows:
        alternatives = (
            [("delete", i, x) for i, x in enumerate(row["delete_answers"])]
            + [("mask", i, x) for i, x in enumerate(row["mask_answers"])]
            + [("joint_delete", -1, row["joint_gold_delete_answer"]),
               ("joint_mask", -1, row["joint_gold_mask_answer"])]
        )
        for kind, index, alternative in alternatives:
            pair_records.append({
                "question_group": row["question_group"], "dataset": row["dataset"],
                "kind": kind, "index": index, "full": row["full_answer"], "alternative": alternative,
                "string_change": normalize(row["full_answer"]) != normalize(alternative),
            })
            texts.extend([row["full_answer"] or "[empty]", alternative or "[empty]"])
    vectors = embed(texts, tokenizer, model, args.batch_size)
    for i, record in enumerate(pair_records):
        record["cosine"] = float(np.dot(vectors[2 * i], vectors[2 * i + 1]))

    result = {
        "n_questions": len(rows), "n_answer_pairs": len(pair_records), "model": args.model,
        "model_commit": getattr(model.config, "_commit_hash", None), "thresholds": {},
    }
    for threshold in [0.80, 0.85, 0.90]:
        for record in pair_records:
            record["semantic_change"] = bool(record["cosine"] < threshold and normalize(record["full"]) != normalize(record["alternative"]))
        string_changes = [r for r in pair_records if r["string_change"]]
        by_key = {(r["question_group"], r["kind"], r["index"]): r for r in pair_records}
        delete_mask_agreement = []
        for row in rows:
            for i in range(len(row["delete_answers"])):
                d = by_key[(row["question_group"], "delete", i)]["semantic_change"]
                m = by_key[(row["question_group"], "mask", i)]["semantic_change"]
                delete_mask_agreement.append(d == m)
        result["thresholds"][f"{threshold:.2f}"] = {
            "string_change_count": len(string_changes),
            "semantic_change_count": sum(r["semantic_change"] for r in pair_records),
            "string_change_precision_for_semantic_change": float(np.mean([r["semantic_change"] for r in string_changes])),
            "delete_mask_semantic_label_agreement": float(np.mean(delete_mask_agreement)),
            "delete_semantic_change_rate": float(np.mean([r["semantic_change"] for r in pair_records if r["kind"] == "delete"])),
            "mask_semantic_change_rate": float(np.mean([r["semantic_change"] for r in pair_records if r["kind"] == "mask"])),
            "joint_gold_mask_semantic_change_rate": float(np.mean([r["semantic_change"] for r in pair_records if r["kind"] == "joint_mask"])),
        }
    result["cosine_quantiles_for_string_changes"] = {
        str(q): float(np.quantile([r["cosine"] for r in pair_records if r["string_change"]], q))
        for q in [0.1, 0.25, 0.5, 0.75, 0.9]
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
