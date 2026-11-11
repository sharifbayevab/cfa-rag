"""Fresh downstream QA and faithful ContextCite on a held-out reader.

ContextCite follows the authors' implementation: Bernoulli-0.5 masks, token
logit probabilities of the fixed full-context response, sequence-probability
logit aggregation, StandardScaler + Lasso(alpha=0.01), and top-k selection.
The primary top-k budget equals the OOF selector budget for every question; a
fixed top-four sensitivity analysis is computed from the same model probes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import string
import time
from collections import Counter
from pathlib import Path

import datasets
import numpy as np
import sklearn
import torch
import transformers
from datasets import load_dataset
from sklearn.linear_model import Lasso
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from transformers import AutoModelForCausalLM, AutoTokenizer


QA_INSTRUCTION = (
    "Answer the question using the numbered context passages below. "
    "Reply with ONLY the short answer (a few words, a name, a number, or yes/no). "
    "Do not explain.\n\n"
)
ARTICLES = re.compile(r"\b(a|an|the)\b")
PUNCT = str.maketrans("", "", string.punctuation)


def normalize(text: str) -> str:
    return " ".join(ARTICLES.sub(" ", text.lower().translate(PUNCT)).split())


def exact_match(pred: str, gold: str) -> float:
    return float(normalize(pred) == normalize(gold))


def token_f1(pred: str, gold: str) -> float:
    p, g = normalize(pred).split(), normalize(gold).split()
    if not p or not g:
        return float(p == g)
    common = Counter(p) & Counter(g)
    overlap = sum(common.values())
    if not overlap:
        return 0.0
    precision, recall = overlap / len(p), overlap / len(g)
    return 2 * precision * recall / (precision + recall)


def metrics(pred: str, golds: list[str]) -> dict:
    n_pred = normalize(pred)
    return {
        "cover": float(any(normalize(g) and normalize(g) in n_pred for g in golds)),
        "em": max(exact_match(pred, g) for g in golds),
        "f1": max(token_f1(pred, g) for g in golds),
    }


def load_hotpot() -> dict[str, dict]:
    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")
    out = {}
    for ex in ds:
        ctx = ex["context"]
        gold_titles = set(ex["supporting_facts"]["title"])
        fragments = [
            {"idx": i, "title": title, "text": " ".join(sentences).strip(), "is_gold": title in gold_titles}
            for i, (title, sentences) in enumerate(zip(ctx["title"], ctx["sentences"]))
        ]
        out[str(ex["id"])] = {"question": ex["question"], "golds": [ex["answer"]], "fragments": fragments}
    return out


def load_2wiki() -> dict[str, dict]:
    error = None
    for name in ["scholarly-shadows-syndicate/2WikiMultihopQA_with_q_gpt35", "voidful/2WikiMultihopQA"]:
        try:
            ds = load_dataset(name, split="validation")
            break
        except Exception as exc:
            error = exc
    else:
        raise RuntimeError("Could not load 2Wiki") from error
    out = {}
    for i, ex in enumerate(ds):
        ctx = ex["context"]
        titles = ctx["title"] if isinstance(ctx, dict) else [c[0] for c in ctx]
        sentences = ctx.get("content", ctx.get("sentences")) if isinstance(ctx, dict) else [c[1] for c in ctx]
        sf = ex.get("supporting_facts", {})
        gold_titles = set(sf["title"]) if isinstance(sf, dict) else {s[0] for s in sf}
        fragments = [
            {"idx": j, "title": title, "text": " ".join(text) if isinstance(text, list) else str(text), "is_gold": title in gold_titles}
            for j, (title, text) in enumerate(zip(titles, sentences))
        ]
        # The historical cache records 2Wiki validation row numbers as qids.
        # Frozen question/gold/context checks below guard against dataset drift.
        out[str(i)] = {
            "question": ex["question"],
            "golds": [ex["answer"]],
            "fragments": fragments,
        }
    return out


def load_musique() -> dict[str, dict]:
    ds = load_dataset("dgslibisey/MuSiQue", split="validation")
    out = {}
    for i, ex in enumerate(ds):
        fragments = [
            {
                "idx": j,
                "title": paragraph.get("title", ""),
                "text": paragraph.get("paragraph_text", ""),
                "is_gold": bool(paragraph.get("is_supporting", False)),
            }
            for j, paragraph in enumerate(ex["paragraphs"])
        ]
        out[str(ex.get("id", i))] = {"question": ex["question"], "golds": [ex["answer"]], "fragments": fragments}
    return out


LOADERS = {"hotpotqa": load_hotpot, "2wiki": load_2wiki, "musique": load_musique}


def render_context(fragments: list[dict], indices: list[int]) -> str:
    return "\n".join(
        f"[{position}] {fragments[index]['title']}: {fragments[index]['text']}"
        for position, index in enumerate(indices, 1)
    )


def user_prompt(question: str, fragments: list[dict], indices: list[int]) -> str:
    return f"{QA_INSTRUCTION}Context:\n{render_context(fragments, indices)}\n\nQuestion: {question}\nAnswer:"


def chat_prompt(tokenizer, question: str, fragments: list[dict], indices: list[int]) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": user_prompt(question, fragments, indices)}],
        tokenize=False,
        add_generation_prompt=True,
    )


def generate_batch(model, tokenizer, prompts: list[str], max_new_tokens: int) -> tuple[list[str], list[list[int]]]:
    encoded = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False)
    encoded = {key: value.to(model.device) for key, value in encoded.items()}
    input_width = encoded["input_ids"].shape[1]
    with torch.inference_mode():
        output = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    response_ids = [row[input_width:].tolist() for row in output]
    answers = [tokenizer.decode(ids, skip_special_tokens=True).strip() for ids in response_ids]
    return answers, response_ids


def official_masks(num_masks: int, num_sources: int) -> np.ndarray:
    return np.stack(
        [np.random.RandomState(seed).choice([False, True], size=num_sources, p=[0.5, 0.5]) for seed in range(num_masks)]
    )


def response_token_logits(model, tokenizer, prompts: list[str], response_ids: list[int], batch_size: int) -> np.ndarray:
    values = []
    tokenizer.padding_side = "left"
    for start in range(0, len(prompts), batch_size):
        prompt_ids = [tokenizer.encode(p, add_special_tokens=False) for p in prompts[start : start + batch_size]]
        sequences = [p + response_ids for p in prompt_ids]
        width = max(map(len, sequences))
        input_ids, attention = [], []
        for sequence in sequences:
            pad = width - len(sequence)
            input_ids.append([tokenizer.pad_token_id] * pad + sequence)
            attention.append([0] * pad + [1] * len(sequence))
        ids = torch.tensor(input_ids, device=model.device)
        mask = torch.tensor(attention, device=model.device)
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            # Qwen3 can apply the language-model head only to the final states.
            # This is mathematically identical to slicing the full logits tensor,
            # while avoiding prompt_length x vocabulary logits that are unused.
            logits = model(
                input_ids=ids,
                attention_mask=mask,
                logits_to_keep=len(response_ids) + 1,
            ).logits[:, :-1]
        log_probs = torch.log_softmax(logits.float(), dim=-1)
        labels = torch.tensor(response_ids, device=model.device)[None, :, None].expand(len(sequences), -1, -1)
        correct_log_probs = log_probs.gather(-1, labels).squeeze(-1)
        token_probs = correct_log_probs.exp().clamp(max=1 - 1e-7)
        token_logits = correct_log_probs - torch.log1p(-token_probs)
        values.append(token_logits.cpu().numpy())
    return np.concatenate(values, axis=0)


def contextcite(model, tokenizer, question: str, fragments: list[dict], response_ids: list[int], num_masks: int, batch_size: int):
    masks = official_masks(num_masks, len(fragments))
    prompts = [
        chat_prompt(tokenizer, question, fragments, np.where(mask)[0].tolist())
        for mask in masks
    ]
    if not response_ids:
        response_ids = [tokenizer.eos_token_id]
    token_logits = response_token_logits(model, tokenizer, prompts, response_ids, batch_size)
    # Official aggregation: sum token log-probabilities, then take the logit of
    # the resulting sequence probability.  log(sigmoid(x)) = -softplus(-x).
    log_sequence_probability = -np.logaddexp(0, -token_logits).sum(axis=1)
    sequence_probability = np.exp(np.minimum(log_sequence_probability, -1e-12))
    sequence_logit = log_sequence_probability - np.log1p(-sequence_probability)
    regression = make_pipeline(StandardScaler(), Lasso(alpha=0.01, random_state=0, fit_intercept=True))
    regression.fit(masks.astype(np.float32), sequence_logit / len(response_ids))
    scaler, lasso = regression.named_steps["standardscaler"], regression.named_steps["lasso"]
    coefficients = (lasso.coef_ / scaler.scale_) * len(response_ids)
    return coefficients, len(response_ids)


def context_words(fragments: list[dict], indices: list[int]) -> int:
    return sum(len((fragments[i]["title"] + " " + fragments[i]["text"]).split()) for i in indices)


def score_method(answer: str, sample: dict, indices: list[int], probe_calls: int = 0) -> dict:
    gold = {f["idx"] for f in sample["fragments"] if f["is_gold"]}
    quality = metrics(answer, sample["golds"])
    return {
        **quality,
        "answer": answer,
        "selected_idxs": indices,
        "n_selected": len(indices),
        "context_words": context_words(sample["fragments"], indices),
        "gold_recall": len(set(indices) & gold) / max(1, len(gold)),
        "probe_calls": probe_calls,
    }


def summarize(rows: list[dict], seed: int = 0) -> dict:
    methods = list(rows[0]["methods"])
    summary = {"n": len(rows), "methods": {}}
    for method in methods:
        summary["methods"][method] = {
            key: float(np.mean([row["methods"][method][key] for row in rows]))
            for key in ["cover", "em", "f1", "n_selected", "context_words", "gold_recall", "probe_calls"]
        }
    rng = np.random.default_rng(seed)
    by_dataset = {d: [r for r in rows if r["dataset"] == d] for d in sorted({r["dataset"] for r in rows})}
    for method in methods:
        if method == "full":
            continue
        observed = float(np.mean([r["methods"][method]["cover"] - r["methods"]["full"]["cover"] for r in rows]))
        boot = []
        for _ in range(10000):
            sample = []
            for group in by_dataset.values():
                sample.extend(group[i] for i in rng.integers(0, len(group), len(group)))
            boot.append(np.mean([r["methods"][method]["cover"] - r["methods"]["full"]["cover"] for r in sample]))
        summary["methods"][method]["cover_delta_vs_full"] = observed
        summary["methods"][method]["cover_delta_ci95"] = [float(v) for v in np.percentile(boot, [2.5, 97.5])]
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", default="selection_plan.jsonl")
    parser.add_argument("--output", default="results/fresh_qa.jsonl")
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--num-masks", type=int, default=32)
    parser.add_argument("--mask-batch-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    plan_path = Path(args.plan)
    plan = [json.loads(line) for line in plan_path.read_text().splitlines() if line.strip()]
    if args.limit:
        plan = plan[: args.limit]

    required = {d: {row["qid"] for row in plan if row["dataset"] == d} for d in LOADERS}
    samples = {}
    for dataset, qids in required.items():
        if not qids:
            continue
        loaded = LOADERS[dataset]()
        missing = sorted(qids - loaded.keys())
        if missing:
            raise ValueError(f"Missing {len(missing)} planned qids in {dataset}: {missing[:5]}")
        samples[dataset] = {qid: loaded[qid] for qid in qids}
        print(f"loaded {dataset}: {len(qids)} planned questions", flush=True)

    # Fail before loading the model if a dataset revision changed the question,
    # answer, pool size, or supporting-passage order used by the frozen cache.
    for planned in plan:
        group = planned["question_group"]
        sample = samples[planned["dataset"]][planned["qid"]]
        fragments = sample["fragments"]
        loaded_gold = [fragment["idx"] for fragment in fragments if fragment["is_gold"]]
        if len(fragments) != planned["pool_size"]:
            raise ValueError(f"Pool-size mismatch for {group}: {len(fragments)} != {planned['pool_size']}")
        if sample["question"] != planned["question"] or sample["golds"] != planned["golds"]:
            raise ValueError(f"Question or gold-answer mismatch for {group}")
        if loaded_gold != planned["gold_idxs"]:
            raise ValueError(f"Gold-passage ordering mismatch for {group}: {loaded_gold} != {planned['gold_idxs']}")
    print(f"validated frozen dataset alignment: {len(plan)} questions", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
    ).eval()
    print(f"model ready: {args.model} on {model.device}", flush=True)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    completed, rows = set(), []
    if output.exists():
        for line in output.read_text().splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            rows.append(row)
            completed.add(row["question_group"])
        print(f"resume: {len(completed)} rows", flush=True)

    with output.open("a") as stream:
        for position, planned in enumerate(plan, 1):
            group = planned["question_group"]
            if group in completed:
                continue
            sample = samples[planned["dataset"]][planned["qid"]]
            fragments = sample["fragments"]
            all_indices = list(range(len(fragments)))
            full_prompt = chat_prompt(tokenizer, sample["question"], fragments, all_indices)
            full_answers, full_response_ids = generate_batch(
                model, tokenizer, [full_prompt], args.max_new_tokens
            )
            full_answer = full_answers[0]
            coefficients, response_tokens = contextcite(
                model,
                tokenizer,
                sample["question"],
                fragments,
                full_response_ids[0],
                args.num_masks,
                args.mask_batch_size,
            )
            cc_equal = sorted(np.argsort(-coefficients, kind="stable")[: planned["budget_k"]].tolist())
            cc_top4 = sorted(np.argsort(-coefficients, kind="stable")[: planned["fixed_budget_k"]].tolist())
            selection_order = [
                "oof_all9",
                "dense_equal_k",
                "cross_encoder_equal_k",
                "ce_rank_length_equal_k",
                "oof_all9_top4",
                "dense_top4",
                "cross_encoder_top4",
                "ce_rank_length_top4",
            ]
            selected = [planned["selections"][name] for name in selection_order] + [cc_equal, cc_top4]
            prompts = [chat_prompt(tokenizer, sample["question"], fragments, indices) for indices in selected]
            answers, _ = generate_batch(model, tokenizer, prompts, args.max_new_tokens)
            methods = {"full": score_method(full_answer, sample, all_indices)}
            for name, indices, answer in zip(selection_order, selected[:-2], answers[:-2]):
                methods[name] = score_method(answer, sample, indices)
            methods["contextcite_equal_k"] = score_method(
                answers[-2], sample, cc_equal, probe_calls=args.num_masks + 1
            )
            methods["contextcite_top4"] = score_method(
                answers[-1], sample, cc_top4, probe_calls=args.num_masks + 1
            )
            row = {
                "question_group": group,
                "dataset": planned["dataset"],
                "qid": planned["qid"],
                "fold": planned["fold"],
                "budget_k": planned["budget_k"],
                "question": sample["question"],
                "golds": sample["golds"],
                "methods": methods,
                "contextcite": {
                    "num_masks": args.num_masks,
                    "keep_probability": 0.5,
                    "lasso_alpha": 0.01,
                    "response_tokens": response_tokens,
                    "coefficients": coefficients.tolist(),
                    "mask_seed_rule": "numpy RandomState(seed), seed=0..num_masks-1",
                },
            }
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
            rows.append(row)
            print(f"[{position}/{len(plan)}] {group} full={methods['full']['cover']:.0f} oof={methods['oof_all9']['cover']:.0f} cc={methods['contextcite_equal_k']['cover']:.0f}", flush=True)

    summary = summarize(rows)
    summary["protocol"] = {
        "model": args.model,
        "model_commit": getattr(model.config, "_commit_hash", None),
        "plan_sha256": hashlib.sha256(plan_path.read_bytes()).hexdigest(),
        "num_masks": args.num_masks,
        "mask_batch_size": args.mask_batch_size,
        "lasso_alpha": 0.01,
        "seed": 0,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "datasets": datasets.__version__,
        "sklearn": sklearn.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "finished_unix": time.time(),
    }
    summary_path = output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
