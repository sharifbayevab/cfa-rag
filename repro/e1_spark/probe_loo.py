"""E1 phase A (Spark). Exact leave-one-out CFA on the fresh HF reader with every
answer stored, plus position-preserving masks, joint-gold interventions,
closed-book, batch-noise duplicates and ContextCite-32 (stable passage IDs).

Runs next to run_fresh_qa.py / run_intervention_controls.py (copied unchanged
from experiments/spark).  One self-contained software stack: nothing from older
answer caches is reused.
"""
from __future__ import annotations

import argparse, hashlib, json, random, time
from pathlib import Path

import datasets, numpy as np, torch, transformers
from sklearn.linear_model import Lasso
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from transformers import AutoModelForCausalLM, AutoTokenizer

from run_fresh_qa import LOADERS, QA_INSTRUCTION, metrics, normalize, token_f1, official_masks, response_token_logits
from run_intervention_controls import (stable_context, rendered_prompt, prompt_ids, find_neutral_id,
                                       full_ids_and_spans, masked_ids)

CB_INSTRUCTION = ("Answer the question with ONLY the short answer (a few words, a name, a "
                  "number, or yes/no). Do not explain.\n\n")


def gen_ids(model, tokenizer, sequences, batch_size, max_new_tokens):
    answers, resp = [], []
    for start in range(0, len(sequences), batch_size):
        batch = sequences[start:start + batch_size]
        width = max(map(len, batch))
        ids = torch.tensor([[tokenizer.pad_token_id] * (width - len(x)) + x for x in batch], device=model.device)
        mask = torch.tensor([[0] * (width - len(x)) + [1] * len(x) for x in batch], device=model.device)
        with torch.inference_mode():
            out = model.generate(input_ids=ids, attention_mask=mask, max_new_tokens=max_new_tokens, do_sample=False,
                                 pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
        for row in out[:, width:]:
            resp.append(row.tolist())
        answers.extend(tokenizer.batch_decode(out[:, width:], skip_special_tokens=True))
    return [a.strip() for a in answers], resp


def load_model(name):
    tokenizer = AutoTokenizer.from_pretrained(name)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    try:
        model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.bfloat16)
    model = model.to("cuda")
    return tokenizer, model.eval()


def load_samples(plan):
    required = {d: {r["qid"] for r in plan if r["dataset"] == d} for d in LOADERS}
    samples = {}
    for d, qids in required.items():
        if qids:
            loaded = LOADERS[d](); samples[d] = {q: loaded[q] for q in qids}
    for p in plan:
        s = samples[p["dataset"]][p["qid"]]
        gold = [f["idx"] for f in s["fragments"] if f["is_gold"]]
        if s["question"] != p["question"] or s["golds"] != p["golds"] or gold != p["gold_idxs"] or len(s["fragments"]) != p["pool_size"]:
            raise ValueError(f"Frozen data mismatch for {p['question_group']}")
    return samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", default="fresh_selection_plan.jsonl")
    ap.add_argument("--output", default="results/e1_loo.jsonl")
    ap.add_argument("--pools", default="results/pools.jsonl")
    ap.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--batch-size", type=int, default=24)
    ap.add_argument("--max-new-tokens", type=int, default=48)
    ap.add_argument("--num-masks", type=int, default=32)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--no-mask", action="store_true", help="skip position-preserving masks (run later as a control)")
    ap.add_argument("--no-cc", action="store_true", help="skip in-stack ContextCite")
    a = ap.parse_args()
    torch.manual_seed(0); np.random.seed(0); random.seed(0)
    plan = [json.loads(x) for x in Path(a.plan).read_text().splitlines() if x.strip()]
    if a.limit:
        plan = plan[:: max(1, len(plan) // a.limit)][: a.limit]
    samples = load_samples(plan)
    tokenizer, model = load_model(a.model)
    neutral_id, neutral_text = find_neutral_id(tokenizer)
    out = Path(a.output); out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            try: done.add(json.loads(line)["question_group"])
            except Exception: pass
    t0 = time.time()
    with out.open("a") as stream, Path(a.pools).open("a") as pools:
        for pos, p in enumerate(plan, 1):
            group = p["question_group"]
            if group in done: continue
            s = samples[p["dataset"]][p["qid"]]; fr = s["fragments"]; m = len(fr); q = s["question"]
            gold = p["gold_idxs"]; allidx = set(range(m))
            full_ids, spans = full_ids_and_spans(tokenizer, q, fr)
            cb = tokenizer.apply_chat_template([{"role": "user", "content": f"{CB_INSTRUCTION}Question: {q}\nAnswer:"}],
                                               tokenize=False, add_generation_prompt=True)
            # canonical full answer: generated alone, no padding
            (full_answer,), (full_resp,) = gen_ids(model, tokenizer, [full_ids], 1, a.max_new_tokens)
            resp = [t for t in full_resp if t not in (tokenizer.eos_token_id, tokenizer.pad_token_id)] or [tokenizer.eos_token_id]
            dele = [prompt_ids(tokenizer, rendered_prompt(tokenizer, q, fr, allidx - {i})) for i in range(m)]
            mask = [] if a.no_mask else [masked_ids(full_ids, spans, [i], neutral_id) for i in range(m)]
            seqs = ([full_ids] + dele + mask +
                    [prompt_ids(tokenizer, rendered_prompt(tokenizer, q, fr, allidx - set(gold))),
                     masked_ids(full_ids, spans, gold, neutral_id), prompt_ids(tokenizer, cb), full_ids])
            ans, _ = gen_ids(model, tokenizer, seqs, a.batch_size, a.max_new_tokens)
            mm = len(mask)
            cc = {}
            if not a.no_cc:
                # ContextCite with stable passage ids
                masks = official_masks(a.num_masks, m)
                cc_prompts = [rendered_prompt(tokenizer, q, fr, set(np.where(mk)[0].tolist())) for mk in masks]
                tl = response_token_logits(model, tokenizer, cc_prompts, resp, 16)
                lsp = -np.logaddexp(0, -tl).sum(axis=1)
                seq_logit = lsp - np.log1p(-np.exp(np.minimum(lsp, -1e-12)))
                def cc_fit(n):
                    reg = make_pipeline(StandardScaler(), Lasso(alpha=0.01, random_state=0, fit_intercept=True))
                    with np.errstate(all="ignore"):
                        reg.fit(masks[:n].astype(np.float32), seq_logit[:n] / len(resp))
                    sc, la = reg.named_steps["standardscaler"], reg.named_steps["lasso"]
                    return ((la.coef_ / np.where(sc.scale_ == 0, 1, sc.scale_)) * len(resp)).tolist()
                cc = {"contextcite32": cc_fit(32), "contextcite16": cc_fit(16), "contextcite_seq_logit": seq_logit.tolist()}
            ptoks = [len(tokenizer.encode(stable_context(fr, {i}), add_special_tokens=False)) for i in range(m)]
            row = {
                "question_group": group, "dataset": p["dataset"], "qid": p["qid"], "question": q, "golds": s["golds"],
                "gold_idxs": gold, "fold": p["fold"], "pool_size": m,
                "full_answer": full_answer, "full_metrics": metrics(full_answer, s["golds"]),
                "full_answer_batched_first": ans[0], "full_answer_batched_last": ans[-1],
                "delete_answers": ans[1:1 + m], "mask_answers": ans[1 + m:1 + m + mm],
                "joint_gold_delete_answer": ans[1 + m + mm], "joint_gold_mask_answer": ans[2 + m + mm],
                "closed_book_answer": ans[3 + m + mm], "closed_book_metrics": metrics(ans[3 + m + mm], s["golds"]),
                "delete_metrics": [metrics(x, s["golds"]) for x in ans[1:1 + m]],
                "mask_metrics": [metrics(x, s["golds"]) for x in ans[1 + m:1 + m + mm]],
                "delete_f1_to_full": [token_f1(x, full_answer) for x in ans[1:1 + m]],
                "mask_f1_to_full": [token_f1(x, full_answer) for x in ans[1 + m:1 + m + mm]],
                "delete_changes": [normalize(x) != normalize(full_answer) for x in ans[1:1 + m]],
                "mask_changes": [normalize(x) != normalize(full_answer) for x in ans[1 + m:1 + m + mm]],
                "prompt_tokens": len(full_ids), "passage_tokens": ptoks,
                **cc, "response_tokens": len(resp),
            }
            stream.write(json.dumps(row, ensure_ascii=False) + "\n"); stream.flush()
            pools.write(json.dumps({"question_group": group, "question": q, "golds": s["golds"], "gold_idxs": gold,
                                    "fragments": [{"idx": f["idx"], "title": f["title"], "text": f["text"], "is_gold": f["is_gold"]} for f in fr],
                                    "passage_tokens": ptoks}, ensure_ascii=False) + "\n"); pools.flush()
            print(f"[{pos}/{len(plan)}] {group} m={m} full={full_answer!r} flips={sum(row['delete_changes'])} {time.time()-t0:.0f}s", flush=True)
    meta = {"model": a.model, "model_commit": getattr(model.config, "_commit_hash", None), "neutral_token_id": neutral_id,
            "neutral_token_probe": neutral_text, "stable_passage_ids": True, "seed": 0, "torch": torch.__version__,
            "transformers": transformers.__version__, "datasets": datasets.__version__, "gpu": torch.cuda.get_device_name(0),
            "plan_sha256": hashlib.sha256(Path(a.plan).read_bytes()).hexdigest(), "batch_size": a.batch_size,
            "max_new_tokens": a.max_new_tokens, "num_masks": a.num_masks, "finished_unix": time.time()}
    out.with_suffix(".metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
