"""E1/E2/E3 phase B (Spark). Token-budget sweep of frozen rankings on the fresh HF reader.

Plan rows: {question_group, dataset, qid, rankings{name: order}, fixed{name: idxs},
matched{name: [ranking, fixed_name]}}.  Selection rule and prompt are those of
run_efficiency_sweep.py (stable passage IDs, greedy skip, source order).  Every
answer, including the full-context reference, is generated in this run.
"""
from __future__ import annotations
import argparse, hashlib, json, random, time
from pathlib import Path
import datasets, numpy as np, torch, transformers
from run_efficiency_sweep import prompt, content_tokens, budget_select, generate, scored
from probe_loo import load_model, load_samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", default="e1_plan.jsonl"); ap.add_argument("--base-plan", default="fresh_selection_plan.jsonl")
    ap.add_argument("--output", default="results/e1_sweep.jsonl"); ap.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--budgets", default="128,256,384,512,768,1024"); ap.add_argument("--batch-size", type=int, default=24)
    ap.add_argument("--max-new-tokens", type=int, default=48); ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    budgets = [int(x) for x in a.budgets.split(",")]
    torch.manual_seed(0); np.random.seed(0); random.seed(0)
    base = {r["question_group"]: r for r in map(json.loads, Path(a.base_plan).read_text().splitlines())}
    plan = [json.loads(x) for x in Path(a.plan).read_text().splitlines() if x.strip()]
    if a.limit: plan = plan[:: max(1, len(plan) // a.limit)][: a.limit]
    samples = load_samples([base[p["question_group"]] for p in plan])
    tokenizer, model = load_model(a.model)
    out = Path(a.output); out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            try: done.add(json.loads(line)["question_group"])
            except Exception: pass
    t0 = time.time()
    with out.open("a") as stream:
        for pos, p in enumerate(plan, 1):
            g = p["question_group"]
            if g in done: continue
            s = samples[p["dataset"]][p["qid"]]; fr = s["fragments"]
            assert all(sorted(r) == list(range(len(fr))) for r in p["rankings"].values()), g
            memo = {}
            def ntok(idxs):
                k = tuple(idxs)
                if k not in memo: memo[k] = content_tokens(tokenizer, fr, list(idxs))
                return memo[k]
            def bsel(ranking, budget):
                sel = []
                for i in ranking:
                    if ntok(sorted(sel + [i])) <= budget: sel.append(i)
                if not sel: sel = [min(ranking, key=lambda i: ntok([i]))]
                return sorted(sel)
            sels = {"full": list(range(len(fr)))}
            for name, idxs in p.get("fixed", {}).items(): sels[name] = sorted(idxs)
            for name, r in p["rankings"].items():
                for b in budgets: sels[f"{name}_tok{b}"] = bsel(r, b)
            for name, (rname, fname) in p.get("matched", {}).items():
                sels[name] = bsel(p["rankings"][rname], ntok(sorted(p["fixed"][fname])))
            uniq = {}
            for idxs in sels.values(): uniq.setdefault(tuple(idxs), len(uniq))
            ordered = sorted(uniq, key=lambda k: -ntok(k))
            answers = generate(model, tokenizer, [prompt(tokenizer, s["question"], fr, list(k)) for k in ordered], a.batch_size, a.max_new_tokens)
            ans = dict(zip(ordered, answers))
            row = {"question_group": g, "dataset": p["dataset"], "qid": p["qid"], "golds": s["golds"], "n_unique": len(uniq),
                   "methods": {n: scored(ans[tuple(i)], s, i, tokenizer) for n, i in sels.items()}}
            stream.write(json.dumps(row, ensure_ascii=False) + "\n"); stream.flush()
            print(f"[{pos}/{len(plan)}] {g} unique={len(uniq)} {time.time()-t0:.0f}s", flush=True)
    meta = {"model": a.model, "model_commit": getattr(model.config, "_commit_hash", None), "budgets": budgets,
            "budget_unit": "tokenizer tokens in rendered context only", "stable_passage_ids": True,
            "selection_rule": "greedy rank order, skip candidates that exceed the budget, restore source order for reading",
            "plan_sha256": hashlib.sha256(Path(a.plan).read_bytes()).hexdigest(), "seed": 0, "torch": torch.__version__,
            "transformers": transformers.__version__, "datasets": datasets.__version__, "gpu": torch.cuda.get_device_name(0),
            "batch_size": a.batch_size, "finished_unix": time.time()}
    out.with_suffix(".metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
