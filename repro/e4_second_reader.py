"""E4. Second reader without new probes (API): GPT-4.1-mini or Gemini-3.5-flash-lite.

The reader's own leave-one-out answers come from the frozen signed grid; the
transferred signal is the consensus of the four *other* generators of the paper
grid.  Only final answers are generated here (plus the full-context reference).
Prompt: template A of the grid with renumbered passages, temperature 0, 48 tokens.
Budgets are o200k_base tokens of the rendered context.

  python3 repro/e4_second_reader.py --reader gpt --estimate     # cost estimate only
  python3 repro/e4_second_reader.py --reader gemini             # run (resumable cache)
For Gemini the o200k_base count is a proxy tokenizer used identically for every arm.
"""
import argparse, json, os, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import tiktoken

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from cfa_lib import (BUDGETS, GRID_GENS, budget_select, exact_cfa_score, jl, load_features, load_pools, load_signed, metrics, rank)

CFG = {"gpt": ("gpt-4.1-mini", "openai-gpt-4.1-mini", "e4_gpt41mini", 0.40, 1.60),
       "gemini": ("gemini-3.5-flash-lite", "gemini-gemini-3.5-flash-lite", "e4_gemini35fl", 0.10, 0.40)}
QA = ("Answer the question using the numbered context passages below. "
      "Reply with ONLY the short answer (a few words, a name, a number, or yes/no). Do not explain.\n\n")
enc = tiktoken.get_encoding("o200k_base")


def render(fr, idxs): return "\n".join(f"[{n}] {fr[i]['title']}: {fr[i]['text']}" for n, i in enumerate(idxs, 1))
def prompt(q, fr, idxs): return f"{QA}Context:\n{render(fr, idxs)}\n\nQuestion: {q}\nAnswer:"


def build(READER):
    pools = load_pools(); feats, labels, names = load_features(); own = load_signed(READER)
    bge = json.load(open(HERE / "outputs/e3_bge_reranker_scores.json"))["scores"]
    ce_i, rel_i = names.index("ce"), names.index("rel")
    plan = {}
    for q, p in pools.items():
        fr = p["fragments"]; m = len(fr); ce = feats[q][:, ce_i]; b = np.array(bge[q]); gold = np.array([f["is_gold"] for f in fr], float)
        s = own[q]; assert [f["idx"] for f in s["fragments"]] == list(range(m))
        exact = np.array(exact_cfa_score(s["full"]["answer"], [f["delete"]["answer"] for f in s["fragments"]]))
        tr4 = np.mean([labels[g][q] for g in GRID_GENS if g != READER], axis=0)
        loc2 = np.mean([labels[g][q] for g in ("ollama-gemma3-4b", "ollama-gemma3-12b")], axis=0)
        role = np.array([{"supporting": 2, "brittle": 1, "inert": 0, "harmful": -1}[f["del_role"]] for f in s["fragments"]], float)
        rankings = {
            "ce": rank(ce, ce), "dense": rank(feats[q][:, rel_i], ce), "bge": rank(b, ce),
            "cfa_transfer": rank(tr4, ce), "cfa_transfer_local2": rank(loc2, ce), "cfa_exact": rank(exact, ce),
            "cfa_transfer_bge": rank(tr4, b), "cfa_exact_bge": rank(exact, b),
            "cfa_transfer_local2_bge": rank(loc2, b), "cfa_transfer_gemma4b": rank(labels["ollama-gemma3-4b"][q], ce),
            "cfa_transfer_gemma4b_bge": rank(labels["ollama-gemma3-4b"][q], b),
            "gold_oracle": rank(gold, ce), "gold_cfa_transfer": rank(gold * 10 + tr4, ce), "gold_cfa_exact": rank(gold * 10 + exact, ce),
            "signed_oracle": rank(role, ce),
        }
        cache_tok = {}
        def ntok(idxs, fr=fr, cache_tok=cache_tok):
            k = tuple(idxs)
            if k not in cache_tok: cache_tok[k] = len(enc.encode(render(fr, list(idxs))))
            return cache_tok[k]
        sels = {"full": list(range(m))}
        for name, r in rankings.items():
            for bud in BUDGETS: sels[f"{name}_tok{bud}"] = budget_select(r, ntok, bud)
        prune = [i for i in range(m) if exact[i] > 0] or [rankings["ce"][0]]
        sels["cfa_prune"] = sorted(prune)
        for base in ("ce", "bge"):
            sels[f"{base}_match_prune"] = budget_select(rankings[base], ntok, ntok(sorted(prune)))
        plan[q] = {"rankings": rankings, "selections": sels, "tokens": {k: ntok(v) for k, v in sels.items()}}
    return pools, plan


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--estimate", action="store_true"); ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--reader", choices=list(CFG), required=True)
    a = ap.parse_args()
    MODEL, READER, stem, pin, pout = CFG[a.reader]
    CACHE = HERE / f"outputs/{stem}_cache.jsonl"; OUT = HERE / f"outputs/{stem}_answers.jsonl"
    pools, plan = build(READER)
    jobs = {}
    for q, pl in plan.items():
        for sel in pl["selections"].values(): jobs[(q, tuple(sel))] = None
    cache = {}
    if CACHE.exists():
        for r in jl(CACHE): cache[(r["q"], tuple(r["sel"]))] = r["answer"]
    todo = [k for k in jobs if k not in cache]
    intok = sum(len(enc.encode(prompt(pools[q]["question"], pools[q]["fragments"], list(sel)))) + 8 for q, sel in todo)
    cost = intok / 1e6 * pin + len(todo) * 12 / 1e6 * pout
    print(f"unique contexts {len(jobs)}, to generate {len(todo)}, input tokens {intok/1e6:.2f}M, est. cost ${cost:.2f} ({MODEL} ${pin}/${pout} per M)", flush=True)
    if a.estimate: return
    keyname = "OPENAI_API_KEY" if a.reader == "gpt" else "GOOGLE_API_KEY"
    for line in open(".env"):
        if line.startswith(keyname + "="): os.environ[keyname] = line.split("=", 1)[1].strip().strip('"').strip("'")
    if a.reader == "gpt":
        from openai import OpenAI
        client = OpenAI(timeout=90)
        def ask(text):
            r = client.chat.completions.create(model=MODEL, temperature=0, max_tokens=48, seed=0, messages=[{"role": "user", "content": text}])
            return (r.choices[0].message.content or "").strip(), r.model
    else:
        import logging
        from google import genai
        from google.genai import types as gt
        logging.getLogger("google_genai").setLevel(logging.ERROR); logging.getLogger("google_genai.types").setLevel(logging.ERROR)
        client = genai.Client(api_key=os.environ[keyname], http_options=gt.HttpOptions(timeout=90000))
        cfg = gt.GenerateContentConfig(temperature=0.0, max_output_tokens=48)
        def ask(text):
            r = client.models.generate_content(model=MODEL, contents=text, config=cfg)
            return (r.text or "").strip(), MODEL
    def call(k):
        q, sel = k
        for attempt in range(6):
            try:
                ans, model = ask(prompt(pools[q]["question"], pools[q]["fragments"], list(sel)))
                return k, ans, model
            except Exception as e:
                err = e
                if "401" in str(e) or "API key" in str(e): break
                time.sleep(2 * (attempt + 1))
        raise RuntimeError(f"failed {k[0]}: {str(err)[:120]}")
    t0 = time.time()
    with CACHE.open("a") as stream, ThreadPoolExecutor(a.workers) as ex:
        for n, (k, ans, model) in enumerate(ex.map(call, todo), 1):
            cache[k] = ans; stream.write(json.dumps({"q": k[0], "sel": list(k[1]), "answer": ans, "model": model}, ensure_ascii=False) + "\n")
            if n % 500 == 0: stream.flush(); print(n, len(todo), f"{time.time()-t0:.0f}s", flush=True)
    with OUT.open("w") as out:
        for q, pl in plan.items():
            p = pools[q]; gold = set(p["gold_idxs"])
            methods = {name: {**metrics(cache[(q, tuple(sel))], p["golds"]), "answer": cache[(q, tuple(sel))], "selected_idxs": sel,
                              "context_tokens": pl["tokens"][name], "gold_recall": len(gold & set(sel)) / max(1, len(gold))}
                       for name, sel in pl["selections"].items()}
            out.write(json.dumps({"question_group": q, "dataset": p["dataset"], "golds": p["golds"], "rankings": pl["rankings"], "methods": methods}, ensure_ascii=False) + "\n")
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
