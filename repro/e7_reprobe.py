"""E7. Fresh leave-one-out probes of the API readers (GPT-4.1-mini, Gemini-3.5-flash-lite).

Same prompt, decoding and model ids as e4_second_reader.py, so the deletion
answers are exact for the reader that produces the final answers.  The
full-context answer is the one already cached by E4 (same run conditions);
this script only adds the |P| single-deletion contexts per question.

  python3 repro/e7_reprobe.py --reader gpt --estimate
  python3 repro/e7_reprobe.py --reader gemini
Cache (resumable): outputs/e7_<stem>_cache.jsonl; result: inputs_e6/e7_loo_<reader>.jsonl
"""
import argparse, json, os, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from cfa_lib import jl, load_pools, metrics, normalize
from e4_second_reader import CFG, enc, prompt


def load_cache(*paths):
    cache = {}
    for p in paths:
        if Path(p).exists():
            for r in jl(p): cache[(r["q"], tuple(r["sel"]))] = r["answer"]
    return cache


def make_ask(reader):
    MODEL = CFG[reader][0]; keyname = "OPENAI_API_KEY" if reader == "gpt" else "GOOGLE_API_KEY"
    for line in open(".env"):
        if line.startswith(keyname + "="): os.environ[keyname] = line.split("=", 1)[1].strip().strip('"').strip("'")
    if reader == "gpt":
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
    return ask


def estimate(reader, pools, todo):
    _, _, _, pin, pout = CFG[reader]
    intok = sum(len(enc.encode(prompt(pools[q]["question"], pools[q]["fragments"], list(sel)))) + 8 for q, sel in todo)
    cost = intok / 1e6 * pin + len(todo) * 12 / 1e6 * pout
    print(f"[{reader}] contexts to generate {len(todo)}, input tokens {intok/1e6:.2f}M, est. cost ${cost:.2f}", flush=True)
    return cost


def run_jobs(reader, pools, todo, cache, cache_path, workers=12):
    ask = make_ask(reader)
    def call(k):
        q, sel = k
        for attempt in range(6):
            try:
                ans, model = ask(prompt(pools[q]["question"], pools[q]["fragments"], list(sel))); return k, ans, model
            except Exception as e:
                err = e
                if "401" in str(e) or "API key" in str(e): break
                time.sleep(2 * (attempt + 1))
        raise RuntimeError(f"failed {k[0]}: {type(err).__name__}")
    t0 = time.time()
    with open(cache_path, "a") as stream, ThreadPoolExecutor(workers) as ex:
        for n, (k, ans, model) in enumerate(ex.map(call, todo), 1):
            cache[k] = ans; stream.write(json.dumps({"q": k[0], "sel": list(k[1]), "answer": ans, "model": model}, ensure_ascii=False) + "\n")
            if n % 500 == 0: stream.flush(); print(n, len(todo), f"{time.time()-t0:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--reader", choices=list(CFG), required=True)
    ap.add_argument("--estimate", action="store_true"); ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--repeat", action="store_true", help="noise floor: ask the full-context prompt once more and compare with the cached answer")
    a = ap.parse_args(); stem = CFG[a.reader][2]
    pools = load_pools()
    if a.repeat:
        ref = load_cache(HERE / f"outputs/{stem}_cache.jsonl"); rp = HERE / f"outputs/e7_{stem}_repeat.jsonl"; rep = load_cache(rp)
        todo = [(q, tuple(range(len(p["fragments"])))) for q, p in sorted(pools.items()) if (q, tuple(range(len(p["fragments"])))) not in rep]
        estimate(a.reader, pools, todo)
        if a.estimate: return
        run_jobs(a.reader, pools, todo, rep, rp, a.workers)
        same = [normalize(rep[k]) == normalize(ref[k]) for k in rep]; cov = [metrics(rep[(q, tuple(range(len(p["fragments"]))))], p["golds"])["cover"] for q, p in pools.items()]
        out = {"reader": CFG[a.reader][0], "n": len(same), "repeat_full_answer_same_normalized": sum(same) / len(same), "repeat_full_cover": sum(cov) / len(cov)}
        (HERE / f"outputs/e7_{stem}_repeat.json").write_text(json.dumps(out, indent=1) + "\n"); print(out); return
    old = HERE / f"outputs/{stem}_cache.jsonl"; new = HERE / f"outputs/e7_{stem}_cache.jsonl"
    cache = load_cache(old, new)
    jobs = [(q, tuple(j for j in range(len(p["fragments"])) if j != i)) for q, p in sorted(pools.items()) for i in reversed(range(len(p["fragments"])))]  # late deletions first: longer shared prefix
    todo = [k for k in jobs if k not in cache]
    estimate(a.reader, pools, todo)
    if a.estimate: return
    run_jobs(a.reader, pools, todo, cache, new, a.workers)
    out = HERE / f"inputs_e6/e7_loo_{a.reader}.jsonl"
    with out.open("w") as f:
        for q, p in sorted(pools.items()):
            m = len(p["fragments"]); full = cache[(q, tuple(range(m)))]
            dels = [cache[(q, tuple(j for j in range(m) if j != i))] for i in range(m)]
            f.write(json.dumps({"question_group": q, "dataset": p["dataset"], "golds": p["golds"], "gold_idxs": p["gold_idxs"], "full_answer": full,
                                "full_metrics": metrics(full, p["golds"]), "delete_answers": dels, "delete_metrics": [metrics(d, p["golds"]) for d in dels],
                                "delete_changes": [normalize(d) != normalize(full) for d in dels]}, ensure_ascii=False) + "\n")
    print("DONE", out.name, flush=True)


if __name__ == "__main__":
    main()
