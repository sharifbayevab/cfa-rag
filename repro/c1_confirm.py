"""C1 confirmatory run (reviewer Comment 1): own harm labels from run A, outcome on a fresh run B.

Pre-registered in review/COMMENT1_CONFIRM_PROTOCOL_RU.md (incl. Amendment 1), written and hashed
before any generation.  Two tests (Holm): T1 HARM - CONT, T2 HARM - RAND-M.
Readers (after Amendments 1-3): GPT-4o-mini (run A = signed grid, 450 q), Gemini-3.1-flash-lite (run A generated here, 300 q).
Prompt/decoding: e4_second_reader.prompt, temperature 0, 48 tokens (e7_reprobe conventions).

  python3 repro/c1_confirm.py --estimate
  python3 repro/c1_confirm.py --gen A        # Gemini-3.1-flash-lite full LOO
  python3 repro/c1_confirm.py --gen B        # all three readers
  python3 repro/c1_confirm.py --analyze
Cache: outputs/c1_confirm_answers.jsonl ; result: outputs/c1_confirm.json
"""
import argparse, hashlib, json, os, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from cfa_lib import boot_indices, holm, jl, load_pools, load_signed, metrics, normalize
from e6_lib import contains
from e4_second_reader import enc, prompt

PROTOCOL = HERE.parent / "review/COMMENT1_CONFIRM_PROTOCOL_RU.md"
CACHE = HERE / "outputs/c1_confirm_answers.jsonl"
READERS = ["openai-gpt-4o-mini", "gemini-gemini-3.1-flash-lite"]  # Amendment 3: gpt-4.1-nano deprecated (404), dropped before any run-B answer
NICE = {"openai-gpt-4.1-nano": "GPT-4.1-nano", "openai-gpt-4o-mini": "GPT-4o-mini", "gemini-gemini-3.1-flash-lite": "Gemini-3.1-flash-lite"}
MODEL = {"openai-gpt-4.1-nano": ("openai", "gpt-4.1-nano", 0.10, 0.40), "openai-gpt-4o-mini": ("openai", "gpt-4o-mini", 0.15, 0.60),
         "gemini-gemini-3.1-flash-lite": ("gemini", "gemini-3.1-flash-lite", 0.25, 1.50)}
NEW = "gemini-gemini-3.1-flash-lite"
K = 3; B = 2000


def new_subset(pools):
    """Amendment 2 (R2): 100 questions per dataset for the new reader, one generator, datasets in name order."""
    rng = np.random.default_rng(20261111); out = set()
    for d in sorted({p["dataset"] for p in pools.values()}):
        out |= set(rng.choice(sorted(q for q, p in pools.items() if p["dataset"] == d), 100, replace=False).tolist())
    return out


def load_cache():
    c = {}
    if CACHE.exists():
        for r in jl(CACHE): c[(r["reader"], r["run"], r["q"], tuple(r["sel"]))] = r["answer"]
    return c


def key_for(name):
    for line in open(".env"):
        if line.startswith(name + "="): return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise RuntimeError("missing key " + name)


def make_ask(reader):
    prov, model, _, _ = MODEL[reader]
    if prov == "openai":
        from openai import OpenAI
        client = OpenAI(api_key=key_for("OPENAI_API_KEY"), timeout=90)
        def ask(text):
            r = client.chat.completions.create(model=model, temperature=0, max_tokens=48, seed=0, messages=[{"role": "user", "content": text}])
            return (r.choices[0].message.content or "").strip(), r.model
    else:
        import logging
        from google import genai
        from google.genai import types as gt
        logging.getLogger("google_genai").setLevel(logging.ERROR); logging.getLogger("google_genai.types").setLevel(logging.ERROR)
        client = genai.Client(api_key=key_for("GOOGLE_API_KEY"), http_options=gt.HttpOptions(timeout=90000))
        cfg = gt.GenerateContentConfig(temperature=0.0, max_output_tokens=48)
        def ask(text):
            r = client.models.generate_content(model=model, contents=text, config=cfg)
            return (r.text or "").strip(), model
    return ask


def a_views(pools, qs, cache, need_new=True):
    """Run-A per-passage arrays (cf, dc, answer) for each reader."""
    V = {}
    for r in READERS:
        V[r] = {}
        if r != NEW:
            S = load_signed(r)
            for q in qs:
                s = S[q]; sf = s["fragments"]; assert len(sf) == len(pools[q]["fragments"]) and s["golds"] == pools[q]["golds"]
                V[r][q] = {"cf": s["full"]["cover"], "dc": np.array([f["delete"]["cover"] for f in sf]), "ans": s["full"]["answer"]}
        elif need_new:
            for q in sorted(new_subset(pools)):
                p = pools[q]; m = len(p["fragments"]); full = cache[(r, "A", q, tuple(range(m)))]
                dels = [cache[(r, "A", q, tuple(j for j in range(m) if j != i))] for i in range(m)]
                V[r][q] = {"cf": metrics(full, p["golds"])["cover"], "dc": np.array([metrics(d, p["golds"])["cover"] for d in dels]), "ans": full}
    return V


def rules(reader, q, v, p, b):
    """Passages removed by each rule (protocol section 4); everything from run A only."""
    fr = p["fragments"]; m = len(fr); harm = v["dc"] > v["cf"]; wrong = v["cf"] == 0
    gold = np.array([f["is_gold"] for f in fr], bool)
    out = {"HARM": [], "CONT": [], "RANDM": [], "RANDA": []}
    rng = np.random.default_rng(int(hashlib.sha256(f"c1confirm|{reader}|{q}".encode()).hexdigest()[:16], 16))
    if harm.any():
        c = np.where(harm)[0]; out["HARM"] = [int(c[np.argmin(b[c])])]
        cand = sorted(int(i) for i in range(m) if not gold[i] and not harm[i]) or sorted(int(i) for i in range(m) if not harm[i])
        out["RANDM"] = [int(x) for x in rng.permutation(cand)[:K]]
    cont = contains(v["ans"], fr) * wrong
    if cont.max() > 0.5:
        c = np.where(cont > 0.5)[0]; out["CONT"] = [int(c[np.argmin(b[c])])]
    if wrong: out["RANDA"] = [int(rng.integers(m))]
    return out


def sel_del(m, i): return tuple(j for j in range(m) if j != i)


def jobs_B(pools, qs, V, bge, readers):
    jobs = []
    for r in readers:
        for q in qs:
            if q not in V[r]: continue
            p = pools[q]; m = len(p["fragments"]); R = rules(r, q, V[r][q], p, np.array(bge[q], float))
            sels = {tuple(range(m))} | {sel_del(m, i) for k in R for i in R[k]}
            jobs += [(r, "B", q, s) for s in sorted(sels)]
    return jobs


def estimate(pools, jobs, margin=1.0):
    tot = {}
    for r, run, q, sel in jobs:
        _, _, pin, pout = MODEL[r]; n = len(enc.encode(prompt(pools[q]["question"], pools[q]["fragments"], list(sel)))) + 8
        t = tot.setdefault((r, run), [0, 0, 0.0]); t[0] += 1; t[1] += n; t[2] += (n * margin * pin + 12 * pout) / 1e6
    for k, t in sorted(tot.items()): print(f"  {k[0]} run {k[1]}: contexts {t[0]}, input {t[1]/1e6:.2f}M tok, est ${t[2]:.2f}", flush=True)
    return {f"{k[0]}|{k[1]}": {"contexts": t[0], "input_tokens_o200k": t[1], "est_usd": t[2]} for k, t in tot.items()}


def run_jobs(pools, todo, cache, workers=12):
    asks = {r: make_ask(r) for r in sorted({j[0] for j in todo})}
    def call(k):
        r, run, q, sel = k
        for attempt in range(8):
            try:
                ans, model = asks[r](prompt(pools[q]["question"], pools[q]["fragments"], list(sel))); return k, ans, model
            except Exception as e:
                err = e
                if "401" in str(e) or "API key" in str(e) or "insufficient_quota" in str(e): break
                time.sleep(3 * (attempt + 1))
        raise RuntimeError(f"failed {r} {q}: {type(err).__name__} {str(err)[:200]}")
    t0 = time.time()
    with open(CACHE, "a") as stream, ThreadPoolExecutor(workers) as ex:
        for n, (k, ans, model) in enumerate(ex.map(call, todo), 1):
            cache[k] = ans; stream.write(json.dumps({"reader": k[0], "run": k[1], "q": k[2], "sel": list(k[3]), "answer": ans, "model": model}, ensure_ascii=False) + "\n")
            if n % 200 == 0: stream.flush(); print(n, len(todo), f"{time.time()-t0:.0f}s", flush=True)


def pooled(A, bi):
    A = np.asarray(A, float); ok = ~np.isnan(A); S = np.where(ok, A, 0).sum(0); N = ok.sum(0)
    boots = S[bi].sum(1) / np.maximum(N[bi].sum(1), 1)
    p = 2 * min(np.mean(boots <= 0), np.mean(boots >= 0))
    return {"mean": float(S.sum() / max(N.sum(), 1)), "ci95": [float(x) for x in np.percentile(boots, [2.5, 97.5])],
            "p": float(min(1.0, max(p, 1 / len(bi)))), "n_pairs": int(N.sum()), "n_questions": int((N > 0).sum())}


def analyze(pools, qs, bge, cache):
    V = a_views(pools, qs, cache); ds = np.array([pools[q]["dataset"] for q in qs]); bi = boot_indices(ds, B=B, seed=0)
    R, Q = len(READERS), len(qs)
    D = {k: np.zeros((R, Q)) for k in ("HARM", "CONT", "RANDM", "RANDA")}; fire = {k: np.zeros((R, Q)) for k in D}
    cfA = np.zeros((R, Q)); cfB = np.zeros((R, Q)); same = np.zeros((R, Q)); empty = {r: 0 for r in READERS}
    for ri, r in enumerate(READERS):
        for qi, q in enumerate(qs):
            if q not in V[r]:
                for arr in (*D.values(), *fire.values(), cfA, cfB, same): arr[ri, qi] = np.nan
                continue
            p = pools[q]; m = len(p["fragments"]); full = cache[(r, "B", q, tuple(range(m)))]
            cb = metrics(full, p["golds"])["cover"]; cfB[ri, qi] = cb; cfA[ri, qi] = V[r][q]["cf"]; same[ri, qi] = normalize(full) == normalize(V[r][q]["ans"])
            empty[r] += full == ""
            rl = rules(r, q, V[r][q], p, np.array(bge[q], float))
            for k, idx in rl.items():
                if idx:
                    answers = [cache[(r, "B", q, sel_del(m, i))] for i in idx]; empty[r] += sum(a == "" for a in answers)
                    D[k][ri, qi] = np.mean([metrics(a, p["golds"])["cover"] - cb for a in answers]); fire[k][ri, qi] = 1
    pp = lambda a, rows=slice(None): pooled(np.asarray(a)[rows] * 100, bi)
    fam = {"T1_HARM-CONT": pp(D["HARM"] - D["CONT"]), "T2_HARM-RANDM": pp(D["HARM"] - D["RANDM"])}
    adj = holm({k: v["p"] for k, v in fam.items()})
    for k in fam: fam[k]["p_holm"] = adj[k]
    sens = {"T1_HARM-CONT": pp(D["HARM"] - D["CONT"], slice(0, 1)), "T2_HARM-RANDM": pp(D["HARM"] - D["RANDM"], slice(0, 1))}
    ok = all(v["p_holm"] < 0.05 and v["mean"] > 0 for v in fam.values())
    t2 = fam["T2_HARM-RANDM"]; t2ok = t2["p_holm"] < 0.05 and t2["mean"] > 0
    if ok and all(v["mean"] > 0 for v in sens.values()): verdict = "confirmed"
    elif ok: verdict = "partially confirmed (GPT-4o-mini-only sensitivity estimate <= 0)"
    elif t2ok: verdict = "partially confirmed (T2 only)"
    else: verdict = "not confirmed"

    def cond(rows):
        """Descriptive, on pairs where HARM fires."""
        f = fire["HARM"][rows] > 0; n = f.sum()
        return {"n_pairs": int(n), "HARM_fixed": float((D["HARM"][rows][f] > 0).mean()), "HARM_broke": float((D["HARM"][rows][f] < 0).mean()),
                "B_full_already_correct": float(cfB[rows][f].mean()), "HARM_delta_pp": float(D["HARM"][rows][f].mean() * 100),
                "RANDM_delta_pp": float(D["RANDM"][rows][f].mean() * 100), "RANDM_any_fixed": float((D["RANDM"][rows][f] > 0).mean()), "RANDM_any_broke": float((D["RANDM"][rows][f] < 0).mean())}
    def block(rows):
        return {"levels_pp": {k: pp(D[k], rows) for k in D}, "T1_HARM-CONT": pp(D["HARM"] - D["CONT"], rows), "T2_HARM-RANDM": pp(D["HARM"] - D["RANDM"], rows),
                "HARM-RANDA": pp(D["HARM"] - D["RANDA"], rows), "CONT-RANDA": pp(D["CONT"] - D["RANDA"], rows),
                "share_removed": {k: float(np.nanmean(fire[k][rows])) for k in D}, "on_harm_pairs": cond(rows),
                "A_vs_B_full": {"cover_A": float(np.nanmean(cfA[rows])), "cover_B": float(np.nanmean(cfB[rows])), "cover_agree": float((cfA[rows] == cfB[rows])[~np.isnan(cfA[rows])].mean()), "answer_same_normalized": float(np.nanmean(same[rows]))}}
    out = {"protocol": "review/COMMENT1_CONFIRM_PROTOCOL_RU.md", "protocol_sha256": hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),
           "readers": READERS, "B": B, "family": fam, "verdict": verdict, "pooled": block(slice(None)),
           "sensitivity_gpt4omini_only": {**block(slice(0, 1)), "note": "pre-declared sensitivity (Amendment 1, P3), no Holm"},
           "by_reader": {r: block(slice(ri, ri + 1)) for ri, r in enumerate(READERS)}, "by_dataset": {}, "empty_answers_B": empty}
    for dname in sorted(set(ds)):
        mk = np.where(ds == dname, 1.0, np.nan)[None, :]
        out["by_dataset"][dname] = {"T1_HARM-CONT": pooled((D["HARM"] - D["CONT"]) * 100 * mk, bi), "T2_HARM-RANDM": pooled((D["HARM"] - D["RANDM"]) * 100 * mk, bi)}
    json.dump(out, open(HERE / "outputs/c1_confirm.json", "w"), indent=1)
    f3 = lambda e: f"{e['mean']:+.2f} [{e['ci95'][0]:+.2f}; {e['ci95'][1]:+.2f}] p={e['p']:.4f}" + (f" pH={e['p_holm']:.4f}" if "p_holm" in e else "")
    print("protocol sha256", out["protocol_sha256"]); print("VERDICT:", verdict)
    print("FAMILY"); [print(" ", k, f3(v)) for k, v in fam.items()]
    def show(name, b):
        print(name); [print("   level", k, f3(v), "removed", round(b["share_removed"][k], 3)) for k, v in b["levels_pp"].items()]
        [print("  ", k, f3(b[k])) for k in ("T1_HARM-CONT", "T2_HARM-RANDM", "HARM-RANDA", "CONT-RANDA")]
        print("   on harm pairs", {k: round(v, 3) for k, v in b["on_harm_pairs"].items()}); print("   A vs B", {k: round(v, 3) for k, v in b["A_vs_B_full"].items()})
    show("POOLED", out["pooled"]); show("SENS GPT-4o-mini only", out["sensitivity_gpt4omini_only"])
    for r in READERS: show(NICE[r], out["by_reader"][r])
    print("BY DATASET"); [print(" ", d, {k: f3(v) for k, v in e.items()}) for d, e in out["by_dataset"].items()]
    print("empty", empty)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--estimate", action="store_true"); ap.add_argument("--gen", choices=["A", "B"]); ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--workers", type=int, default=12); a = ap.parse_args()
    pools = load_pools(); qs = sorted(pools); bge = json.load(open(HERE / "outputs/e3_bge_reranker_scores.json"))["scores"]; cache = load_cache()
    sub = new_subset(pools)
    jobsA = [(NEW, "A", q, s) for q in sorted(sub) for s in [tuple(range(len(pools[q]["fragments"])))] + [sel_del(len(pools[q]["fragments"]), i) for i in range(len(pools[q]["fragments"]))]]
    haveA = all(j in cache for j in jobsA)
    if a.estimate:
        print("run A (new reader), o200k proxy, margin 1.25:"); estimate(pools, [j for j in jobsA if j not in cache], 1.25)
        V = a_views(pools, qs, cache, need_new=haveA); rd = READERS if haveA else READERS[:1]
        print("run B:"); estimate(pools, [j for j in jobs_B(pools, qs, V, bge, rd) if j not in cache], 1.25 if haveA else 1.0)
        if not haveA: print("  (run B of the new reader depends on its run A; full contexts alone: 0.71M tok)")
    if a.gen == "A":
        todo = [j for j in jobsA if j not in cache]; print("todo A", len(todo), flush=True); run_jobs(pools, todo, cache, a.workers)
    if a.gen == "B":
        assert haveA, "run A of the new reader must be complete first"
        V = a_views(pools, qs, cache); todo = [j for j in jobs_B(pools, qs, V, bge, READERS) if j not in cache]
        print("todo B", len(todo), flush=True); estimate(pools, todo); run_jobs(pools, todo, cache, a.workers)
    if a.analyze: analyze(pools, qs, bge, cache)


if __name__ == "__main__":
    main()
