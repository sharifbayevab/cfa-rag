"""C1 confirmatory run, Amendment 4: GPT-4.1 as the second independent confirmatory reader.

Pre-registered in review/COMMENT1_CONFIRM_PROTOCOL_RU.md, Amendment 4 (hashed before any GPT-4.1 call).
Only the contexts needed by T1/T2 are generated (full, HARM, CONT, RAND-M on questions where HARM or
CONT fires), questions in a seeded random order, spend guard at $1.30 from API usage.

  python3 repro/c1_confirm41.py --gen
  python3 repro/c1_confirm41.py --analyze      -> outputs/c1_confirm41.json
"""
import argparse, hashlib, json, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from cfa_lib import boot_indices, holm, jl, load_pools, load_signed, metrics, normalize
from e4_second_reader import prompt
from c1_confirm import B, CACHE, PROTOCOL, key_for, load_cache, pooled, rules, sel_del

R41, R4O = "openai-gpt-4.1", "openai-gpt-4o-mini"
STOP_USD = 1.30
USED = ("HARM", "CONT", "RANDM")


def view(reader, pools, qs):
    S = load_signed(reader); V = {}
    for q in qs:
        s = S[q]; assert len(s["fragments"]) == len(pools[q]["fragments"]) and s["golds"] == pools[q]["golds"]
        V[q] = {"cf": s["full"]["cover"], "dc": np.array([f["delete"]["cover"] for f in s["fragments"]]), "ans": s["full"]["answer"]}
    return V


def plan(reader, pools, qs, bge, V):
    """Per question: rules (HARM, CONT, RAND-M only) and the contexts they need."""
    out = {}
    for q in qs:
        m = len(pools[q]["fragments"]); R = rules(reader, q, V[q], pools[q], np.array(bge[q], float)); R = {k: R[k] for k in USED}
        dels = sorted({sel_del(m, i) for k in R for i in R[k]})
        out[q] = (R, ([tuple(range(m))] + dels) if dels else [])
    return out


def order41(P):
    fire = sorted(q for q, (R, sels) in P.items() if R["HARM"] or R["CONT"])
    return [fire[i] for i in np.random.default_rng(20261112).permutation(len(fire))]


def spent():
    tot = 0.0
    if CACHE.exists():
        for r in jl(CACHE):
            if r["reader"] == R41 and "usage" in r:
                u = r["usage"]; tot += ((u["prompt"] - u["cached"]) * 2.0 + u["cached"] * 0.5 + u["completion"] * 8.0) / 1e6
    return tot


def gen(pools, P, cache):
    from openai import OpenAI
    client = OpenAI(api_key=key_for("OPENAI_API_KEY"), timeout=90, max_retries=3)
    def ask(job):
        q, sel = job
        r = client.chat.completions.create(model="gpt-4.1", temperature=0, max_tokens=48, seed=0,
                                           messages=[{"role": "user", "content": prompt(pools[q]["question"], pools[q]["fragments"], list(sel))}])
        d = getattr(r.usage, "prompt_tokens_details", None)
        return job, (r.choices[0].message.content or "").strip(), r.model, {"prompt": r.usage.prompt_tokens, "cached": (getattr(d, "cached_tokens", 0) or 0) if d else 0, "completion": r.usage.completion_tokens}
    cost = spent(); done = 0
    with open(CACHE, "a") as stream, ThreadPoolExecutor(6) as ex:
        for q in order41(P):
            todo = [(q, s) for s in P[q][1] if (R41, "B", q, s) not in cache]
            if not todo: done += 1; continue
            if cost >= STOP_USD: print(f"STOP at ${cost:.3f}: questions completed {done}", flush=True); break
            for (qq, sel), ans, model, u in ex.map(ask, todo):
                cache[(R41, "B", qq, sel)] = ans; cost += ((u["prompt"] - u["cached"]) * 2.0 + u["cached"] * 0.5 + u["completion"] * 8.0) / 1e6
                stream.write(json.dumps({"reader": R41, "run": "B", "q": qq, "sel": list(sel), "answer": ans, "model": model, "usage": u}, ensure_ascii=False) + "\n")
            stream.flush(); done += 1
            if done % 20 == 0: print(done, f"${cost:.3f}", flush=True)
    print(f"done questions {done} of {len(order41(P))}, spend by usage ${cost:.3f}", flush=True)


def deltas(reader, pools, qs, P, V, cache):
    """Delta arrays (Q,) per rule; NaN where a firing question has no complete run-B context set."""
    D = {k: np.zeros(len(qs)) for k in USED}; fire = {k: np.zeros(len(qs)) for k in USED}
    cb = np.full(len(qs), np.nan); ca = np.array([V[q]["cf"] for q in qs]); same = np.full(len(qs), np.nan); nfire = ndone = 0
    for qi, q in enumerate(qs):
        R, sels = P[q]; p = pools[q]; m = len(p["fragments"])
        if not sels: continue
        nfire += 1
        if any((reader, "B", q, s) not in cache for s in sels):
            for k in USED: D[k][qi] = np.nan; fire[k][qi] = np.nan
            continue
        ndone += 1; full = cache[(reader, "B", q, tuple(range(m)))]; cb[qi] = metrics(full, p["golds"])["cover"]; same[qi] = normalize(full) == normalize(V[q]["ans"])
        for k in USED:
            if R[k]:
                D[k][qi] = np.mean([metrics(cache[(reader, "B", q, sel_del(m, i))], p["golds"])["cover"] - cb[qi] for i in R[k]]); fire[k][qi] = 1
    return D, fire, {"cover_A": ca, "cover_B": cb, "same": same}, nfire, ndone


def analyze(pools, qs, bge, cache):
    ds = np.array([pools[q]["dataset"] for q in qs]); bi = boot_indices(ds, B=B, seed=0)
    res = {}
    for r in (R41, R4O):
        V = view(r, pools, qs); P = plan(r, pools, qs, bge, V); res[r] = deltas(r, pools, qs, P, V, cache)
    D41, f41, ab41, nfire, ndone = res[R41]; D4o, f4o, ab4o, _, _ = res[R4O]
    w = nfire / ndone  # 1.0 unless the spend guard stopped the run (Amendment 4, D3)
    def wt(a):
        """Weighted variant for GPT-4.1 if truncated: completed firing questions stand for all firing questions."""
        a = np.array(a, float)
        if w == 1.0: return a
        firing = ~np.isnan(ab41["cover_B"]); a = a.copy(); a[firing] *= w; return np.where(np.isnan(a), 0.0, a)
    pp = lambda a: pooled(np.atleast_2d(a) * 100, bi)
    def fam(t1, t2):
        f = {"T1_HARM-CONT": pp(t1), "T2_HARM-RANDM": pp(t2)}; adj = holm({k: v["p"] for k, v in f.items()})
        for k in f: f[k]["p_holm"] = adj[k]
        return f
    t1_41, t2_41 = wt(D41["HARM"] - D41["CONT"]), wt(D41["HARM"] - D41["RANDM"])
    f41fam = fam(t1_41, t2_41)
    poolfam = fam(np.stack([D4o["HARM"] - D4o["CONT"], t1_41]), np.stack([D4o["HARM"] - D4o["RANDM"], t2_41]))
    ok = lambda e: e["p_holm"] < 0.05 and e["mean"] > 0
    if all(ok(v) for v in f41fam.values()) and all(ok(v) for v in poolfam.values()): verdict = "confirmed on two independent readers"
    elif ok(f41fam["T2_HARM-RANDM"]): verdict = "GPT-4.1: better than random deletion (T2), advantage over containment (T1) not confirmed"
    else: verdict = "not replicated on GPT-4.1"
    def desc(D, fire, ab):
        h = fire["HARM"] == 1; g = ~np.isnan(ab["cover_B"])
        return {"levels_pp": {k: pp(wt(D[k]) if D is D41 else D[k]) for k in USED}, "share_removed": {k: float(np.nanmean(fire[k])) for k in USED},
                "harm_questions": int(h.sum()), "HARM_fixed_n": int((D["HARM"][h] > 0).sum()), "HARM_broke_n": int((D["HARM"][h] < 0).sum()),
                "B_full_already_correct_n": int(ab["cover_B"][h].sum()), "RANDM_delta_pp_on_harm_q": float(D["RANDM"][h].mean() * 100), "HARM_delta_pp_on_harm_q": float(D["HARM"][h].mean() * 100),
                "RANDM_any_fixed_n": int((D["RANDM"][h] > 0).sum()), "RANDM_any_broke_n": int((D["RANDM"][h] < 0).sum()),
                "A_vs_B_on_generated": {"n": int(g.sum()), "cover_agree": float((ab["cover_A"][g] == ab["cover_B"][g]).mean()), "answer_same_normalized": float(ab["same"][g].mean())}}
    lv_pool = {k: pooled(np.stack([D4o[k], wt(D41[k])]) * 100, bi) for k in USED}
    usage = [r["usage"] for r in jl(CACHE) if r["reader"] == R41 and "usage" in r]
    out = {"protocol": "review/COMMENT1_CONFIRM_PROTOCOL_RU.md (Amendment 4)", "protocol_sha256": hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(), "B": B,
           "gpt41": {"family": f41fam, "firing_questions": nfire, "completed_firing_questions": ndone, "weight": w, **desc(D41, f41, ab41)},
           "gpt4omini_same_rules": {"family_descriptive": fam(D4o["HARM"] - D4o["CONT"], D4o["HARM"] - D4o["RANDM"]), **desc(D4o, f4o, ab4o)},
           "pooled_gpt4omini_gpt41": {"family": poolfam, "levels_pp": lv_pool}, "verdict": verdict,
           "by_dataset_gpt41": {d: {"T1": pooled((np.where(ds == d, 1.0, np.nan) * t1_41)[None] * 100, bi), "T2": pooled((np.where(ds == d, 1.0, np.nan) * t2_41)[None] * 100, bi)} for d in sorted(set(ds))},
           "cost_gpt41": {"calls": len(usage), "prompt_tokens": sum(u["prompt"] for u in usage), "cached_tokens": sum(u["cached"] for u in usage), "completion_tokens": sum(u["completion"] for u in usage), "usd_by_usage": spent()}}
    json.dump(out, open(HERE / "outputs/c1_confirm41.json", "w"), indent=1)
    f3 = lambda e: f"{e['mean']:+.2f} [{e['ci95'][0]:+.2f}; {e['ci95'][1]:+.2f}] p={e['p']:.4f}" + (f" pH={e['p_holm']:.4f}" if "p_holm" in e else "")
    print("protocol sha256", out["protocol_sha256"]); print("VERDICT:", verdict)
    print("GPT-4.1 family"); [print(" ", k, f3(v)) for k, v in f41fam.items()]
    print("POOLED 4o-mini + 4.1 family"); [print(" ", k, f3(v)) for k, v in poolfam.items()]
    print("pooled levels", {k: f3(v) for k, v in lv_pool.items()})
    for name in ("gpt41", "gpt4omini_same_rules"):
        e = out[name]; print(name, {k: f3(v) for k, v in e["levels_pp"].items()}); print("  ", {k: v for k, v in e.items() if k not in ("levels_pp", "family", "family_descriptive")})
    print("4o-mini check", {k: f3(v) for k, v in out["gpt4omini_same_rules"]["family_descriptive"].items()})
    print("by dataset 4.1", {d: {k: f3(v) for k, v in e.items()} for d, e in out["by_dataset_gpt41"].items()}); print("cost", out["cost_gpt41"])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--gen", action="store_true"); ap.add_argument("--analyze", action="store_true"); a = ap.parse_args()
    pools = load_pools(); qs = sorted(pools); bge = json.load(open(HERE / "outputs/e3_bge_reranker_scores.json"))["scores"]; cache = load_cache()
    if a.gen:
        P = plan(R41, pools, qs, bge, view(R41, pools, qs)); print("firing questions", len(order41(P)), "contexts", sum(len(s) for _, s in P.values()), flush=True); gen(pools, P, cache)
    if a.analyze: analyze(pools, qs, bge, cache)


if __name__ == "__main__":
    main()
