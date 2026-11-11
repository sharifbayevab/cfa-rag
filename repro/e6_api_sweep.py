"""E6c + E7 selection arms on the API readers (GPT-4.1-mini, Gemini-3.5-flash-lite).

Adds to the round-1 E4 arms (answers reused from the E4 cache of the same reader,
same prompt and decoding):
  * answer-containment arms (own full-context answer, Gemma-3-4B answer, donors'
    answers, gold-answer oracle) and their combinations with transferred/exact CFA;
  * exact-CFA arms recomputed from the fresh leave-one-out probes of E7
    (cfa_exact, cfa_exact_bge, gold_cfa_exact, cfa_prune and matched baselines);
    the round-1 versions built on the months-old probes are kept as *_stale.
Only contexts that are not in a cache are generated.

  python3 repro/e6_api_sweep.py --reader gpt --estimate
  python3 repro/e6_api_sweep.py --reader gemini
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from cfa_lib import BUDGETS, GRID_GENS, budget_select, exact_cfa_score, jl, load_features, load_pools, load_signed, metrics, normalize, rank
from e4_second_reader import CFG, enc, render
from e6_lib import contains, contains_any, e6_rankings, load_gemma4b_answers, load_grid_answers, overlap
from e7_reprobe import estimate, load_cache, run_jobs


def build(reader, cache):
    READER = CFG[reader][1]
    pools = load_pools(); feats, labels, names = load_features(); gem = load_gemma4b_answers(); GA = load_grid_answers(); stale = load_signed(READER)
    bge = json.load(open(HERE / "outputs/e3_bge_reranker_scores.json"))["scores"]
    loo = {r["question_group"]: r for r in jl(HERE / f"inputs_e6/e7_loo_{reader}.jsonl")}
    donors = [g for g in GRID_GENS if g != READER]; plan = {}
    for q, p in sorted(pools.items()):
        fr = p["fragments"]; m = len(fr); ce = feats[q][:, names.index("ce")]; b = np.array(bge[q]); gold = np.array([f["is_gold"] for f in fr], float)
        full = cache[(q, tuple(range(m)))]; assert loo[q]["full_answer"] == full
        exact = np.array(exact_cfa_score(full, loo[q]["delete_answers"]))
        s = stale[q]; exact_old = np.array(exact_cfa_score(s["full"]["answer"], [f["delete"]["answer"] for f in s["fragments"]]))
        tr = np.mean([labels[g][q] for g in donors], axis=0); acd = np.mean([contains(GA[g][q], fr) for g in donors], axis=0)
        ac = contains(full, fr); rk = e6_rankings(ac, contains(gem[q], fr), contains_any(p["golds"], fr), overlap(full, fr), tr, exact, b, ce)
        rk.update({"ce": rank(ce, ce), "cfa_transfer": rank(tr, ce), "cfa_exact": rank(exact, ce), "gold_oracle": rank(gold, ce),
                   "gold_cfa_transfer": rank(gold * 10 + tr, ce), "gold_cfa_exact": rank(gold * 10 + exact, ce),
                   "cfa_exact_stale": rank(exact_old, ce), "cfa_exact_stale_bge": rank(exact_old, b),
                   "anscont_donors_bge": rank(acd, b), "sum_donorcont_transfer_bge": rank(acd + tr, b)})
        tokc = {}
        def ntok(idxs, fr=fr, tokc=tokc):
            k = tuple(idxs)
            if k not in tokc: tokc[k] = len(enc.encode(render(fr, list(idxs))))
            return tokc[k]
        sels = {"full": list(range(m))}
        for name, r in rk.items():
            for bud in BUDGETS: sels[f"{name}_tok{bud}"] = budget_select(r, ntok, bud)
        prune = sorted(i for i in range(m) if exact[i] > 0) or [rk["ce"][0]]
        prune_old = sorted(i for i in range(m) if exact_old[i] > 0) or [rk["ce"][0]]
        acp = sorted(i for i in range(m) if ac[i] > 0) or [rk["bge"][0]]
        sels.update({"cfa_prune": prune, "cfa_prune_stale": prune_old, "anscont_prune": acp})
        for base in ("ce", "bge", "anscont_bge", "cfa_transfer_bge"): sels[f"{base}_match_prune"] = budget_select(rk[base], ntok, ntok(prune))
        sels["bge_match_acprune"] = budget_select(rk["bge"], ntok, ntok(acp))
        plan[q] = {"rankings": {k: [int(x) for x in v] for k, v in rk.items()}, "selections": sels, "tokens": {k: ntok(v) for k, v in sels.items()},
                   "probe": {"n_flip": int((exact > 0).sum()), "n_flip_stale": int((exact_old > 0).sum()),
                             "label_agreement_with_stale": float(np.mean((exact > 0) == (exact_old > 0))), "full_same_as_stale": normalize(full) == normalize(s["full"]["answer"])}}
    return pools, plan


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--reader", choices=list(CFG), required=True)
    ap.add_argument("--estimate", action="store_true"); ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args(); stem = CFG[a.reader][2]
    paths = [HERE / f"outputs/{stem}_cache.jsonl", HERE / f"outputs/e7_{stem}_cache.jsonl", HERE / f"outputs/e6_{stem}_cache.jsonl"]
    cache = load_cache(*paths); pools, plan = build(a.reader, cache)
    todo = sorted({(q, tuple(sel)) for q, pl in plan.items() for sel in pl["selections"].values()} - set(cache))
    estimate(a.reader, pools, todo)
    if a.estimate: return
    run_jobs(a.reader, pools, todo, cache, paths[2], a.workers)
    with (HERE / f"outputs/e6_{stem}_answers.jsonl").open("w") as out:
        for q, pl in plan.items():
            p = pools[q]; gold = set(p["gold_idxs"])
            methods = {name: {**metrics(cache[(q, tuple(sel))], p["golds"]), "answer": cache[(q, tuple(sel))], "selected_idxs": sel,
                              "context_tokens": pl["tokens"][name], "gold_recall": len(gold & set(sel)) / max(1, len(gold))} for name, sel in pl["selections"].items()}
            out.write(json.dumps({"question_group": q, "dataset": p["dataset"], "golds": p["golds"], "pool_size": len(p["fragments"]), "probe": pl["probe"],
                                  "rankings": pl["rankings"], "methods": methods}, ensure_ascii=False) + "\n")
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
