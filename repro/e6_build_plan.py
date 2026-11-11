"""E6c. Frozen ranking plan for the round-2 sweep on the fresh HF Qwen3-4B reader (Spark).

Containment uses the reader's own full-context answer saved in phase A
(inputs_e1/e1_loo.jsonl); transferred CFA is the same five-generator consensus as
round 1.  The three reference arms (bge, cfa_transfer_bge, cfa_exact_bge) are
re-generated in the same run so that no round-1 answers are mixed into contrasts.
"""
import hashlib, json, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from cfa_lib import GRID_GENS, exact_cfa_score, jl, load_features, load_pools, rank
from e6_lib import contains, contains_any, e6_rankings, load_gemma4b_answers, load_grid_answers, overlap


def main():
    pools = load_pools(); feats, labels, names = load_features(); gem = load_gemma4b_answers()
    bge = json.load(open(HERE / "outputs/e3_bge_reranker_scores.json"))["scores"]
    loo = {r["question_group"]: r for r in jl(HERE / "inputs_e1/e1_loo.jsonl")}
    assert set(loo) == set(pools) == set(gem)
    rows = []; rows2 = []; GA = load_grid_answers()
    for q in sorted(pools):
        p = pools[q]; fr = p["fragments"]; m = len(fr); L = loo[q]
        assert L["question"] == p["question"] and len(L["delete_answers"]) == m
        ce = feats[q][:, names.index("ce")]; b = np.array(bge[q])
        exact = np.array(exact_cfa_score(L["full_answer"], L["delete_answers"])); tr = np.mean([labels[g][q] for g in GRID_GENS], axis=0)
        ac = contains(L["full_answer"], fr); acg = contains(gem[q], fr); gold = contains_any(p["golds"], fr); ov = overlap(L["full_answer"], fr)
        rk = e6_rankings(ac, acg, gold, ov, tr, exact, b, ce); rank_ce = rank(ce, ce)
        prune = sorted(i for i in range(m) if exact[i] > 0) or [int(rank_ce[0])]  # round-1 fallback: top-1 CE
        acp = sorted(i for i in range(m) if ac[i] > 0) or [rk["bge"][0]]
        rows.append({"question_group": q, "dataset": p["dataset"], "qid": p["qid"], "rankings": {k: [int(x) for x in v] for k, v in rk.items()},
                     "fixed": {"cfa_prune": [int(i) for i in prune], "anscont_prune": [int(i) for i in acp]},
                     "matched": {"bge_match_prune": ["bge", "cfa_prune"], "anscont_bge_match_prune": ["anscont_bge", "cfa_prune"],
                                 "cfa_transfer_bge_match_prune": ["cfa_transfer_bge", "cfa_prune"], "bge_match_acprune": ["bge", "anscont_prune"]},
                     "scores": {"anscont": ac.tolist(), "anscont_gemma4b": acg.tolist(), "goldans": gold.tolist(), "overlap": ov.tolist(),
                                "exact": exact.tolist(), "transfer5": tr.tolist()}})
        # supplementary plan (second Spark run): containment of the five donors' own full-context answers, zero target-reader calls
        acd = np.mean([contains(GA[g][q], fr) for g in GRID_GENS], axis=0)
        rows2.append({"question_group": q, "dataset": p["dataset"], "qid": p["qid"],
                      "rankings": {"bge": rk["bge"], "cfa_transfer_bge": rk["cfa_transfer_bge"], "anscont_bge": rk["anscont_bge"],
                                   "anscont_donors_bge": rank(acd, b), "sum_donorcont_transfer_bge": rank(acd + tr, b)},
                      "scores": {"anscont_donors": acd.tolist()}})
        rows2[-1]["rankings"] = {k: [int(x) for x in v] for k, v in rows2[-1]["rankings"].items()}
    out2 = HERE / "inputs_e6/e6_plan_donors.jsonl"
    out2.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows2))
    print("donor plan sha256", hashlib.sha256(out2.read_bytes()).hexdigest())
    out = HERE / "inputs_e6/e6_plan.jsonl"
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    print("rows", len(rows), "sha256", hashlib.sha256(out.read_bytes()).hexdigest())
    old = {r["question_group"]: r for r in jl(HERE / "inputs_e1/e1_plan.jsonl")}
    for k in ("bge", "cfa_transfer_bge", "cfa_exact_bge"): assert all(r["rankings"][k] == old[r["question_group"]]["rankings"][k] for r in rows), k
    assert all(r["fixed"]["cfa_prune"] == old[r["question_group"]]["fixed"]["cfa_prune"] for r in rows)
    print("reference rankings and cfa_prune identical to round-1 plan")
    print("questions with no passage containing own answer: %d; mean anscont_prune size %.2f; mean cfa_prune size %.2f" % (
        sum(not any(r["scores"]["anscont"]) for r in rows), np.mean([len(r["fixed"]["anscont_prune"]) for r in rows]), np.mean([len(r["fixed"]["cfa_prune"]) for r in rows])))


if __name__ == "__main__":
    main()
