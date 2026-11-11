"""Build the frozen ranking plan for the E1/E2/E3 sweep on the fresh Qwen3-4B reader.

Needs repro/inputs_e1/e1_loo.jsonl (exact LOO answers of the same reader, phase A)
and repro/outputs/e3_bge_reranker_scores.json.  No generator calls.

Label-free arms never read gold answers or gold passages.  Arms that do are
named *_oracle / gold_* and are diagnostics, not deployable selectors.
"""
import hashlib, json, sys
from pathlib import Path
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from cfa_lib import (GRID_GENS, SIGNED_API_TEACHERS, exact_cfa_score, jl, load_features, load_pools, load_signed, normalize, rank)


def main(cc_from_loo=False):
    pools = load_pools(); feats, labels, names = load_features()
    bge = json.load(open(HERE / "outputs/e3_bge_reranker_scores.json"))["scores"]
    loo = {r["question_group"]: r for r in jl(HERE / "inputs_e1/e1_loo.jsonl")}
    eff = {r["question_group"]: r for r in jl(HERE / "inputs/efficiency_plan.jsonl")}
    ssp = {r["question_group"]: r for r in jl(HERE.parent / "experiments/spark/signed_support_plan.jsonl")}
    teachers = {t: load_signed(t) for t in SIGNED_API_TEACHERS}
    assert set(loo) == set(pools) == set(feats), (len(loo), len(pools))
    ce_i, rel_i = names.index("ce"), names.index("rel")
    qs = sorted(pools)
    # ---- E2: same nine features, same folds, different training targets (question-held-out GBM regressors)
    Q, PI, XA, T = [], [], [], {"gold": [], "cfa": [], "signed": [], "ce": []}
    for q in qs:
        m = len(pools[q]["fragments"])
        for i in range(m):
            Q.append(q); PI.append(i); XA.append(feats[q][i])
            T["gold"].append(float(i in pools[q]["gold_idxs"]))
            T["cfa"].append(float(np.mean([labels[g][q][i] for g in GRID_GENS])))
            T["signed"].append(float(np.mean([teachers[t][q]["fragments"][i]["del_role"] == "supporting" and
                                              teachers[t][q]["fragments"][i]["rep_role"] == "supporting" for t in teachers])))
            T["ce"].append(float(feats[q][i][ce_i]))
    Q = np.array(Q); XA = np.stack(XA); fold = np.array([pools[q]["fold"] for q in Q])
    tgt = {}
    for name, y in T.items():
        y = np.array(y); p = np.zeros(len(y))
        for f in range(5):
            tr, te = fold != f, fold == f
            assert not set(Q[tr]) & set(Q[te])
            p[te] = GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=.05, random_state=0).fit(XA[tr], y[tr]).predict(XA[te])
        tgt[name] = {q: p[Q == q] for q in qs}
    rows = []
    for q in qs:
        p = pools[q]; fr = p["fragments"]; m = len(fr); L = loo[q]
        assert L["question"] == p["question"] and L["gold_idxs"] == p["gold_idxs"] and len(L["delete_answers"]) == m
        ce = feats[q][:, ce_i]; b = np.array(bge[q]); gold = np.array([f["is_gold"] for f in fr], float)
        exact = np.array(exact_cfa_score(L["full_answer"], L["delete_answers"]))
        tr5 = np.mean([labels[g][q] for g in GRID_GENS], axis=0)
        loc2 = np.mean([labels[g][q] for g in ("ollama-gemma3-4b", "ollama-gemma3-12b")], axis=0)
        cf = L["full_metrics"]["cover"]
        role = np.array([(2 if cf > dm["cover"] else -1 if cf < dm["cover"] else 1) if ch else 0
                         for dm, ch in zip(L["delete_metrics"], L["delete_changes"])], float)
        rk = {
            "ce": rank(ce, ce), "dense": rank(feats[q][:, rel_i], ce), "bge": rank(b, ce), "oof": eff[q]["rankings"]["oof"],
            "contextcite": eff[q]["rankings"]["contextcite"],
            "cfa_transfer": rank(tr5, ce), "cfa_transfer_local2": rank(loc2, ce), "cfa_transfer_bge": rank(tr5, b),
            "cfa_transfer_local2_bge": rank(loc2, b), "cfa_transfer_gemma4b": rank(labels["ollama-gemma3-4b"][q], ce),
            "cfa_transfer_gemma4b_bge": rank(labels["ollama-gemma3-4b"][q], b),
            "cfa_exact": rank(exact, ce), "cfa_exact_bge": rank(exact, b),
            "gold_oracle": rank(gold, ce), "gold_cfa_transfer": rank(gold * 10 + tr5, ce), "gold_cfa_exact": rank(gold * 10 + exact, ce),
            "signed_oracle": rank(role, ce),
            "tgt_gold": rank(tgt["gold"][q], ce), "tgt_cfa": rank(tgt["cfa"][q], ce), "tgt_signed": rank(tgt["signed"][q], ce),
            "tgt_ce": rank(tgt["ce"][q], ce), "signed_support": ssp[q]["rankings"]["signed_support"],
        }
        if cc_from_loo and "contextcite32" in L:
            rk["contextcite_stable"] = rank(L["contextcite32"], ce); rk["contextcite16_stable"] = rank(L["contextcite16"], ce)
        prune = sorted(i for i in range(m) if exact[i] > 0) or [rk["ce"][0]]
        rows.append({"question_group": q, "dataset": p["dataset"], "qid": p["qid"], "rankings": {k: [int(x) for x in v] for k, v in rk.items()},
                     "fixed": {"cfa_prune": prune},
                     "matched": {"ce_match_prune": ["ce", "cfa_prune"], "bge_match_prune": ["bge", "cfa_prune"], "contextcite_match_prune": ["contextcite", "cfa_prune"]},
                     "scores": {"exact": exact.tolist(), "transfer5": tr5.tolist(), "bge": b.tolist(), "ce": ce.tolist(), "role": role.tolist()}})
    out = HERE / "inputs_e1/e1_plan.jsonl"
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    print("plan rows", len(rows), "sha256", hashlib.sha256(out.read_bytes()).hexdigest())
    ex = np.concatenate([r["scores"]["exact"] for r in rows]) > 0
    print("exact flip rate %.3f; questions with no flip %d; mean prune size %.2f" % (ex.mean(), sum(not any(np.array(r["scores"]["exact"]) > 0) for r in rows), np.mean([len(r["fixed"]["cfa_prune"]) for r in rows])))


if __name__ == "__main__":
    main()
