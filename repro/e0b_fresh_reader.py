"""E0b. Information test on a reader that never entered the grid.

Target: exact leave-one-out labels of the fresh HF Qwen3-4B reader (E1, all answers stored).
Predictors: nine retrieval features, gold indicator, bge-reranker-v2-m3 score, and the
passage-level CFA consensus of the five grid generators (none of them is the target reader).
Targets: 'flip' (answer changed; label-free) and 'support' (deletion turns a correct
answer into an incorrect one; needs the gold answer, diagnostic).
Question-held-out folds, dataset-stratified question bootstrap (B=1000, seed 0), Holm.
"""
import json, sys
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
from sklearn.ensemble import GradientBoostingClassifier

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from cfa_lib import GRID_GENS, holm, jl, load_features, load_pools

B = 1000


def auc(y, s):
    y = np.asarray(y, bool); n1 = y.sum(); n0 = len(y) - n1
    if n1 == 0 or n0 == 0: return np.nan
    r = rankdata(s); return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def ap(y, s):
    o = np.argsort(-s, kind="stable"); y = np.asarray(y, float)[o]
    return float(((np.cumsum(y) / (np.arange(len(y)) + 1)) * y).sum() / y.sum()) if y.sum() else np.nan


def main():
    pools = load_pools(); feats, labels, names = load_features()
    bge = json.load(open(HERE / "outputs/e3_bge_reranker_scores.json"))["scores"]
    loo = {r["question_group"]: r for r in jl(HERE / "inputs_e1/e1_loo.jsonl")}
    qs = np.array(sorted(pools)); rng = np.random.default_rng(0)
    Q, X, G, BG, C, QM, PERM, Y = [], [], [], [], [], [], [], {"flip": [], "support": []}
    for q in qs:
        L = loo[q]; m = len(L["delete_answers"]); gold = np.array([i in L["gold_idxs"] for i in range(m)], float)
        cons = np.mean([labels[g][q] for g in GRID_GENS], axis=0); perm = cons.copy()
        for s in (0.0, 1.0):
            t = np.where(gold == s)[0]; perm[t] = rng.permutation(cons[t])
        Q += [q] * m; X.append(feats[q]); G.append(gold); BG.append(bge[q]); C.append(cons); QM.append(np.full(m, cons.mean())); PERM.append(perm)
        Y["flip"].append(np.array(L["delete_changes"], float))
        Y["support"].append(np.array([L["full_metrics"]["cover"] > dm["cover"] for dm in L["delete_metrics"]], float))
    Q = np.array(Q); X = np.vstack(X); G, BG, C, QM, PERM = map(np.concatenate, (G, BG, C, QM, PERM))
    fold = np.array([pools[q]["fold"] for q in Q]); rows_of = {q: np.where(Q == q)[0] for q in qs}
    qds = np.array([q.split(":")[0] for q in qs])
    FE = {"F9": X, "F9+gold": np.column_stack([X, G]), "F9+gold+bge": np.column_stack([X, G, BG]),
          "F9+gold+bge+qmean": np.column_stack([X, G, BG, QM]), "F9+gold+bge+perm": np.column_stack([X, G, BG, PERM]),
          "F9+gold+bge+cfa": np.column_stack([X, G, BG, C])}
    result = {"B": B, "n_questions": len(qs), "n_passages": len(Q), "targets": {}}
    for tname, y in Y.items():
        y = np.concatenate(y); P = {}
        for name, Z in FE.items():
            p = np.zeros(len(y))
            for f in range(5):
                tr, te = fold != f, fold == f
                p[te] = GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=.05, random_state=0).fit(Z[tr], y[tr]).predict_proba(Z[te])[:, 1]
            P[name] = p
        P["consensus_only"] = C; P["bge_only"] = BG; P["ce_only"] = X[:, names.index("ce")]
        def wq(name, gold_only):
            out = []
            for q in qs:
                t = rows_of[q]
                if gold_only: t = t[G[t] == 1]
                out.append(auc(y[t], P[name][t]))
            return np.array(out)
        WQ = {n: (wq(n, False), wq(n, True)) for n in P}
        def stats(si):
            out = {}
            t = np.concatenate([rows_of[q] for q in qs[si]])
            for n in P:
                out[f"{n}|auc"] = auc(y[t], P[n][t]); out[f"{n}|ap"] = ap(y[t], P[n][t])
                out[f"{n}|auc_gold"] = auc(y[t][G[t] == 1], P[n][t][G[t] == 1]); out[f"{n}|auc_nongold"] = auc(y[t][G[t] == 0], P[n][t][G[t] == 0])
                out[f"{n}|wq_auc"] = np.nanmean(WQ[n][0][si]); out[f"{n}|wq_auc_gold"] = np.nanmean(WQ[n][1][si])
            for metric in ["auc", "ap", "auc_gold", "auc_nongold", "wq_auc", "wq_auc_gold"]:
                for base in ["F9+gold+bge", "F9+gold+bge+qmean", "F9+gold+bge+perm"]:
                    out[f"delta_cfa_vs_{base}|{metric}"] = out[f"F9+gold+bge+cfa|{metric}"] - out[f"{base}|{metric}"]
            return out
        idx = np.arange(len(qs)); point = stats(idx); boots = {k: [] for k in point}
        parts = [idx[qds == d] for d in sorted(set(qds))]
        for _ in range(B):
            st = stats(np.concatenate([rng.choice(p, len(p)) for p in parts]))
            for k, v in st.items(): boots[k].append(v)
        est = {}
        for k, v in point.items():
            arr = np.array(boots[k]); e = {"value": float(v), "ci95": [float(x) for x in np.nanpercentile(arr, [2.5, 97.5])]}
            if k.startswith("delta"): e["p_boot"] = float(max(2 * min(np.mean(arr <= 0), np.mean(arr >= 0)), 1 / B))
            est[k] = e
        fam = {k: est[k]["p_boot"] for k in est if k.startswith("delta_cfa_vs_F9+gold+bge|") or k.startswith("delta_cfa_vs_F9+gold+bge+qmean|")}
        result["targets"][tname] = {"base_rate": float(y.mean()), "base_rate_gold": float(y[G == 1].mean()), "estimates": est, "holm": holm(fam)}
        print(f"\n== target {tname}: base rate {y.mean():.3f} (gold {y[G==1].mean():.3f}, non-gold {y[G==0].mean():.3f})")
        for n in P:
            print(f"  {n:20s} AUC {est[n+'|auc']['value']:.3f} AP {est[n+'|ap']['value']:.3f} gold {est[n+'|auc_gold']['value']:.3f} nongold {est[n+'|auc_nongold']['value']:.3f} withinQ {est[n+'|wq_auc']['value']:.3f} withinQ-gold {est[n+'|wq_auc_gold']['value']:.3f}")
        for k, e in est.items():
            if k.startswith("delta"): print(f"  {k:48s} {e['value']:+.3f} [{e['ci95'][0]:+.3f}, {e['ci95'][1]:+.3f}] p={e['p_boot']:.3f} holm={result['targets'][tname]['holm'].get(k, float('nan')):.3f}")
    (HERE / "outputs/e0b_fresh_reader.json").write_text(json.dumps(result, indent=1) + "\n")


if __name__ == "__main__":
    main()
