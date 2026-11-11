"""E0. Information test: do CFA labels carry passage-level information beyond
relevance features and gold annotation?

Leave-one-generator-out on the frozen 5-generator grid (no generator calls).
For generator g we predict its deletion-sensitivity label with question-held-out
GBMs trained on
  F9            nine retrieval features
  F9+gold       + gold-support indicator
  F9+gold+qmean + question-level mean of the other four generators' labels
                  (control: question brittleness without passage identity)
  F9+gold+perm  + other-generator consensus permuted within question and gold stratum
  F9+gold+cfa   + passage-level consensus of the other four generators
Statistics: dataset-stratified, question-clustered bootstrap (B=1000, seed 0),
Holm correction over the headline family.
Output: repro/outputs/e0_information_test.json
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import rankdata
from sklearn.ensemble import GradientBoostingClassifier

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from selector_oof import load_aligned  # noqa: E402

B = 1000
SEED = 0


def auc(y, s):
    y = np.asarray(y, bool)
    n1 = y.sum(); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(s)
    return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def ap(y, s):
    order = np.argsort(-s, kind="stable"); y = np.asarray(y)[order]
    if y.sum() == 0:
        return np.nan
    prec = np.cumsum(y) / (np.arange(len(y)) + 1)
    return float((prec * y).sum() / y.sum())


def holm(pvals):
    names = list(pvals); p = np.array([pvals[k] for k in names]); order = np.argsort(p)
    adj = np.empty(len(p)); running = 0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i]); adj[i] = min(1.0, running)
    return dict(zip(names, adj.tolist()))


def main():
    X, y, ds, gen, qid, pidx, names = load_aligned()
    plan = {}
    for line in open(HERE / "inputs/fresh_selection_plan.jsonl"):
        r = json.loads(line); plan[r["question_group"]] = r
    fold = np.array([int(plan[q]["fold"]) for q in qid])
    gold = np.array([float(p in plan[q]["gold_idxs"]) for q, p in zip(qid, pidx)])
    key = np.array([f"{q}|{p}" for q, p in zip(qid, pidx)])
    gens = sorted(set(gen))
    lab = {g: dict(zip(key[gen == g], y[gen == g])) for g in gens}
    rng = np.random.default_rng(SEED)

    MODELS = ["F9", "F9+gold", "F9+gold+qmean", "F9+gold+perm", "F9+gold+cfa"]
    store = {}
    for g in gens:
        m = gen == g
        Xg, yg, fg, gg, kg, qg, dg = X[m], y[m], fold[m], gold[m], key[m], qid[m], ds[m]
        oth = np.array([np.mean([lab[h][k] for h in gens if h != g]) for k in kg])
        qmean = np.zeros(len(oth)); perm = oth.copy()
        for q in np.unique(qg):
            t = np.where(qg == q)[0]; qmean[t] = oth[t].mean()
            for stratum in (0.0, 1.0):
                tt = t[gg[t] == stratum]
                perm[tt] = rng.permutation(oth[tt])
        feats = {
            "F9": Xg, "F9+gold": np.column_stack([Xg, gg]),
            "F9+gold+qmean": np.column_stack([Xg, gg, qmean]),
            "F9+gold+perm": np.column_stack([Xg, gg, perm]),
            "F9+gold+cfa": np.column_stack([Xg, gg, oth]),
        }
        preds = {}
        for name, Z in feats.items():
            p = np.zeros(len(yg))
            for f in range(5):
                tr, te = fg != f, fg == f
                assert not (set(qg[tr]) & set(qg[te]))
                p[te] = GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=.05,
                                                   random_state=0).fit(Z[tr], yg[tr]).predict_proba(Z[te])[:, 1]
            preds[name] = p
        preds["consensus_only"] = oth  # training-free score
        preds["ce_only"] = Xg[:, names.index("ce")]
        store[g] = dict(y=yg, gold=gg, q=qg, d=dg, preds=preds)
        print(g, {k: round(auc(yg, v), 3) for k, v in preds.items()}, flush=True)

    qs = np.array(sorted(set(qid))); qds = np.array([q.split(":")[0] for q in qs])
    datasets = sorted(set(qds))
    rows_of = {g: {q: np.where(store[g]["q"] == q)[0] for q in qs} for g in gens}

    def within_q_auc(g, name, mask=None):
        s = store[g]; out = {}
        for q in qs:
            t = rows_of[g][q]
            if mask is not None:
                t = t[mask[t]]
            out[q] = auc(s["y"][t], s["preds"][name][t]) if len(t) else np.nan
        return out

    wq = {g: {name: within_q_auc(g, name) for name in MODELS + ["consensus_only", "ce_only"]} for g in gens}
    wq_gold = {g: {name: within_q_auc(g, name, store[g]["gold"] == 1) for name in MODELS + ["consensus_only", "ce_only"]} for g in gens}

    def stats(sample_qs):
        out = {}
        for scope in ["all"] + datasets:
            sq = sample_qs if scope == "all" else [q for q in sample_qs if q.split(":")[0] == scope]
            for name in MODELS + ["consensus_only", "ce_only"]:
                a, p_, ag, an, w, wg = [], [], [], [], [], []
                for g in gens:
                    s = store[g]; t = np.concatenate([rows_of[g][q] for q in sq])
                    yy, pp, gg = s["y"][t], s["preds"][name][t], s["gold"][t]
                    a.append(auc(yy, pp)); p_.append(ap(yy, pp))
                    ag.append(auc(yy[gg == 1], pp[gg == 1])); an.append(auc(yy[gg == 0], pp[gg == 0]))
                    w.append(np.nanmean([wq[g][name][q] for q in sq]))
                    wg.append(np.nanmean([wq_gold[g][name][q] for q in sq]))
                out[f"{scope}|{name}|auc"] = float(np.mean(a)); out[f"{scope}|{name}|ap"] = float(np.mean(p_))
                out[f"{scope}|{name}|auc_gold"] = float(np.mean(ag)); out[f"{scope}|{name}|auc_nongold"] = float(np.mean(an))
                out[f"{scope}|{name}|wq_auc"] = float(np.mean(w)); out[f"{scope}|{name}|wq_auc_gold"] = float(np.mean(wg))
            for metric in ["auc", "ap", "auc_gold", "auc_nongold", "wq_auc", "wq_auc_gold"]:
                for base in ["F9+gold", "F9+gold+qmean", "F9+gold+perm"]:
                    out[f"{scope}|delta_cfa_vs_{base}|{metric}"] = out[f"{scope}|F9+gold+cfa|{metric}"] - out[f"{scope}|{base}|{metric}"]
                out[f"{scope}|delta_perm_vs_F9+gold|{metric}"] = out[f"{scope}|F9+gold+perm|{metric}"] - out[f"{scope}|F9+gold|{metric}"]
                out[f"{scope}|delta_gold_vs_F9|{metric}"] = out[f"{scope}|F9+gold|{metric}"] - out[f"{scope}|F9|{metric}"]
        for g in gens:  # per-generator headline
            s = store[g]
            t = np.concatenate([rows_of[g][q] for q in sample_qs])
            out[f"gen:{g}|delta_cfa_vs_F9+gold|auc"] = auc(s["y"][t], s["preds"]["F9+gold+cfa"][t]) - auc(s["y"][t], s["preds"]["F9+gold"][t])
        return out

    point = stats(list(qs))
    boot = {k: [] for k in point}
    by_ds = {d: qs[qds == d] for d in datasets}
    for b in range(B):
        samp = np.concatenate([rng.choice(v, len(v)) for v in by_ds.values()])
        st = stats(list(samp))
        for k, v in st.items():
            boot[k].append(v)
        if (b + 1) % 100 == 0:
            print("bootstrap", b + 1, flush=True)
    result = {"B": B, "seed": SEED, "generators": gens, "n_questions": len(qs), "estimates": {}}
    for k, v in point.items():
        arr = np.array(boot[k]); lo, hi = np.nanpercentile(arr, [2.5, 97.5])
        entry = {"value": v, "ci95": [float(lo), float(hi)]}
        if "delta" in k:
            p = 2 * min(np.mean(arr <= 0), np.mean(arr >= 0)); entry["p_boot"] = float(max(p, 1 / B))
        result["estimates"][k] = entry
    family = {k: result["estimates"][k]["p_boot"] for k in
              [f"{s}|delta_cfa_vs_F9+gold|auc" for s in ["all"] + datasets] +
              [f"{s}|delta_cfa_vs_F9+gold+qmean|auc" for s in ["all"] + datasets] +
              ["all|delta_cfa_vs_F9+gold|auc_gold", "all|delta_cfa_vs_F9+gold|auc_nongold",
               "all|delta_cfa_vs_F9+gold|wq_auc", "all|delta_cfa_vs_F9+gold|wq_auc_gold"]}
    result["holm_family"] = holm(family)
    (HERE / "outputs/e0_information_test.json").write_text(json.dumps(result, indent=1) + "\n")
    E = result["estimates"]
    for scope in ["all"] + datasets:
        print(f"\n== {scope}")
        for name in MODELS + ["consensus_only", "ce_only"]:
            print(f"  {name:16s} AUC {E[f'{scope}|{name}|auc']['value']:.3f} AP {E[f'{scope}|{name}|ap']['value']:.3f} "
                  f"gold {E[f'{scope}|{name}|auc_gold']['value']:.3f} nongold {E[f'{scope}|{name}|auc_nongold']['value']:.3f} "
                  f"withinQ {E[f'{scope}|{name}|wq_auc']['value']:.3f} withinQ-gold {E[f'{scope}|{name}|wq_auc_gold']['value']:.3f}")
        for k in E:
            if k.startswith(scope + "|delta"):
                e = E[k]; print(f"  {k:55s} {e['value']:+.3f} [{e['ci95'][0]:+.3f}, {e['ci95'][1]:+.3f}] p={e['p_boot']:.3f}")
    print("\nHolm:", json.dumps(result["holm_family"], indent=1))
    for g in gens:
        e = E[f"gen:{g}|delta_cfa_vs_F9+gold|auc"]; print(g, f"{e['value']:+.3f} [{e['ci95'][0]:+.3f}, {e['ci95'][1]:+.3f}]")


if __name__ == "__main__":
    main()
