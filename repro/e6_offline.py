"""E6a / E6b / E9 (offline, no generator calls).

E6a  answer-containment baselines for the within-gold test of Tab. 3 (six readers of
     the signed grid + the fresh HF Qwen3-4B reader of E1).
E6b  what transferred CFA adds beyond containment: stratified AUCs, question-held-out
     logistic models [bge, containment] vs [bge, containment, transferred CFA],
     and the supporting passages that do not contain the answer string.
E9   transfer with donors restricted to other model families.

Bootstrap: dataset-stratified over questions, B=2000, seed 0 (same as round 1).
Output: outputs/e6_offline.json
"""
import json, sys, warnings
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from cfa_lib import GRID_GENS, boot_indices, holm, jl, load_features, load_pools, load_signed
from e6_lib import FAMILY, NICE, READERS6, contains, contains_any, load_gemma4b_answers, load_grid_answers, overlap
warnings.filterwarnings("ignore")
B = 2000


def auc(y, s):
    y = np.asarray(y, bool); n1 = y.sum(); n0 = len(y) - n1
    if n1 == 0 or n0 == 0: return np.nan
    r = rankdata(s); return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def nm(a, bi):
    a = np.asarray(a, float)
    if np.all(np.isnan(a)): return {"mean": None, "ci95": [None, None], "n": 0}
    boots = np.array([np.nanmean(a[i]) for i in bi])
    return {"mean": float(np.nanmean(a)), "ci95": [float(x) for x in np.nanpercentile(boots, [2.5, 97.5])], "n": int((~np.isnan(a)).sum())}


def nmd(a, b, bi):
    d = np.asarray(a, float) - np.asarray(b, float); boots = np.array([np.nanmean(d[i]) for i in bi]); boots = boots[~np.isnan(boots)]
    p = 2 * min(np.mean(boots <= 0), np.mean(boots >= 0))
    return {"diff": float(np.nanmean(d)), "ci95": [float(x) for x in np.percentile(boots, [2.5, 97.5])], "p": float(max(p, 1 / len(bi))), "n": int((~np.isnan(d)).sum())}


def load_all():
    pools = load_pools(); feats, labels, names = load_features(); gem = load_gemma4b_answers()
    bge = json.load(open(HERE / "outputs/e3_bge_reranker_scores.json"))["scores"]
    S = {r: load_signed(r) for r in READERS6}; loo = {r["question_group"]: r for r in jl(HERE / "inputs_e1/e1_loo.jsonl")}
    qs = sorted(pools); D = {}; GA = load_grid_answers()
    for reader in READERS6 + ["hf-qwen3-4b"]:
        rows = []
        for q in qs:
            fr = pools[q]["fragments"]; m = len(fr); gold = np.array([f["is_gold"] for f in fr], bool)
            if reader == "hf-qwen3-4b":
                L = loo[q]; full = L["full_answer"]; cf = L["full_metrics"]["cover"]
                flip = np.array(L["delete_changes"], float)
                sup = np.array([ch and cf > dm["cover"] for dm, ch in zip(L["delete_metrics"], L["delete_changes"])], float)
                tr = np.mean([labels[g][q] for g in GRID_GENS], axis=0); donors = {}
                acd = np.mean([contains(GA[g][q], fr) for g in GRID_GENS], axis=0)
            else:
                s = S[reader][q]; sf = s["fragments"]; assert len(sf) == m and [f["is_gold"] for f in sf] == gold.tolist()
                full = s["full"]["answer"]; cf = s["full"]["cover"]
                flip = np.array([f["del_flip"] > 0 for f in sf], float); sup = np.array([f["del_role"] == "supporting" for f in sf], float)
                others = [o for o in READERS6 if o != reader]
                donors = {o: np.array([f["del_flip"] for f in S[o][q]["fragments"]], float) for o in others}
                tr = np.mean(list(donors.values()), axis=0)
                acd = np.mean([contains(S[o][q]["full"]["answer"], fr) for o in others], axis=0)
            rows.append({"q": q, "ds": pools[q]["dataset"], "fold": pools[q]["fold"], "gold": gold, "flip": flip, "sup": sup, "tr": tr, "donors": donors,
                         "grid": {g: labels[g][q] for g in GRID_GENS}, "full_correct": cf, "yesno": normalize_yesno(full),
                         "ac": contains(full, fr), "ac_donors": acd, "ac_gemma": contains(gem[q], fr), "ac_gold": contains_any(pools[q]["golds"], fr), "ov": overlap(full, fr),
                         "bge": np.array(bge[q], float), "ce": feats[q][:, names.index("ce")]})
        D[reader] = rows
    return qs, D


def normalize_yesno(a):
    from cfa_lib import normalize
    return normalize(a) in ("yes", "no")


def tb(primary, bge):  # primary with bge rank as tie-break
    return np.asarray(primary, float) * 1000 + rankdata(bge) / (len(bge) + 1)


def e6a(D, bi):
    res = {}
    for reader, rows in D.items():
        for tname in ("sup", "flip"):
            sc = {"transfer": lambda r: r["tr"], "anscont_own": lambda r: r["ac"], "anscont_own_bge": lambda r: tb(r["ac"], r["bge"]),
                  "anscont_gemma4b": lambda r: r["ac_gemma"], "anscont_gemma4b_bge": lambda r: tb(r["ac_gemma"], r["bge"]),
                  "anscont_gold_oracle": lambda r: r["ac_gold"], "anscont_donors": lambda r: r["ac_donors"], "anscont_donors_bge": lambda r: tb(r["ac_donors"], r["bge"]),
                  "sum_donorcont_transfer": lambda r: r["ac_donors"] + r["tr"], "overlap_bge": lambda r: tb(r["ov"], r["bge"]),
                  "transfer_bge": lambda r: tb(r["tr"], r["bge"]), "anscont_then_transfer": lambda r: r["ac"] * 10 + r["tr"],
                  "sum_anscont_transfer": lambda r: r["ac"] + r["tr"], "bge": lambda r: r["bge"], "ce": lambda r: r["ce"]}
            per = {k: np.array([auc(r[tname][r["gold"]], f(r)[r["gold"]]) for r in rows]) for k, f in sc.items()}
            e = {"n_informative": int((~np.isnan(per["transfer"])).sum())}
            for k in per: e[k] = nm(per[k], bi)
            for a, b in (("transfer", "anscont_own"), ("transfer", "anscont_own_bge"), ("transfer", "anscont_gemma4b_bge"), ("transfer", "bge"),
                         ("anscont_own", "bge"), ("transfer", "anscont_donors"), ("sum_donorcont_transfer", "anscont_donors"), ("anscont_donors", "anscont_own"), ("anscont_then_transfer", "anscont_own"), ("sum_anscont_transfer", "anscont_own"), ("sum_anscont_transfer", "transfer")):
                e[f"{a}-{b}"] = nmd(per[a], per[b], bi)
            # by dataset
            ds = np.array([r["ds"] for r in rows])
            e["by_dataset"] = {d: {k: {"mean": float(np.nanmean(per[k][ds == d])) if np.any(~np.isnan(per[k][ds == d])) else None, "n": int((~np.isnan(per[k][ds == d])).sum())}
                                   for k in ("transfer", "anscont_own", "bge")} for d in sorted(set(ds))}
            res[f"{reader}|{tname}"] = e
    for tname in ("sup", "flip"):
        for c in ("transfer-anscont_own", "transfer-anscont_own_bge", "sum_anscont_transfer-anscont_own", "transfer-anscont_donors", "sum_donorcont_transfer-anscont_donors"):
            adj = holm({r: res[f"{r}|{tname}"][c]["p"] for r in D})
            for r, v in adj.items(): res[f"{r}|{tname}"][c]["p_holm"] = v
    return res


def e6b_strata(D, bi):
    res = {}
    for reader, rows in D.items():
        e = {}
        for tname in ("sup", "flip"):
            # (i) within-gold AUC of transfer where containment is uninformative among gold (all gold passages share the same containment value)
            def wg(r, key, cond):
                g = r["gold"]; return auc(r[tname][g], r[key][g] if isinstance(key, str) else key(r)[g]) if cond(r) else np.nan
            unin = lambda r: len(set(r["ac"][r["gold"]])) == 1
            info = lambda r: len(set(r["ac"][r["gold"]])) > 1
            t_un = np.array([wg(r, "tr", unin) for r in rows]); b_un = np.array([wg(r, "bge", unin) for r in rows])
            t_in = np.array([wg(r, "tr", info) for r in rows]); a_in = np.array([wg(r, "ac", info) for r in rows])
            e[f"{tname}_within_gold_containment_uninformative"] = {"transfer": nm(t_un, bi), "bge": nm(b_un, bi), "transfer-bge": nmd(t_un, b_un, bi)}
            e[f"{tname}_within_gold_containment_informative"] = {"transfer": nm(t_in, bi), "anscont": nm(a_in, bi), "transfer-anscont": nmd(t_in, a_in, bi)}
            # (ii) all passages of a question, by containment stratum
            for sname, mask in (("noncontaining", lambda r: r["ac"] == 0), ("containing", lambda r: r["ac"] == 1), ("all", lambda r: np.ones(len(r["ac"]), bool))):
                t = np.array([auc(r[tname][mask(r)], r["tr"][mask(r)]) for r in rows]); b = np.array([auc(r[tname][mask(r)], r["bge"][mask(r)]) for r in rows])
                a = np.array([auc(r[tname][mask(r)], tb(r["ac"], r["bge"])[mask(r)]) for r in rows]); tbb = np.array([auc(r[tname][mask(r)], tb(r["tr"], r["bge"])[mask(r)]) for r in rows])
                cb = np.array([auc(r[tname][mask(r)], (r["ac"] * 10 + r["tr"]) * 1000 + rankdata(r["bge"])[...] / 100)[()] if False else auc(r[tname][mask(r)], tb(r["ac"] * 10 + r["tr"], r["bge"])[mask(r)]) for r in rows])
                e[f"{tname}_allpassages_{sname}"] = {"transfer": nm(t, bi), "transfer_bge": nm(tbb, bi), "bge": nm(b, bi), "anscont_bge": nm(a, bi), "anscont_then_transfer_bge": nm(cb, bi),
                                                     "transfer_bge-bge": nmd(tbb, b, bi), "transfer_bge-anscont_bge": nmd(tbb, a, bi), "anscont_then_transfer_bge-anscont_bge": nmd(cb, a, bi)}
        res[reader] = e
    return res


def e6b_disagreement(D, bi):
    """Supporting passages that do not contain the answer string (bridge hops) and who retrieves them."""
    res = {}
    for reader, rows in D.items():
        e = {}
        for d in ("hotpotqa", "2wiki", "musique", "all"):
            rr = [r for r in rows if d in ("all", r["ds"])]
            sup = np.concatenate([r["sup"] for r in rr]).astype(bool); ac = np.concatenate([r["ac"] for r in rr]); gold = np.concatenate([r["gold"] for r in rr])
            acg = np.concatenate([r["ac_gold"] for r in rr]); tr = np.concatenate([r["tr"] for r in rr]); flip = np.concatenate([r["flip"] for r in rr]).astype(bool)
            hid = sup & (ac == 0)
            def topk(score_fn, k_extra=0):  # share of non-containing supporting passages ranked within the top-|gold| of the question
                hit = tot = 0
                for r in rr:
                    h = (r["sup"] > 0) & (r["ac"] == 0)
                    if not h.any(): continue
                    k = int(r["gold"].sum()) + k_extra; order = np.argsort(-score_fn(r), kind="stable")[:k]
                    hit += h[order].sum(); tot += h.sum()
                return hit / tot if tot else None
            e[d] = {"n_supporting": int(sup.sum()), "n_supporting_not_containing_own_answer": int(hid.sum()),
                    "share_supporting_not_containing_own_answer": float(hid.sum() / max(1, sup.sum())),
                    "n_gold": int(gold.sum()), "share_gold_not_containing_gold_answer": float(((acg == 0) & gold).sum() / max(1, gold.sum())),
                    "share_gold_not_containing_own_answer": float(((ac == 0) & gold).sum() / max(1, gold.sum())),
                    "share_flip_not_containing_own_answer": float(((ac == 0) & flip).sum() / max(1, flip.sum())),
                    "share_noncontaining_supporting_that_are_gold": float((hid & gold).sum() / max(1, hid.sum())),
                    "mean_transfer_on_noncontaining_supporting": float(tr[hid].mean()) if hid.any() else None,
                    "share_noncontaining_supporting_with_transfer_gt0": float((tr[hid] > 0).mean()) if hid.any() else None,
                    "mean_transfer_on_noncontaining_nonsupporting": float(tr[(~sup) & (ac == 0)].mean()),
                    "recall_in_top_ngold": {"bge": topk(lambda r: r["bge"]), "anscont_bge": topk(lambda r: tb(r["ac"], r["bge"])),
                                            "transfer_bge": topk(lambda r: tb(r["tr"], r["bge"])), "anscont_then_transfer_bge": topk(lambda r: tb(r["ac"] * 10 + r["tr"], r["bge"]))}}
        # question-level bootstrap of the headline share (all datasets) and of recall difference transfer vs containment
        num = np.array([((r["sup"] > 0) & (r["ac"] == 0)).sum() for r in rows], float); den = np.array([r["sup"].sum() for r in rows], float)
        boots = np.array([num[i].sum() / max(1, den[i].sum()) for i in bi]); e["all"]["share_ci95"] = [float(x) for x in np.percentile(boots, [2.5, 97.5])]
        def hits(fn):
            out = []
            for r in rows:
                h = (r["sup"] > 0) & (r["ac"] == 0); k = int(r["gold"].sum()); order = np.argsort(-fn(r), kind="stable")[:k]; out.append(h[order].sum())
            return np.array(out, float)
        for a, bname, fa, fb in (("transfer_bge", "anscont_bge", lambda r: tb(r["tr"], r["bge"]), lambda r: tb(r["ac"], r["bge"])),
                                 ("anscont_then_transfer_bge", "anscont_bge", lambda r: tb(r["ac"] * 10 + r["tr"], r["bge"]), lambda r: tb(r["ac"], r["bge"])),
                                 ("transfer_bge", "bge", lambda r: tb(r["tr"], r["bge"]), lambda r: r["bge"])):
            ha, hb = hits(fa), hits(fb); boots = np.array([(ha[i].sum() - hb[i].sum()) / max(1, num[i].sum()) for i in bi])
            e["all"][f"recall_diff_{a}-{bname}"] = {"diff": float((ha.sum() - hb.sum()) / max(1, num.sum())), "ci95": [float(x) for x in np.percentile(boots, [2.5, 97.5])]}
        res[reader] = e
    return res


def e6b_models(D, bi):
    """Question-held-out logistic models; OOF scores pooled, then evaluated."""
    MODELS = {"bge": ["bge", "bger"], "bge+anscont": ["bge", "bger", "ac"], "bge+transfer": ["bge", "bger", "tr"], "bge+anscont+transfer": ["bge", "bger", "ac", "tr"],
              "bge+anscont+transfer+inter": ["bge", "bger", "ac", "tr", "ac*tr"], "bge+anscont+qmean": ["bge", "bger", "ac", "qmean"],
              "bge+donorcont": ["bge", "bger", "acd"], "bge+donorcont+transfer": ["bge", "bger", "acd", "tr"]}
    res = {}
    for reader, rows in D.items():
        nq = len(rows); qi = np.concatenate([[k] * len(r["ac"]) for k, r in enumerate(rows)]); fold = np.concatenate([[r["fold"]] * len(r["ac"]) for r in rows])
        col = {"bge": np.concatenate([r["bge"] for r in rows]), "bger": np.concatenate([rankdata(r["bge"]) / len(r["bge"]) for r in rows]),
               "ac": np.concatenate([r["ac"] for r in rows]), "tr": np.concatenate([r["tr"] for r in rows])}
        col["ac*tr"] = col["ac"] * col["tr"]; col["acd"] = np.concatenate([r["ac_donors"] for r in rows])
        col["qmean"] = np.concatenate([np.full(len(r["tr"]), r["tr"].mean()) for r in rows]); gold = np.concatenate([r["gold"] for r in rows])
        starts = np.concatenate([[0], np.cumsum([len(r["ac"]) for r in rows])]); sl = [np.arange(starts[k], starts[k + 1]) for k in range(nq)]
        for tname in ("sup", "flip"):
            y = np.concatenate([r[tname] for r in rows]); P = {}
            for mname, cols in MODELS.items():
                X = np.stack([col[c] for c in cols], axis=1); p = np.zeros(len(y))
                for f in range(5):
                    tr_, te = fold != f, fold == f
                    sc = StandardScaler().fit(X[tr_]); p[te] = LogisticRegression(C=1.0, max_iter=1000).fit(sc.transform(X[tr_]), y[tr_]).decision_function(sc.transform(X[te]))
                P[mname] = p
            def metrics_for(p, idx=None):
                ii = np.concatenate([sl[k] for k in idx]) if idx is not None else np.arange(len(y))
                return auc(y[ii], p[ii]), average_precision_score(y[ii], p[ii])
            wq = {m: np.array([auc(y[s], P[m][s]) for s in sl]) for m in P}
            wg = {m: np.array([auc(y[s][gold[s]], P[m][s][gold[s]]) for s in sl]) for m in P}
            e = {"base_rate": float(y.mean())}
            pooled = {m: metrics_for(P[m]) for m in P}
            bidx = bi[:1000]
            pb = {m: np.array([metrics_for(P[m], i) for i in bidx]) for m in P}
            for m in P:
                e[m] = {"auc_pooled": {"mean": float(pooled[m][0]), "ci95": [float(x) for x in np.percentile(pb[m][:, 0], [2.5, 97.5])]},
                        "ap_pooled": {"mean": float(pooled[m][1]), "ci95": [float(x) for x in np.percentile(pb[m][:, 1], [2.5, 97.5])]},
                        "auc_within_question": nm(wq[m], bi), "auc_within_gold": nm(wg[m], bi)}
            for a, b in (("bge+anscont+transfer", "bge+anscont"), ("bge+anscont+transfer+inter", "bge+anscont"), ("bge+anscont+transfer", "bge+transfer"), ("bge+anscont", "bge"), ("bge+transfer", "bge"), ("bge+transfer", "bge+anscont"),
                         ("bge+anscont+transfer", "bge+anscont+qmean"), ("bge+donorcont+transfer", "bge+donorcont"), ("bge+donorcont", "bge+anscont")):
                d = pb[a] - pb[b]
                pv = lambda x: float(max(2 * min(np.mean(x <= 0), np.mean(x >= 0)), 1 / len(x)))
                e[f"{a} - {b}"] = {"d_auc_pooled": {"diff": float(pooled[a][0] - pooled[b][0]), "ci95": [float(x) for x in np.percentile(d[:, 0], [2.5, 97.5])], "p": pv(d[:, 0])},
                                   "d_ap_pooled": {"diff": float(pooled[a][1] - pooled[b][1]), "ci95": [float(x) for x in np.percentile(d[:, 1], [2.5, 97.5])], "p": pv(d[:, 1])},
                                   "d_auc_within_question": nmd(wq[a], wq[b], bi), "d_auc_within_gold": nmd(wg[a], wg[b], bi)}
            res[f"{reader}|{tname}"] = e
    for tname in ("sup", "flip"):
        for metric in ("d_auc_pooled", "d_ap_pooled", "d_auc_within_question", "d_auc_within_gold"):
            for c in ("bge+anscont+transfer - bge+anscont", "bge+anscont+transfer - bge+anscont+qmean", "bge+donorcont+transfer - bge+donorcont"):
                adj = holm({r: res[f"{r}|{tname}"][c][metric]["p"] for r in D})
                for r, v in adj.items(): res[f"{r}|{tname}"][c][metric]["p_holm"] = v
    return res


def e9(D, bi):
    res = {}
    for reader in READERS6:
        rows = D[reader]; fam = FAMILY(reader); others = [o for o in READERS6 if o != reader]; out_fam = [o for o in others if FAMILY(o) != fam]
        sets = {"all_other_readers": others, "other_families_only": out_fam, "same_family_only": [o for o in others if FAMILY(o) == fam]}
        for o in out_fam: sets[f"single:{o}"] = [o]
        e = {"donors": {k: v for k, v in sets.items()}}
        for tname in ("sup", "flip"):
            per = {k: np.array([auc(r[tname][r["gold"]], np.mean([r["donors"][o] for o in v], axis=0)[r["gold"]]) if v else np.nan for r in rows]) for k, v in sets.items()}
            per["gemma_only_main_grid"] = np.array([auc(r[tname][r["gold"]], np.mean([r["grid"][g] for g in ("ollama-gemma3-4b", "ollama-gemma3-12b")], axis=0)[r["gold"]]) for r in rows])
            per["bge"] = np.array([auc(r[tname][r["gold"]], r["bge"][r["gold"]]) for r in rows]); per["anscont_own"] = np.array([auc(r[tname][r["gold"]], r["ac"][r["gold"]]) for r in rows])
            t = {k: nm(v, bi) for k, v in per.items()}
            t["other_families_only-bge"] = nmd(per["other_families_only"], per["bge"], bi)
            t["other_families_only-all_other_readers"] = nmd(per["other_families_only"], per["all_other_readers"], bi)
            t["other_families_only-anscont_own"] = nmd(per["other_families_only"], per["anscont_own"], bi)
            t["gemma_only_main_grid-bge"] = nmd(per["gemma_only_main_grid"], per["bge"], bi)
            e[tname] = t
        res[reader] = e
    adj = holm({r: res[r]["sup"]["other_families_only-bge"]["p"] for r in res})
    for r, v in adj.items(): res[r]["sup"]["other_families_only-bge"]["p_holm"] = v
    return res


def main():
    qs, D = load_all(); ds = np.array([q.split(":")[0] for q in qs]); bi = boot_indices(ds, B)
    out = {"bootstrap": {"B": B, "seed": 0, "unit": "question, stratified by dataset"}, "n_questions": len(qs)}
    out["descriptives"] = {r: {"full_correct": float(np.mean([x["full_correct"] for x in rows])), "share_yesno_answers": float(np.mean([x["yesno"] for x in rows])),
                               "share_questions_no_passage_contains_own_answer": float(np.mean([x["ac"].sum() == 0 for x in rows])),
                               "mean_passages_containing_own_answer": float(np.mean([x["ac"].sum() for x in rows])),
                               "share_nongold_among_containing": float(sum(((x["ac"] > 0) & ~x["gold"]).sum() for x in rows) / max(1, sum(x["ac"].sum() for x in rows)))}
                           for r, rows in D.items()}
    out["e6a_within_gold"] = e6a(D, bi); print("e6a done", flush=True)
    out["e6b_strata"] = e6b_strata(D, bi); print("strata done", flush=True)
    out["e6b_disagreement"] = e6b_disagreement(D, bi); print("disagreement done", flush=True)
    out["e6b_models"] = e6b_models(D, bi); print("models done", flush=True)
    out["e9_leave_family_out"] = e9(D, bi)
    (HERE / "outputs/e6_offline.json").write_text(json.dumps(out, indent=1) + "\n")
    f = lambda x: "  nan" if x["mean"] is None else f"{x['mean']:.3f}"
    print("\nE6a within-gold AUC, target supporting")
    for r in D:
        e = out["e6a_within_gold"][f"{r}|sup"]; d = e["transfer-anscont_own"]
        print(f"{NICE[r]:24s} n={e['n_informative']:3d} transfer {f(e['transfer'])} anscont {f(e['anscont_own'])} +bge {f(e['anscont_own_bge'])} gemma {f(e['anscont_gemma4b'])} gemma+bge {f(e['anscont_gemma4b_bge'])} gold {f(e['anscont_gold_oracle'])} ovl+bge {f(e['overlap_bge'])} "
              f"ac>tr {f(e['anscont_then_transfer'])} sum {f(e['sum_anscont_transfer'])} bge {f(e['bge'])} | tr-ac {d['diff']:+.3f} [{d['ci95'][0]:+.3f},{d['ci95'][1]:+.3f}] pH {d['p_holm']:.3f}")


if __name__ == "__main__":
    main()
