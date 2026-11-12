"""C1 (reviewer Comment 1): what deletion labels do that answer containment cannot.

Pre-registered in review/COMMENT1_PROTOCOL_RU.md (written before this script was run).
Offline, no generator calls.  Family of 7 tests (H1a,H1b,H2a,H2b,H2c,H3a,H3b), Holm.
Pooled over the six readers of the signed grid; question-clustered, dataset-stratified
bootstrap (B=2000, seed 0).  Output: outputs/c1_harm.json
"""
import json, sys
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from cfa_lib import SIGNED_API_TEACHERS, boot_indices, holm, jl, load_pools, load_signed, normalize
from e6_lib import FAMILY, NICE, READERS6, contains

B = 2000
THR = 0.4 - 1e-9


def auc(y, s):
    y = np.asarray(y, bool); n1 = y.sum(); n0 = len(y) - n1
    if n1 == 0 or n0 == 0: return np.nan
    r = rankdata(s); return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def pooled(A, bi):
    """A: readers x questions (NaN = pair not in population). Mean over defined pairs, question-clustered CI and p."""
    A = np.asarray(A, float); ok = ~np.isnan(A); S = np.where(ok, A, 0).sum(0); N = ok.sum(0)
    boots = S[bi].sum(1) / np.maximum(N[bi].sum(1), 1)
    p = 2 * min(np.mean(boots <= 0), np.mean(boots >= 0))
    return {"mean": float(S.sum() / max(N.sum(), 1)), "ci95": [float(x) for x in np.percentile(boots, [2.5, 97.5])],
            "p": float(min(1.0, max(p, 1 / len(bi)))), "n_pairs": int(N.sum()), "n_questions": int((N > 0).sum())}


def pick(score, bge, thr):
    """Index of the passage to remove (max score, ties to lower bge) or -1."""
    score = np.asarray(score, float)
    if score.max() < thr: return -1
    c = np.where(score >= score.max() - 1e-9)[0]
    return int(c[np.argmin(np.asarray(bge)[c])])


def main():
    pools = load_pools(); qs = sorted(pools); ds = np.array([pools[q]["dataset"] for q in qs])
    bge = json.load(open(HERE / "outputs/e3_bge_reranker_scores.json"))["scores"]
    S = {r: load_signed(r) for r in READERS6}
    bi = boot_indices(ds, B=B, seed=0)

    def view(reader, q):
        """Per-passage arrays of a signed-grid reader."""
        s = S[reader][q]; fr = pools[q]["fragments"]; sf = s["fragments"]; assert len(sf) == len(fr)
        cf = s["full"]["cover"]; dc = np.array([f["delete"]["cover"] for f in sf])
        return {"cf": cf, "dc": dc, "harm": (dc > cf).astype(float), "sup": (dc < cf).astype(float),
                "flip": np.array([f["del_flip"] > 0 for f in sf], float), "ans": s["full"]["answer"],
                "cont": contains(s["full"]["answer"], fr), "gold": np.array([f["is_gold"] for f in fr], bool)}

    V = {r: {q: view(r, q) for q in qs} for r in READERS6}

    def donor_scores(donors, q):
        d = [V[o][q] for o in donors]
        return {"T_harm": np.mean([x["harm"] for x in d], 0), "T_flip": np.mean([x["flip"] for x in d], 0),
                "T_sup": np.mean([x["sup"] for x in d], 0),
                "C_don_wrong": np.mean([x["cont"] * (x["cf"] == 0) for x in d], 0), "C_don": np.mean([x["cont"] for x in d], 0)}

    R, Q = len(READERS6), len(qs)
    nan = lambda: np.full((R, Q), np.nan)
    h1 = {k: nan() for k in ("T_harm", "C_don_wrong", "C_own_wrong", "T_flip", "neg_bge", "C_don", "T_harm_xfam")}
    h3 = {k: nan() for k in ("T_flip", "bge", "C_don", "T_sup_flip")}; h3s = {k: nan() for k in ("T_flip", "T_sup", "bge", "C_don")}; h3yn = {k: nan() for k in ("T_flip", "bge", "C_don")}
    pol = {k: np.zeros((R, Q)) for k in ("T_harm", "C_don_wrong", "C_own_wrong", "T_harm_gated", "C_don_wrong_gated", "T_harm_thr1", "T_harm_thr3", "C_don_wrong_thr1", "C_don_wrong_thr3", "oracle_own", "neg_bge_gated", "random_gated")}
    nrem = {k: np.zeros((R, Q)) for k in pol}
    desc = {}
    for ri, r in enumerate(READERS6):
        donors = [o for o in READERS6 if o != r]; xf = [o for o in donors if FAMILY(o) != FAMILY(r)]
        nh = nhc = nhg = nwrong = nnolit = 0
        for qi, q in enumerate(qs):
            v = V[r][q]; b = np.array(bge[q], float); d = donor_scores(donors, q); wrong = v["cf"] == 0
            own_wrong = v["cont"] * wrong
            nh += v["harm"].sum(); nhc += (v["harm"] * v["cont"]).sum(); nhg += (v["harm"] * v["gold"]).sum(); nwrong += wrong
            # H1
            if 0 < v["harm"].sum() < len(b):
                for k, s in (("T_harm", d["T_harm"]), ("C_don_wrong", d["C_don_wrong"]), ("C_own_wrong", own_wrong), ("T_flip", d["T_flip"]), ("neg_bge", -b), ("C_don", d["C_don"]),
                             ("T_harm_xfam", donor_scores(xf, q)["T_harm"])):
                    h1[k][ri, qi] = auc(v["harm"], s)
            # H2
            def apply(name, score, thr, gate=True):
                i = pick(score, b, thr) if gate else -1
                if i >= 0: pol[name][ri, qi] = v["dc"][i] - v["cf"]; nrem[name][ri, qi] = 1
            apply("T_harm", d["T_harm"], THR); apply("C_don_wrong", d["C_don_wrong"], THR)
            apply("T_harm_thr1", d["T_harm"], 0.2 - 1e-9); apply("T_harm_thr3", d["T_harm"], 0.6 - 1e-9)
            apply("C_don_wrong_thr1", d["C_don_wrong"], 0.2 - 1e-9); apply("C_don_wrong_thr3", d["C_don_wrong"], 0.6 - 1e-9)
            apply("C_own_wrong", own_wrong, 0.5); apply("T_harm_gated", d["T_harm"], THR, wrong); apply("C_don_wrong_gated", d["C_don_wrong"], THR, wrong)
            apply("oracle_own", v["harm"], 0.5); apply("neg_bge_gated", -b, -1e9, wrong)
            if wrong: pol["random_gated"][ri, qi] = float(np.mean(v["dc"] - v["cf"])); nrem["random_gated"][ri, qi] = 1  # expectation over a uniformly random passage
            # H3
            if v["cont"].sum() == 0:
                nnolit += 1
                if 0 < v["flip"].sum() < len(b):
                    for k, s in (("T_flip", d["T_flip"]), ("bge", b), ("C_don", d["C_don"])):
                        h3[k][ri, qi] = auc(v["flip"], s)
                        if normalize(v["ans"]) in ("yes", "no"): h3yn[k][ri, qi] = auc(v["flip"], s)
                if 0 < v["sup"].sum() < len(b):
                    for k, s in (("T_flip", d["T_flip"]), ("T_sup", d["T_sup"]), ("bge", b), ("C_don", d["C_don"])): h3s[k][ri, qi] = auc(v["sup"], s)
        desc[r] = {"n_wrong_full": int(nwrong), "n_harmful_passages": int(nh), "harmful_containing_own_wrong_answer": float(nhc / max(nh, 1)),
                   "harmful_gold_share": float(nhg / max(nh, 1)), "n_questions_with_harmful": int((~np.isnan(h1["T_harm"][ri])).sum()), "n_questions_no_literal_match": int(nnolit)}

    out = {"protocol": "review/COMMENT1_PROTOCOL_RU.md", "B": B, "readers": READERS6, "descriptive": desc}
    fam = {"H1a": pooled(h1["T_harm"] - h1["C_don_wrong"], bi), "H1b": pooled(h1["T_harm"] - h1["C_own_wrong"], bi),
           "H2a": pooled(pol["T_harm"] * 100, bi), "H2b": pooled((pol["T_harm"] - pol["C_don_wrong"]) * 100, bi),
           "H2c": pooled((pol["T_harm_gated"] - pol["C_own_wrong"]) * 100, bi),
           "H3a": pooled(h3["T_flip"] - h3["bge"], bi), "H3b": pooled(h3["T_flip"] - h3["C_don"], bi)}
    adj = holm({k: v["p"] for k, v in fam.items()})
    for k in fam: fam[k]["p_holm"] = adj[k]
    out["family"] = fam
    out["H1_levels"] = {k: pooled(a, bi) for k, a in h1.items()}
    out["H1_other_contrasts"] = {"T_harm-T_flip": pooled(h1["T_harm"] - h1["T_flip"], bi), "T_harm-neg_bge": pooled(h1["T_harm"] - h1["neg_bge"], bi),
                                 "T_harm_xfam-C_don_wrong": pooled(h1["T_harm_xfam"] - h1["C_don_wrong"], bi), "T_harm_xfam-C_own_wrong": pooled(h1["T_harm_xfam"] - h1["C_own_wrong"], bi),
                                 "C_own_wrong-C_don_wrong": pooled(h1["C_own_wrong"] - h1["C_don_wrong"], bi)}
    out["H2_levels_pp"] = {k: {**pooled(a * 100, bi), "share_questions_removed": float(nrem[k].mean()),
                               "helped_pp": float((a > 0).mean() * 100), "hurt_pp": float((a < 0).mean() * 100)} for k, a in pol.items()}
    out["H2_other_contrasts_pp"] = {"T_harm_gated-C_don_wrong_gated": pooled((pol["T_harm_gated"] - pol["C_don_wrong_gated"]) * 100, bi),
                                    "T_harm_thr1-C_don_wrong_thr1": pooled((pol["T_harm_thr1"] - pol["C_don_wrong_thr1"]) * 100, bi),
                                    "T_harm_thr3-C_don_wrong_thr3": pooled((pol["T_harm_thr3"] - pol["C_don_wrong_thr3"]) * 100, bi),
                                    "T_harm_gated-neg_bge_gated": pooled((pol["T_harm_gated"] - pol["neg_bge_gated"]) * 100, bi)}
    out["H3_levels"] = {k: pooled(a, bi) for k, a in h3.items() if not np.all(np.isnan(a))}
    out["H3_supporting_target"] = {**{k: pooled(a, bi) for k, a in h3s.items()}, "T_sup-bge": pooled(h3s["T_sup"] - h3s["bge"], bi), "T_sup-C_don": pooled(h3s["T_sup"] - h3s["C_don"], bi),
                                   "T_flip-bge": pooled(h3s["T_flip"] - h3s["bge"], bi), "T_flip-C_don": pooled(h3s["T_flip"] - h3s["C_don"], bi)}
    out["H3_yesno"] = {**{k: pooled(a, bi) for k, a in h3yn.items()}, "T_flip-bge": pooled(h3yn["T_flip"] - h3yn["bge"], bi), "T_flip-C_don": pooled(h3yn["T_flip"] - h3yn["C_don"], bi)}

    # per reader and per dataset (descriptive)
    one = lambda a, ri: pooled(a[ri:ri + 1], bi)
    out["by_reader"] = {r: {"H1_T_harm": one(h1["T_harm"], ri), "H1_C_don_wrong": one(h1["C_don_wrong"], ri), "H1_C_own_wrong": one(h1["C_own_wrong"], ri),
                            "H1a": one(h1["T_harm"] - h1["C_don_wrong"], ri), "H1b": one(h1["T_harm"] - h1["C_own_wrong"], ri),
                            "H2_T_harm": one(pol["T_harm"] * 100, ri), "H2_C_don_wrong": one(pol["C_don_wrong"] * 100, ri), "H2b": one((pol["T_harm"] - pol["C_don_wrong"]) * 100, ri),
                            "H2_T_harm_gated": one(pol["T_harm_gated"] * 100, ri), "H2_C_own_wrong": one(pol["C_own_wrong"] * 100, ri), "H2c": one((pol["T_harm_gated"] - pol["C_own_wrong"]) * 100, ri),
                            "H3_T_flip": one(h3["T_flip"], ri), "H3a": one(h3["T_flip"] - h3["bge"], ri), "H3b": one(h3["T_flip"] - h3["C_don"], ri)} for ri, r in enumerate(READERS6)}
    out["by_dataset"] = {}
    for dname in sorted(set(ds)):
        m = np.where(ds == dname, 1.0, np.nan)[None, :]
        out["by_dataset"][dname] = {"H1a": pooled((h1["T_harm"] - h1["C_don_wrong"]) * m, bi), "H1b": pooled((h1["T_harm"] - h1["C_own_wrong"]) * m, bi),
                                    "H2a": pooled(pol["T_harm"] * 100 * m, bi), "H2b": pooled((pol["T_harm"] - pol["C_don_wrong"]) * 100 * m, bi),
                                    "H2c": pooled((pol["T_harm_gated"] - pol["C_own_wrong"]) * 100 * m, bi),
                                    "H3a": pooled((h3["T_flip"] - h3["bge"]) * m, bi), "H3b": pooled((h3["T_flip"] - h3["C_don"]) * m, bi)}

    # ---- independent evaluation runs (rules never see the answers of the evaluation run)
    fresh = {"openai-gpt-4.1-mini": {r["question_group"]: r for r in jl(HERE / "inputs_e6/e7_loo_gpt.jsonl")},
             "gemini-gemini-3.5-flash-lite": {r["question_group"]: r for r in jl(HERE / "inputs_e6/e7_loo_gemini.jsonl")},
             "hf-qwen3-4b": {r["question_group"]: r for r in jl(HERE / "inputs_e1/e1_loo.jsonl")}}
    indep = {}; allP = {}
    for name, F in fresh.items():
        a_run = "ollama-qwen3-4b-instruct" if name == "hf-qwen3-4b" else name
        donors = SIGNED_API_TEACHERS if name == "hf-qwen3-4b" else [o for o in READERS6 if o != name]
        P = {k: np.zeros((1, Q)) for k in ("T_harm", "C_don_wrong", "T_harm_gatedA", "C_don_wrong_gatedA", "C_own_wrong_A", "own_harm_A", "T_harm_gatedB", "C_own_wrong_B", "neg_bge_gatedA", "random_gatedA")}
        A1 = {k: np.full((1, Q), np.nan) for k in ("T_harm", "C_don_wrong", "C_own_wrong_A", "own_harm_A", "C_own_wrong_B")}
        rep = [0, 0]
        for qi, q in enumerate(qs):
            f = F[q]; assert f["gold_idxs"] == pools[q]["gold_idxs"]
            cfB = f["full_metrics"]["cover"]; dcB = np.array([m["cover"] for m in f["delete_metrics"]]); harmB = (dcB > cfB).astype(float)
            b = np.array(bge[q], float); d = donor_scores(donors, q); a = V[a_run][q]; wrongA = a["cf"] == 0
            contB = contains(f["full_answer"], pools[q]["fragments"]) * (cfB == 0)
            rep[0] += (a["harm"] * harmB).sum(); rep[1] += a["harm"].sum()
            def apply(k, score, thr, gate=True):
                i = pick(score, b, thr) if gate else -1
                if i >= 0: P[k][0, qi] = dcB[i] - cfB
            apply("T_harm", d["T_harm"], THR); apply("C_don_wrong", d["C_don_wrong"], THR)
            apply("T_harm_gatedA", d["T_harm"], THR, wrongA); apply("C_don_wrong_gatedA", d["C_don_wrong"], THR, wrongA)
            apply("C_own_wrong_A", a["cont"] * wrongA, 0.5); apply("own_harm_A", a["harm"], 0.5)
            apply("T_harm_gatedB", d["T_harm"], THR, cfB == 0); apply("C_own_wrong_B", contB, 0.5)
            apply("neg_bge_gatedA", -b, -1e9, wrongA)
            if wrongA: P["random_gatedA"][0, qi] = float(np.mean(dcB - cfB))
            if 0 < harmB.sum() < len(b):
                for k, s in (("T_harm", d["T_harm"]), ("C_don_wrong", d["C_don_wrong"]), ("C_own_wrong_A", a["cont"] * wrongA), ("own_harm_A", a["harm"]), ("C_own_wrong_B", contB)):
                    A1[k][0, qi] = auc(harmB, s)
        allP[name] = P
        indep[name] = {"harm_label_replication_A_to_B": rep[0] / max(rep[1], 1), "n_harm_A": int(rep[1]),
                       "policy_pp": {k: pooled(v * 100, bi) for k, v in P.items()},
                       "policy_contrasts_pp": {"T_harm-C_don_wrong": pooled((P["T_harm"] - P["C_don_wrong"]) * 100, bi),
                                               "T_harm_gatedA-C_own_wrong_A": pooled((P["T_harm_gatedA"] - P["C_own_wrong_A"]) * 100, bi),
                                               "T_harm_gatedB-C_own_wrong_B": pooled((P["T_harm_gatedB"] - P["C_own_wrong_B"]) * 100, bi),
                                               "own_harm_A-C_own_wrong_A": pooled((P["own_harm_A"] - P["C_own_wrong_A"]) * 100, bi),
                                               "T_harm-own_harm_A": pooled((P["T_harm"] - P["own_harm_A"]) * 100, bi)},
                       "auc": {k: pooled(v, bi) for k, v in A1.items()},
                       "auc_contrasts": {"T_harm-C_don_wrong": pooled(A1["T_harm"] - A1["C_don_wrong"], bi), "T_harm-C_own_wrong_B": pooled(A1["T_harm"] - A1["C_own_wrong_B"], bi),
                                         "T_harm-own_harm_A": pooled(A1["T_harm"] - A1["own_harm_A"], bi)}}
    out["independent_runs"] = indep

    # ---- POST HOC (added after the pre-registered family was computed; exploratory, no Holm, no confirmatory claims)
    def ratio(num, den):
        n, d = num.sum(0), den.sum(0); boots = n[bi].sum(1) / np.maximum(d[bi].sum(1), 1e-9)
        return {"value": float(n.sum() / max(d.sum(), 1e-9)), "ci95": [float(x) for x in np.percentile(boots, [2.5, 97.5])], "n_removed": float(d.sum())}
    st = lambda k: np.concatenate([allP[n][k] for n in allP], 0)
    out["posthoc"] = {
        "note": "exploratory; defined after seeing the family results",
        "precision_per_removal_six_readers": {k: ratio((pol[k] > 0).astype(float), nrem[k]) for k in ("T_harm", "T_harm_gated", "C_own_wrong", "C_don_wrong", "C_don_wrong_gated", "neg_bge_gated")},
        "harm_rate_per_removal_six_readers": {k: ratio((pol[k] < 0).astype(float), nrem[k]) for k in ("T_harm", "C_don_wrong")},
        "random_gated_pp_six_readers": pooled(pol["random_gated"] * 100, bi),
        "T_harm_gated-random_gated_pp": pooled((pol["T_harm_gated"] - pol["random_gated"]) * 100, bi),
        "C_own_wrong-random_gated_pp": pooled((pol["C_own_wrong"] - pol["random_gated"]) * 100, bi),
        "independent_pooled_3_readers_pp": {**{k: pooled(st(k) * 100, bi) for k in ("own_harm_A", "C_own_wrong_A", "T_harm_gatedA", "neg_bge_gatedA", "random_gatedA", "T_harm")},
                                            "own_harm_A-C_own_wrong_A": pooled((st("own_harm_A") - st("C_own_wrong_A")) * 100, bi),
                                            "own_harm_A-neg_bge_gatedA": pooled((st("own_harm_A") - st("neg_bge_gatedA")) * 100, bi),
                                            "own_harm_A-random_gatedA": pooled((st("own_harm_A") - st("random_gatedA")) * 100, bi),
                                            "own_harm_A-T_harm_gatedA": pooled((st("own_harm_A") - st("T_harm_gatedA")) * 100, bi),
                                            "T_harm_gatedA-random_gatedA": pooled((st("T_harm_gatedA") - st("random_gatedA")) * 100, bi),
                                            "T_harm_gatedA-C_own_wrong_A": pooled((st("T_harm_gatedA") - st("C_own_wrong_A")) * 100, bi),
                                            "C_own_wrong_A-random_gatedA": pooled((st("C_own_wrong_A") - st("random_gatedA")) * 100, bi)},
        "independent_by_reader_pp": {n: {k: pooled(allP[n][k] * 100, bi) for k in ("neg_bge_gatedA", "random_gatedA")} | {"own_harm_A-random_gatedA": pooled((allP[n]["own_harm_A"] - allP[n]["random_gatedA"]) * 100, bi), "share_removed_own_harm_A": float((np.array([V["ollama-qwen3-4b-instruct" if n == "hf-qwen3-4b" else n][q]["harm"].sum() > 0 for q in qs])).mean())} for n in allP}}

    # ---- cost amortization (arithmetic)
    m1 = float(np.mean([len(pools[q]["fragments"]) for q in qs]) + 1)
    out["cost"] = {"calls_per_donor_loo": m1, "amortized_per_target_reader": {str(N): {"transferred_cfa_5_donors": 5 * m1 / N, "donor_containment_5_donors": 5 / N, "own_containment": 1.0} for N in (1, 3, 10, 30, 100)},
                   "ratio_transfer_to_donor_containment": m1, "breakeven_N_vs_own_containment": 5 * m1}
    json.dump(out, open(HERE / "outputs/c1_harm.json", "w"), indent=1)

    f3 = lambda e: f"{e['mean']:+.3f} [{e['ci95'][0]:+.3f}; {e['ci95'][1]:+.3f}] p={e['p']:.4f}" + (f" pH={e['p_holm']:.4f}" if "p_holm" in e else "") + f" n={e['n_pairs']}/{e['n_questions']}q"
    print("DESCRIPTIVE"); [print(" ", NICE[r], v) for r, v in desc.items()]
    print("FAMILY"); [print(" ", k, f3(v)) for k, v in fam.items()]
    for sec in ("H1_levels", "H1_other_contrasts", "H2_levels_pp", "H2_other_contrasts_pp", "H3_levels", "H3_supporting_target", "H3_yesno"):
        print(sec); [print("  ", k, f3(v), {x: round(v[x], 3) for x in ("share_questions_removed", "helped_pp", "hurt_pp") if x in v}) for k, v in out[sec].items()]
    print("BY READER")
    for r, e in out["by_reader"].items():
        print(" ", NICE[r]); [print("    ", k, f3(v)) for k, v in e.items()]
    print("BY DATASET")
    for r, e in out["by_dataset"].items():
        print(" ", r); [print("    ", k, f3(v)) for k, v in e.items()]
    print("INDEPENDENT RUNS")
    for r, e in indep.items():
        print(" ", r, "replication", round(e["harm_label_replication_A_to_B"], 3), "of", e["n_harm_A"])
        for sec in ("policy_pp", "policy_contrasts_pp", "auc", "auc_contrasts"):
            print("   ", sec); [print("      ", k, f3(v)) for k, v in e[sec].items()]
    print("POSTHOC"); print(json.dumps(out["posthoc"], indent=1))
    print("COST", json.dumps(out["cost"]))


if __name__ == "__main__":
    main()
