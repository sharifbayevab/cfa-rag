"""Round-2 analysis of the token-budget sweeps (E6c, E7, E8 full-context family).

  python3 repro/e6_analyze.py qwen          # inputs_e6/e6_sweep.jsonl (Spark)          -> outputs/e6_qwen_analysis.json
  python3 repro/e6_analyze.py qwen_donors   # inputs_e6/e6_sweep_donors.jsonl (Spark)   -> outputs/e6_qwen_donors_analysis.json
  python3 repro/e6_analyze.py gpt|gemini    # outputs/e6_<stem>_answers.jsonl           -> outputs/e6_<reader>_analysis.json
  python3 repro/e6_analyze.py qwen_r1       # inputs_e1/e1_sweep.jsonl (round 1), only the full-context Holm family

Dataset-stratified paired question bootstrap, B=2000 (10000 for the full-context
family), seed 0; Holm within each family.  Metric: cover-EM.
"""
import json, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from cfa_lib import BUDGETS, boot_indices, holm, jl, mean_ci
from cfa_lib import paired as _paired


def paired(a, b, bi):
    r = _paired(a, b, bi); r["p"] = min(1.0, r["p"]); return r

ORACLE = ("gold_", "signed_oracle", "goldans_oracle")
FAMILIES = {
    "cfa_vs_containment": [("cfa_transfer_bge", "anscont_bge"), ("cfa_exact_bge", "anscont_bge"), ("anscont_transfer_bge", "anscont_bge"),
                           ("sum_anscont_transfer_bge", "anscont_bge"), ("transfer_anscont_bge", "anscont_bge"), ("anscont_exact_bge", "anscont_bge"),
                           ("sum_anscont_exact_bge", "anscont_bge"), ("transfer_anscont_bge", "cfa_transfer_bge"), ("sum_anscont_transfer_bge", "cfa_transfer_bge")],
    "vs_bge": [("anscont_bge", "bge"), ("anscont_gemma4b_bge", "bge"), ("ansoverlap_bge", "bge"), ("goldans_oracle_bge", "bge"), ("cfa_transfer_bge", "bge"),
               ("cfa_exact_bge", "bge"), ("sum_anscont_transfer_bge", "bge"), ("anscont_gemma4b_bge", "anscont_bge"), ("goldans_oracle_bge", "anscont_bge")],
    "donor_only": [("anscont_donors_bge", "bge"), ("cfa_transfer_bge", "anscont_donors_bge"), ("sum_donorcont_transfer_bge", "anscont_donors_bge"),
                   ("anscont_donors_bge", "anscont_bge"), ("sum_donorcont_transfer_bge", "cfa_transfer_bge")],
    "exact_fresh": [("cfa_exact", "ce"), ("cfa_transfer", "ce"), ("cfa_exact", "cfa_transfer"), ("cfa_exact_bge", "cfa_transfer_bge"), ("cfa_exact_bge", "cfa_exact_stale_bge"),
                    ("cfa_exact", "cfa_exact_stale"), ("gold_cfa_exact", "gold_oracle"), ("gold_cfa_transfer", "gold_oracle")],
}
SRC = {"qwen": "inputs_e6/e6_sweep.jsonl", "qwen_donors": "inputs_e6/e6_sweep_donors.jsonl", "gpt": "outputs/e6_e4_gpt41mini_answers.jsonl",
       "gemini": "outputs/e6_e4_gemini35fl_answers.jsonl", "qwen_r1": "inputs_e1/e1_sweep.jsonl"}


def main(which):
    rows = sorted(jl(HERE / SRC[which]), key=lambda r: r["question_group"]); ds = np.array([r["dataset"] for r in rows])
    bi = boot_indices(ds); bi10 = boot_indices(ds, 10000)
    arms = sorted({k.rsplit("_tok", 1)[0] for k in rows[0]["methods"] if "_tok" in k}); dsets = sorted(set(ds))
    def col(name, key="cover"): return np.array([r["methods"][name][key] for r in rows], float)
    full = col("full")
    res = {"source": SRC[which], "n": len(rows), "arms": arms, "budgets": BUDGETS, "table": {}, "contrasts": {}, "fixed": {},
           "full": {**mean_ci(full, bi), "f1": float(col("full", "f1").mean()), "em": float(col("full", "em").mean()), "tokens": float(col("full", "context_tokens").mean()),
                    "by_dataset": {d: float(full[ds == d].mean()) for d in dsets}}}
    for arm in arms:
        avg = np.mean([col(f"{arm}_tok{b}") for b in BUDGETS], axis=0)
        res["table"][arm] = {"avg": mean_ci(avg, bi), "avg_f1": float(np.mean([col(f"{arm}_tok{b}", "f1") for b in BUDGETS])), "avg_em": float(np.mean([col(f"{arm}_tok{b}", "em") for b in BUDGETS])),
                             "by_dataset_avg": {d: float(avg[ds == d].mean()) for d in dsets}}
        for b in BUDGETS:
            n = f"{arm}_tok{b}"
            res["table"][arm][str(b)] = {**mean_ci(col(n), bi), "f1": float(col(n, "f1").mean()), "tokens": float(col(n, "context_tokens").mean()),
                                         "gold_recall": float(col(n, "gold_recall").mean()), "vs_full": paired(col(n), full, bi10)}
    for fam, pairs in FAMILIES.items():
        pv = {}
        for a, b in pairs:
            if a not in arms or b not in arms: continue
            key = f"{a}-{b}"; c = res["contrasts"][key] = {"family": fam}
            for bud in BUDGETS:
                c[str(bud)] = paired(col(f"{a}_tok{bud}"), col(f"{b}_tok{bud}"), bi); pv[f"{key}@{bud}"] = c[str(bud)]["p"]
            aa = np.mean([col(f"{a}_tok{x}") for x in BUDGETS], axis=0); bb = np.mean([col(f"{b}_tok{x}") for x in BUDGETS], axis=0)
            c["avg"] = paired(aa, bb, bi); lo = [x for x in BUDGETS if x <= 384]
            c["avg_low"] = paired(np.mean([col(f"{a}_tok{x}") for x in lo], axis=0), np.mean([col(f"{b}_tok{x}") for x in lo], axis=0), bi)
            c["by_dataset_avg"] = {d: paired(aa[ds == d], bb[ds == d], boot_indices(ds[ds == d], seed=1)) for d in dsets}
            c["same_selection_share"] = float(np.mean([[r["methods"][f"{a}_tok{x}"]["selected_idxs"] == r["methods"][f"{b}_tok{x}"]["selected_idxs"] for x in BUDGETS] for r in rows]))
        for k, v in (holm(pv) if pv else {}).items():
            key, bud = k.split("@"); res["contrasts"][key][bud]["p_holm"] = v
        pa = {k: c["avg"]["p"] for k, c in res["contrasts"].items() if c["family"] == fam}
        for k, v in (holm(pa) if pa else {}).items(): res["contrasts"][k]["avg"]["p_holm"] = v
    # ---- fixed-size arms
    fixed = [k for k in rows[0]["methods"] if "_tok" not in k and k != "full"]
    for name in fixed:
        res["fixed"][name] = {**mean_ci(col(name), bi), "tokens": float(col(name, "context_tokens").mean()),
                              "n_selected": float(np.mean([len(r["methods"][name]["selected_idxs"]) for r in rows])), "gold_recall": float(col(name, "gold_recall").mean()),
                              "vs_full": paired(col(name), full, bi10), "token_ratio_full_over_sel": float(col("full", "context_tokens").sum() / col(name, "context_tokens").sum()),
                              "by_dataset": {d: float(col(name)[ds == d].mean()) for d in dsets}}
    for a, others in (("cfa_prune", ["ce_match_prune", "bge_match_prune", "anscont_bge_match_prune", "cfa_transfer_bge_match_prune", "contextcite_match_prune", "cfa_prune_stale", "anscont_prune"]),
                      ("anscont_prune", ["bge_match_acprune"])):
        if a in res["fixed"]:
            for o in others:
                if o in res["fixed"]: res["fixed"][a][f"vs_{o}"] = paired(col(a), col(o), bi)
    # ---- E8: every label-free selection against the full context, one Holm family per run (B=10000)
    lf = [a for a in arms if not a.startswith(ORACLE) and not any(o in a for o in ORACLE)]
    fam = {f"{a}@{b}": res["table"][a][str(b)]["vs_full"]["p"] for a in lf for b in BUDGETS}
    fam.update({f"{n}@fixed": res["fixed"][n]["vs_full"]["p"] for n in fixed})
    adj = holm(fam)
    for k, v in adj.items():
        a, b = k.split("@")
        (res["fixed"][a] if b == "fixed" else res["table"][a][b])["vs_full"]["p_holm_all_arms"] = v
    for a in lf:  # narrower family: the six budgets of one arm
        for k, v in holm({str(b): res["table"][a][str(b)]["vs_full"]["p"] for b in BUDGETS}).items(): res["table"][a][k]["vs_full"]["p_holm_six_budgets"] = v
    res["full_context_family"] = {"size_all_arms": len(fam), "bootstrap_B": 10000, "label_free_arms": lf,
                                  "n_significant_after_holm": int(sum(v < .05 for v in adj.values())),
                                  "significant": {k: {"p_holm": v, "diff": (res["fixed"][k.split('@')[0]] if k.endswith("fixed") else res["table"][k.split('@')[0]][k.split('@')[1]])["vs_full"]["diff"]} for k, v in adj.items() if v < .05}}
    # ---- strata by whether any passage contains the reader's own full-context answer (known before selection)
    if which != "qwen_r1" and "anscont_bge" in arms:
        from cfa_lib import load_pools, normalize
        from e6_lib import contains
        pools = load_pools()
        if which.startswith("qwen"):
            loo = {r["question_group"]: r["full_answer"] for r in jl(HERE / "inputs_e1/e1_loo.jsonl")}; fa = [loo[r["question_group"]] for r in rows]
        else: fa = [r["methods"]["full"]["answer"] for r in rows]
        some = np.array([contains(a, pools[r["question_group"]]["fragments"]).sum() > 0 for a, r in zip(fa, rows)]); yn = np.array([normalize(a) in ("yes", "no") for a in fa])
        res["strata"] = {}
        for sname, mask in (("some_passage_contains_answer", some), ("no_passage_contains_answer", ~some), ("yes_no_answer", yn)):
            bs = boot_indices(ds[mask], seed=2); e = {"n": int(mask.sum()), "full": float(full[mask].mean()), "by_dataset_n": {d: int((ds[mask] == d).sum()) for d in dsets}}
            for a, b in (("cfa_transfer_bge", "anscont_bge"), ("sum_anscont_transfer_bge", "anscont_bge"), ("cfa_exact_bge", "anscont_bge"), ("cfa_transfer_bge", "bge"), ("anscont_bge", "bge"), ("cfa_exact_bge", "bge"),
                         ("anscont_donors_bge", "anscont_bge"), ("cfa_transfer_bge", "anscont_donors_bge")):
                if a in arms and b in arms:
                    aa = np.mean([col(f"{a}_tok{x}") for x in BUDGETS], axis=0)[mask]; bb = np.mean([col(f"{b}_tok{x}") for x in BUDGETS], axis=0)[mask]
                    e[f"{a}-{b}"] = {**paired(aa, bb, bs), "mean_a": float(aa.mean()), "mean_b": float(bb.mean())}
            res["strata"][sname] = e
    if "probe" in rows[0]:
        res["probe"] = {"flip_rate": float(sum(r["probe"]["n_flip"] for r in rows) / sum(r["pool_size"] for r in rows)),
                        "flip_rate_stale": float(sum(r["probe"]["n_flip_stale"] for r in rows) / sum(r["pool_size"] for r in rows)),
                        "label_agreement_with_stale": float(np.average([r["probe"]["label_agreement_with_stale"] for r in rows], weights=[r["pool_size"] for r in rows])),
                        "full_answer_same_as_stale": float(np.mean([r["probe"]["full_same_as_stale"] for r in rows])),
                        "questions_without_flip": int(sum(r["probe"]["n_flip"] == 0 for r in rows)), "mean_calls_exact": float(np.mean([r["pool_size"] + 1 for r in rows]))}
    (HERE / f"outputs/e6_{which}_analysis.json").write_text(json.dumps(res, indent=1) + "\n")
    pct = lambda x: f"{100*x:5.1f}"
    print(f"== {which}: n={len(rows)} full {pct(res['full']['mean'])} tokens {res['full']['tokens']:.0f}")
    if which == "qwen_r1":
        for a in ("cfa_transfer_bge", "cfa_transfer", "bge", "cfa_exact_bge", "contextcite"):
            print(f"{a:20s}", " ".join(f"{b}: {100*res['table'][a][str(b)]['vs_full']['diff']:+.1f} p={res['table'][a][str(b)]['vs_full']['p']:.4f} pH6={res['table'][a][str(b)]['vs_full']['p_holm_six_budgets']:.3f} pHall={res['table'][a][str(b)]['vs_full']['p_holm_all_arms']:.3f} |" for b in BUDGETS))
        print("family size", res["full_context_family"]["size_all_arms"], "significant after Holm:", res["full_context_family"]["n_significant_after_holm"]); return
    print(f"{'arm':28s}" + "".join(f"{b:>7d}" for b in BUDGETS) + "    avg   F1")
    for arm in sorted(arms, key=lambda a: -res["table"][a]["avg"]["mean"]):
        print(f"{arm:28s}" + "".join(f"  {pct(res['table'][arm][str(b)]['mean'])}" for b in BUDGETS) + f"  {pct(res['table'][arm]['avg']['mean'])} {pct(res['table'][arm]['avg_f1'])}")
    for key, c in res["contrasts"].items():
        s = " ".join(f"{100*c[str(b)]['diff']:+5.1f}{'*' if c[str(b)].get('p_holm', 1) < .05 else ' '}" for b in BUDGETS); a = c["avg"]; lo = c["avg_low"]
        print(f"[{c['family']:18s}] {key:44s} {s} | avg {100*a['diff']:+5.1f} [{100*a['ci95'][0]:+.1f},{100*a['ci95'][1]:+.1f}] pH={a.get('p_holm', float('nan')):.3f}"
              f" | <=384 {100*lo['diff']:+5.1f} [{100*lo['ci95'][0]:+.1f},{100*lo['ci95'][1]:+.1f}] | same {c['same_selection_share']:.2f} | " + " ".join(f"{d[:3]} {100*v['diff']:+.1f}" for d, v in c["by_dataset_avg"].items()))
    for name, f in res["fixed"].items():
        print(f"fixed {name:30s} cover {pct(f['mean'])} tok {f['tokens']:.0f} n_sel {f['n_selected']:.2f} vs full {100*f['vs_full']['diff']:+.1f} [{100*f['vs_full']['ci95'][0]:+.1f},{100*f['vs_full']['ci95'][1]:+.1f}] pH={f['vs_full']['p_holm_all_arms']:.3f} ratio {f['token_ratio_full_over_sel']:.2f}x",
              " ".join(f"{k} {100*v['diff']:+.1f} [{100*v['ci95'][0]:+.1f},{100*v['ci95'][1]:+.1f}]" for k, v in f.items() if k.startswith("vs_") and k != "vs_full"))
    print("full-context family:", res["full_context_family"]["size_all_arms"], "tests; significant after Holm:", {k: (round(100*v['diff'], 1), round(v['p_holm'], 4)) for k, v in res["full_context_family"]["significant"].items() if v["diff"] > 0} or "none positive",
          "| negative:", sum(v["diff"] < 0 for v in res["full_context_family"]["significant"].values()))
    for sname, e in res.get("strata", {}).items():
        print(f"stratum {sname} n={e['n']} full {pct(e['full'])}: " + " | ".join(f"{k} {100*v['diff']:+.1f} [{100*v['ci95'][0]:+.1f},{100*v['ci95'][1]:+.1f}]" for k, v in e.items() if isinstance(v, dict) and "diff" in v))
    if "probe" in res: print("probe:", {k: round(v, 3) for k, v in res["probe"].items()})


if __name__ == "__main__":
    main(sys.argv[1])
