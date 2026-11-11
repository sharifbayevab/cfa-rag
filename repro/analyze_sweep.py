"""Analysis of a token-budget sweep (E1/E2/E3 on Qwen3-4B, E4 on GPT-4.1-mini).

  python3 repro/analyze_sweep.py qwen      # repro/inputs_e1/e1_sweep.jsonl  -> outputs/e1_sweep_analysis.json
  python3 repro/analyze_sweep.py gemini    # outputs/e4_gemini35fl_answers.jsonl -> outputs/e4_gemini35fl_analysis.json
  python3 repro/analyze_sweep.py gpt       # outputs/e4_gpt41mini_answers.jsonl  -> outputs/e4_gpt41mini_analysis.json

Dataset-stratified paired question bootstrap (B=2000, seed 0); Holm correction
within each prespecified family of contrasts.  Metric: cover-EM.
"""
import json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from cfa_lib import BUDGETS, boot_indices, holm, jl, mean_ci, paired

FAMILIES = {
    "headline_vs_ce": [("cfa_exact", "ce"), ("cfa_transfer", "ce")],
    "vs_bge": [("cfa_exact", "bge"), ("cfa_exact_bge", "bge"), ("cfa_transfer_bge", "bge"), ("cfa_transfer", "bge")],
    "vs_contextcite": [("cfa_exact", "contextcite"), ("cfa_exact_bge", "contextcite"), ("contextcite", "ce"), ("contextcite", "bge")],
    "gold_internal": [("gold_cfa_exact", "gold_oracle"), ("gold_cfa_transfer", "gold_oracle")],
    "modern_reranker": [("bge", "ce"), ("oof", "ce"), ("dense", "ce")],
    "target_ablation": [("tgt_cfa", "tgt_gold"), ("tgt_signed", "tgt_gold"), ("tgt_cfa", "ce"), ("tgt_gold", "ce"), ("tgt_signed", "ce"),
                        ("tgt_ce", "ce"), ("signed_support", "ce")],
    "cheap_prober": [("cfa_transfer_local2", "ce"), ("cfa_transfer_gemma4b", "ce"), ("cfa_transfer_local2_bge", "bge"), ("cfa_transfer_gemma4b_bge", "bge")],
    "other": [("signed_oracle", "ce"), ("gold_oracle", "ce"), ("cfa_exact", "cfa_transfer"),
              ("contextcite_stable", "contextcite"), ("cfa_exact", "contextcite_stable"), ("cfa_exact", "contextcite16_stable")],
}


def main(which):
    SRC = {"qwen": ("inputs_e1/e1_sweep.jsonl", "outputs/e1_sweep_analysis.json"),
           "gemini": ("outputs/e4_gemini35fl_answers.jsonl", "outputs/e4_gemini35fl_analysis.json"),
           "gpt": ("outputs/e4_gpt41mini_answers.jsonl", "outputs/e4_gpt41mini_analysis.json")}
    src = HERE / SRC[which][0]
    rows = sorted(jl(src), key=lambda r: r["question_group"])
    ds = np.array([r["dataset"] for r in rows]); bi = boot_indices(ds)
    arms = sorted({k.rsplit("_tok", 1)[0] for k in rows[0]["methods"] if "_tok" in k})
    def col(name, key="cover"): return np.array([r["methods"][name][key] for r in rows], float)
    res = {"source": src.name, "n": len(rows), "arms": arms, "budgets": BUDGETS, "table": {}, "contrasts": {}, "by_dataset": {}, "fixed": {}}
    full = col("full")
    res["full"] = {**mean_ci(full, bi), "tokens": float(col("full", "context_tokens").mean()),
                   "by_dataset": {d: float(full[ds == d].mean()) for d in sorted(set(ds))}}
    for arm in arms:
        avg = np.mean([col(f"{arm}_tok{b}") for b in BUDGETS], axis=0)
        res["table"][arm] = {"avg": mean_ci(avg, bi)}
        for b in BUDGETS:
            n = f"{arm}_tok{b}"
            res["table"][arm][str(b)] = {**mean_ci(col(n), bi), "f1": float(col(n, "f1").mean()), "tokens": float(col(n, "context_tokens").mean()),
                                         "gold_recall": float(col(n, "gold_recall").mean()), "vs_full": paired(col(n), full, bi)}
        res["by_dataset"][arm] = {d: {str(b): float(col(f"{arm}_tok{b}")[ds == d].mean()) for b in BUDGETS} |
                                  {"avg": float(avg[ds == d].mean())} for d in sorted(set(ds))}
    for fam, pairs in FAMILIES.items():
        pv = {}
        for a, b in pairs:
            if a not in arms or b not in arms: continue
            key = f"{a}-{b}"; res["contrasts"][key] = {"family": fam}
            for bud in BUDGETS:
                c = paired(col(f"{a}_tok{bud}"), col(f"{b}_tok{bud}"), bi); res["contrasts"][key][str(bud)] = c; pv[f"{key}@{bud}"] = c["p"]
            aa = np.mean([col(f"{a}_tok{x}") for x in BUDGETS], axis=0); bb = np.mean([col(f"{b}_tok{x}") for x in BUDGETS], axis=0)
            res["contrasts"][key]["avg"] = paired(aa, bb, bi)
            lo = [x for x in BUDGETS if x <= 384]
            res["contrasts"][key]["avg_low"] = paired(np.mean([col(f"{a}_tok{x}") for x in lo], axis=0), np.mean([col(f"{b}_tok{x}") for x in lo], axis=0), bi)
            res["contrasts"][key]["by_dataset_avg"] = {}
            for d in sorted(set(ds)):
                m = ds == d; bid = boot_indices(ds[m], seed=1)
                res["contrasts"][key]["by_dataset_avg"][d] = paired(aa[m], bb[m], bid)
        adj = holm(pv) if pv else {}
        for k, v in adj.items():
            key, bud = k.split("@"); res["contrasts"][key][bud]["p_holm"] = v
        # Holm over the budget-averaged contrasts of the family
        pa = {f"{a}-{b}": res["contrasts"][f"{a}-{b}"]["avg"]["p"] for a, b in pairs if f"{a}-{b}" in res["contrasts"]}
        for k, v in (holm(pa) if pa else {}).items(): res["contrasts"][k]["avg"]["p_holm"] = v
    for name in [k for k in rows[0]["methods"] if "_tok" not in k and k != "full"]:
        res["fixed"][name] = {**mean_ci(col(name), bi), "tokens": float(col(name, "context_tokens").mean()),
                              "n_selected": float(np.mean([len(r["methods"][name]["selected_idxs"]) for r in rows])),
                              "gold_recall": float(col(name, "gold_recall").mean()), "vs_full": paired(col(name), full, bi),
                              "token_ratio_full_over_sel": float(col("full", "context_tokens").sum() / col(name, "context_tokens").sum()),
                              "by_dataset": {d: float(col(name)[ds == d].mean()) for d in sorted(set(ds))}}
    if "cfa_prune" in res["fixed"]:
        for other in ("ce_match_prune", "bge_match_prune", "contextcite_match_prune"):
            if other in res["fixed"]: res["fixed"]["cfa_prune"][f"vs_{other}"] = paired(col("cfa_prune"), col(other), bi)
    out = HERE / SRC[which][1]
    out.write_text(json.dumps(res, indent=1) + "\n")
    pct = lambda x: f"{100*x:5.1f}"
    print(f"{src.name}: n={len(rows)} full {pct(res['full']['mean'])} tokens {res['full']['tokens']:.0f}")
    print(f"{'arm':22s}" + "".join(f"{b:>7d}" for b in BUDGETS) + "    avg")
    for arm in sorted(arms, key=lambda a: -res["table"][a]["avg"]["mean"]):
        print(f"{arm:22s}" + "".join(f"  {pct(res['table'][arm][str(b)]['mean'])}" for b in BUDGETS) + f"  {pct(res['table'][arm]['avg']['mean'])}")
    print("tokens ce:", [round(res["table"]["ce"][str(b)]["tokens"]) for b in BUDGETS], "gold recall ce/cfa_exact:",
          [(round(100*res["table"]["ce"][str(b)]["gold_recall"]), round(100*res["table"]["cfa_exact"][str(b)]["gold_recall"])) for b in BUDGETS])
    for key, c in res["contrasts"].items():
        s = " ".join(f"{100*c[str(b)]['diff']:+5.1f}{'*' if c[str(b)].get('p_holm', 1) < .05 else ' '}" for b in BUDGETS)
        a = c["avg"]; lo = c["avg_low"]
        print(f"[{c['family']:15s}] {key:34s} {s} | avg {100*a['diff']:+5.1f} [{100*a['ci95'][0]:+.1f},{100*a['ci95'][1]:+.1f}] pH={a.get('p_holm', float('nan')):.3f}"
              f" | <=384 {100*lo['diff']:+5.1f} [{100*lo['ci95'][0]:+.1f},{100*lo['ci95'][1]:+.1f}] | " +
              " ".join(f"{d[:3]} {100*v['diff']:+.1f}" for d, v in c["by_dataset_avg"].items()))
    for name, f in res["fixed"].items():
        print(f"fixed {name:26s} cover {pct(f['mean'])} tokens {f['tokens']:.0f} n_sel {f['n_selected']:.2f} gold_recall {pct(f['gold_recall'])} vs full {100*f['vs_full']['diff']:+.1f} [{100*f['vs_full']['ci95'][0]:+.1f},{100*f['vs_full']['ci95'][1]:+.1f}] ratio {f['token_ratio_full_over_sel']:.2f}x",
              " ".join(f"{k} {100*v['diff']:+.1f} [{100*v['ci95'][0]:+.1f},{100*v['ci95'][1]:+.1f}]" for k, v in f.items() if k.startswith("vs_") and k != "vs_full"))


if __name__ == "__main__":
    main(sys.argv[1])
