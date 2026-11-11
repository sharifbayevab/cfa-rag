"""E8. Probe cost against accuracy for every label-free arm (offline).

Calls per question are LLM calls made *before* the final answer (the final
answer call is one more for every arm).  m is the pool size (mean 13.32), D the
number of donor readers (5 for Qwen3-4B and Gemini, 4 for GPT-4.1-mini).
target calls: calls to the reader that answers; donor calls: calls to other models.
Accuracy: cover-EM averaged over the six budgets.  A point is Pareto-optimal if no
other arm has fewer-or-equal total calls and higher accuracy.
Qwen: arms of the round-2 run come from e6_qwen(_donors)_analysis.json; arms that
were only generated in round 1 (same stack, separate run) are flagged run="r1".
"""
import json, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from cfa_lib import load_pools
OUT = HERE / "outputs"
M1 = float(np.mean([len(p["fragments"]) + 1 for p in load_pools().values()]))


def calls(arm, D):
    """(target calls, donor calls) before the final answer."""
    T = {"ce": (0, 0), "bge": (0, 0), "dense": (0, 0), "oof": (0, 0),
         "anscont_bge": (1, 0), "ansoverlap_bge": (1, 0), "anscont_gemma4b_bge": (0, 1), "anscont_donors_bge": (0, D),
         "cfa_exact": (M1, 0), "cfa_exact_bge": (M1, 0), "anscont_exact_bge": (M1, 0), "sum_anscont_exact_bge": (M1, 0), "contextcite": (33, 0),
         "cfa_transfer": (0, D * M1), "cfa_transfer_bge": (0, D * M1), "sum_donorcont_transfer_bge": (0, D * M1),
         "anscont_transfer_bge": (1, D * M1), "sum_anscont_transfer_bge": (1, D * M1), "transfer_anscont_bge": (1, D * M1),
         "cfa_transfer_gemma4b": (0, M1), "cfa_transfer_gemma4b_bge": (0, M1), "cfa_transfer_local2": (0, 2 * M1), "cfa_transfer_local2_bge": (0, 2 * M1)}
    return T.get(arm)


def main():
    res = {"mean_pool_size_plus_1": M1, "readers": {}}
    spec = {"Qwen3-4B (HF)": (5, [("e6_qwen_analysis.json", "r2"), ("e6_qwen_donors_analysis.json", "r2-donors"), ("e1_sweep_analysis.json", "r1")]),
            "GPT-4.1-mini": (4, [("e6_gpt_analysis.json", "r2"), ("e4_gpt41mini_analysis.json", "r1")]),
            "Gemini-3.5-flash-lite": (5, [("e6_gemini_analysis.json", "r2"), ("e4_gemini35fl_analysis.json", "r1")])}
    for reader, (D, files) in spec.items():
        rows = {}; full = None
        for fn, run in files:
            if not (OUT / fn).exists(): continue
            a = json.load(open(OUT / fn)); full = full or a["full"]
            for arm, t in a["table"].items():
                c = calls(arm, D)
                if c is None or arm in rows: continue
                rows[arm] = {"run": run, "target_calls": c[0], "donor_calls": c[1], "total_calls": c[0] + c[1], "avg_cover": t["avg"]["mean"], "avg_ci95": t["avg"]["ci95"],
                             "cover_by_budget": {b: t[b]["mean"] for b in ("128", "256", "384", "512", "768", "1024")}}
        for arm, r in rows.items():
            r["pareto"] = not any((o["total_calls"] <= r["total_calls"] and o["avg_cover"] > r["avg_cover"] + 1e-12) or
                                  (o["total_calls"] < r["total_calls"] and o["avg_cover"] >= r["avg_cover"] - 1e-12) for k, o in rows.items() if k != arm)
        fixed = {}
        for fn, run in files[:1]:
            if (OUT / fn).exists():
                a = json.load(open(OUT / fn))
                for name, c in (("cfa_prune", (M1, 0)), ("anscont_prune", (1, 0))):
                    if name in a["fixed"]:
                        f = a["fixed"][name]; fixed[name] = {"target_calls": c[0], "donor_calls": 0, "cover": f["mean"], "tokens": f["tokens"], "ratio": f["token_ratio_full_over_sel"], "vs_full": f["vs_full"]}
        res["readers"][reader] = {"donors": D, "full": full, "arms": dict(sorted(rows.items(), key=lambda kv: (kv[1]["total_calls"], -kv[1]["avg_cover"]))), "fixed": fixed}
        print(f"\n{reader} (D={D}); full {100*full['mean']:.1f}")
        for arm, r in res["readers"][reader]["arms"].items():
            print(f"  {arm:28s} target {r['target_calls']:5.1f} donor {r['donor_calls']:5.1f} total {r['total_calls']:5.1f}  avg {100*r['avg_cover']:.1f} [{100*r['avg_ci95'][0]:.1f},{100*r['avg_ci95'][1]:.1f}] {'PARETO' if r['pareto'] else ''} ({r['run']})")
    (OUT / "e8_cost.json").write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    main()
