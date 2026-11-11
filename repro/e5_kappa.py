"""E5. Chance-corrected agreement (Cohen's kappa) between deletion and
position-preserving mask labels.  No generator calls.
Usage: python3 repro/e5_kappa.py [path.jsonl ...]   (default: the frozen n=120 control run;
the n=450 mask run (repro/e1_spark/probe_masks.py) is added automatically when present)
Question-clustered, dataset-stratified bootstrap, B=1000, seed 0.
"""
import json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent


def kappa(a, b):
    a = np.asarray(a, bool); b = np.asarray(b, bool)
    po = np.mean(a == b); pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return (po - pe) / (1 - pe) if pe < 1 else np.nan


def analyse(path):
    rows = [json.loads(x) for x in open(path)]
    rng = np.random.default_rng(0)
    out = {"file": str(path.name), "n_questions": len(rows), "n_passages": sum(len(r["delete_changes"]) for r in rows)}
    def stat(sample, gold_only=None):
        d, m = [], []
        for r in sample:
            for i, (x, y) in enumerate(zip(r["delete_changes"], r["mask_changes"])):
                if gold_only is None or (i in r["gold_idxs"]) == gold_only:
                    d.append(x); m.append(y)
        d, m = np.array(d), np.array(m)
        return {"kappa": kappa(d, m), "agreement": float(np.mean(d == m)), "delete_rate": float(d.mean()), "mask_rate": float(m.mean()),
                "pos_agreement": float(2 * np.sum(d & m) / max(1, d.sum() + m.sum()))}
    scopes = {"all": rows}
    for ds in sorted({r["dataset"] for r in rows}):
        scopes[ds] = [r for r in rows if r["dataset"] == ds]
    for name, sub in scopes.items():
        for tag, g in [("", None), ("|gold", True), ("|nongold", False)]:
            point = stat(sub, g); by = {}
            for r in sub: by.setdefault(r["dataset"], []).append(r)
            boot = []
            for _ in range(1000):
                samp = [v[i] for v in by.values() for i in rng.integers(0, len(v), len(v))]
                boot.append(stat(samp, g)["kappa"])
            point["kappa_ci95"] = [float(x) for x in np.nanpercentile(boot, [2.5, 97.5])]
            out[name + tag] = point
    # question-level: any passage flagged
    qa = [any(r["delete_changes"]) for r in rows]; qb = [any(r["mask_changes"]) for r in rows]
    out["question_any_flag_kappa"] = kappa(qa, qb)
    if "joint_gold_delete_change" in rows[0]:
        out["joint_gold_kappa"] = kappa([r["joint_gold_delete_change"] for r in rows], [r["joint_gold_mask_change"] for r in rows])
    return out


if __name__ == "__main__":
    paths = [Path(p) for p in sys.argv[1:]] or [HERE / "inputs/intervention_controls_qwen3_4b_n120.jsonl"]
    if not sys.argv[1:] and (HERE / "inputs_e1/e1_loo_masks.jsonl").exists():
        paths.append(HERE / "inputs_e1/e1_loo_masks.jsonl")
    res = {p.name: analyse(p) for p in paths}
    (HERE / "outputs/e5_kappa.json").write_text(json.dumps(res, indent=1) + "\n")
    for n, r in res.items():
        print(n, r["n_questions"], r["n_passages"])
        for k, v in r.items():
            if isinstance(v, dict):
                print(f"  {k:18s} kappa {v['kappa']:.3f} [{v['kappa_ci95'][0]:.3f}, {v['kappa_ci95'][1]:.3f}] agree {v['agreement']:.3f} del {v['delete_rate']:.3f} mask {v['mask_rate']:.3f} posagree {v['pos_agreement']:.3f}")
        print("  question-any kappa %.3f; joint-gold kappa %.3f" % (r["question_any_flag_kappa"], r.get("joint_gold_kappa", float('nan'))))
