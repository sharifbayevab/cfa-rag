# -*- coding: utf-8 -*-
"""Bootstrap 95% confidence intervals and a significance check for the paper's
headline quantities, computed from per-question data in raw_*.jsonl. Emits
article{2,3}/statmacros.tex.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
RNG = np.random.default_rng(0)


def load_rows():
    """Load the 13 distractor-pool cells only (exclude open-corpus `oc_`,
    set-level, and test files), tagging each row with its cell for clustering."""
    rows = []
    for p in glob.glob(str(RES / "raw_*.jsonl")):
        name = Path(p).name
        if "test_" in name or "raw_oc_" in name:
            continue
        cell = name[len("raw_"):-len(".jsonl")]
        for line in Path(p).read_text().splitlines():
            if line.strip():
                r = json.loads(line); r["_cell"] = cell
                rows.append(r)
    return rows


def cluster_boot_delta(rows, key_a, key_b, B=5000):
    """Two-level (cell-clustered) paired bootstrap of mean(a-b): resample cells
    with replacement, then questions within each sampled cell."""
    by_cell = {}
    for r in rows:
        if "selection" in r:
            by_cell.setdefault(r["_cell"], []).append(
                r["selection"][key_a]["cover"] - r["selection"][key_b]["cover"])
    cells = [np.asarray(v, dtype=float) for v in by_cell.values() if v]
    if not cells:
        return (0.0, 0.0, 0.0)
    point = float(np.mean(np.concatenate(cells)))
    stats = []
    ncell = len(cells)
    for _ in range(B):
        ci = RNG.integers(0, ncell, size=ncell)
        vals = []
        for c in ci:
            arr = cells[c]
            vals.append(arr[RNG.integers(0, len(arr), size=len(arr))])
        stats.append(np.concatenate(vals).mean())
    return point, float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def boot_ci(vals, B=5000, f=np.mean):
    vals = np.asarray(vals, dtype=float)
    if len(vals) == 0:
        return (0.0, 0.0, 0.0)
    idx = RNG.integers(0, len(vals), size=(B, len(vals)))
    stats = f(vals[idx], axis=1)
    return float(f(vals)), float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def paired_boot_delta(a, b, B=5000):
    """95% CI of mean(a-b) by paired bootstrap (a,b aligned per question)."""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    if len(d) == 0:
        return (0.0, 0.0, 0.0)
    idx = RNG.integers(0, len(d), size=(B, len(d)))
    m = d[idx].mean(axis=1)
    return float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main():
    rows = load_rows()
    # parametric rate among correct (cover)
    correct = [r for r in rows if r.get("cover_full", r.get("judge_full", 0)) > 0.5
               or r["judge_full"] > 0.5]
    param = [1.0 if r["judge_cb"] > 0.5 else 0.0 for r in correct]
    pm, plo, phi = boot_ci(param)

    # relevance-causality Jaccard (per question, over all)
    def jac(r):
        a, b = set(r["toprel_idxs"]), set(r["causal_idxs"])
        return len(a & b) / len(a | b) if (a or b) else 1.0
    jm, jlo, jhi = boot_ci([jac(r) for r in rows])

    # causal_prune vs full: paired per-question cover, pooled across cells
    a = [r["selection"]["causal_prune"]["cover"] for r in rows if "selection" in r]
    b = [r["selection"]["full"]["cover"] for r in rows if "selection" in r]
    dm, dlo, dhi = paired_boot_delta(a, b)
    sig = "not significant (CI spans 0)" if (dlo <= 0 <= dhi) else "significant"
    # cell-clustered (hierarchical) bootstrap of the same delta
    cm, clo, chi = cluster_boot_delta(rows, "causal_prune", "full")
    csig = "not significant (CI spans 0)" if (clo <= 0 <= chi) else "significant"

    # token reduction causal_prune vs full
    tcp = np.mean([r["selection"]["causal_prune"]["n_tokens"] for r in rows if "selection" in r])
    tfull = np.mean([r["selection"]["full"]["n_tokens"] for r in rows if "selection" in r])

    out = {
        "n_questions": len(rows), "n_correct": len(correct),
        "parametric": [round(pm, 3), round(plo, 3), round(phi, 3)],
        "rel_caus_jaccard": [round(jm, 3), round(jlo, 3), round(jhi, 3)],
        "cfa_minus_full_cover": [round(dm, 3), round(dlo, 3), round(dhi, 3)], "cfa_vs_full": sig,
        "cfa_minus_full_clustered": [round(cm, 3), round(clo, 3), round(chi, 3)], "cfa_vs_full_clustered": csig,
        "token_cut": round(tfull / max(tcp, 1), 1),
    }
    print(json.dumps(out, indent=2))

    macros = [
        "% bootstrap 95% CIs (exp/stats_ci.py)",
        f"\\newcommand{{\\ciParam}}{{{pm*100:.0f}\\% (95\\% CI {plo*100:.0f}--{phi*100:.0f})}}",
        f"\\newcommand{{\\ciJacc}}{{{jm:.2f} (95\\% CI {jlo:.2f}--{jhi:.2f})}}",
        f"\\newcommand{{\\ciCFAdelta}}{{{dm:+.3f} (95\\% CI {dlo:+.3f} to {dhi:+.3f})}}",
        f"\\newcommand{{\\ciCFAsig}}{{{sig}}}",
        f"\\newcommand{{\\ciCFAclustered}}{{{cm:+.3f} (95\\% CI {clo:+.3f} to {chi:+.3f})}}",
        f"\\newcommand{{\\ciNq}}{{{len(rows)}}}",
    ]
    for o in [ROOT / "article2", ROOT / "article3"]:
        (o / "statmacros.tex").write_text("\n".join(macros) + "\n", encoding="utf-8")
    print("wrote statmacros.tex")


if __name__ == "__main__":
    main()
