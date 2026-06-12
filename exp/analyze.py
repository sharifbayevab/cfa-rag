# -*- coding: utf-8 -*-
"""Aggregate all experiment cells into paper tables + figures.

Reads results/summary_*.json (one per dataset x generator cell) and
results/raw_*.jsonl, and produces:
  * article2/figures/*.pdf|png   -- figures for the paper
  * results/tables.tex           -- LaTeX table fragments
  * console summary
"""
from __future__ import annotations

import glob
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
FIG = ROOT / "article3" / "images"
FIG.mkdir(parents=True, exist_ok=True)

GEN_LABEL = {
    "openai-gpt-4.1-mini": "GPT-4.1-mini", "openai-gpt-4.1-nano": "GPT-4.1-nano",
    "openai-gpt-4.1": "GPT-4.1",
    "ollama-gemma3-4b": "Gemma3-4B", "ollama-gemma3-12b": "Gemma3-12B",
    "ollama-gemma3-27b": "Gemma3-27B", "ollama-gpt-oss-20b": "gpt-oss-20B",
}
DS_LABEL = {"hotpotqa": "HotpotQA", "2wiki": "2Wiki", "musique": "MuSiQue"}
# rough capability ordering for the gradient axis
CAP_ORDER = ["ollama-gemma3-4b", "openai-gpt-4.1-nano", "ollama-gemma3-12b",
             "ollama-gemma3-27b", "openai-gpt-4.1-mini", "openai-gpt-4.1",
             "ollama-gpt-oss-20b"]


def _parse_stem(path):
    m = re.match(r"summary_(?:(test|abl[^_]*)_)?(hotpotqa|2wiki|musique)_(.+)_n(\d+)\.json",
                 Path(path).name)
    if not m:
        return None
    tag, ds, gen, n = m.groups()
    return {"tag": tag or "", "dataset": ds, "gen": gen, "n": int(n)}


def load_cells(exclude_tags=("test",)):
    cells = []
    for p in sorted(glob.glob(str(RES / "summary_*.json"))):
        meta = _parse_stem(p)
        if not meta or meta["tag"] in exclude_tags:
            continue
        meta["summary"] = json.loads(Path(p).read_text())
        meta["path"] = p
        cells.append(meta)
    return cells


def table_faithfulness(cells):
    rows = []
    for c in cells:
        s = c["summary"]; fg = s["faithfulness_gap"]
        rows.append((GEN_LABEL.get(c["gen"], c["gen"]), DS_LABEL.get(c["dataset"], c["dataset"]),
                     s["n"], s["acc_full_judge"], s["acc_closedbook_judge"],
                     fg["frac_correct_parametric"], fg["frac_correct_no_causal_fragment"]))
    rows.sort()
    print("\n=== FAITHFULNESS GAP ===")
    print(f"{'Model':14}{'Data':10}{'n':>4}{'AccFull':>9}{'AccCB':>7}{'%Param':>8}{'%NoCausal':>10}")
    for r in rows:
        print(f"{r[0]:14}{r[1]:10}{r[2]:>4}{r[3]:>9}{r[4]:>7}{r[5]:>8}{r[6]:>10}")
    return rows


def table_relcausal(cells):
    print("\n=== RELEVANCE != CAUSALITY  &  CAUSAL vs GOLD ===")
    print(f"{'Model':14}{'Data':10}{'relCausJacc':>12}{'Spearman':>10}{'cgPrec':>8}{'cgRec':>7}")
    for c in sorted(cells, key=lambda x: (x["gen"], x["dataset"])):
        s = c["summary"]; rc = s["relevance_vs_causality"]; cg = s["causal_vs_gold"]
        print(f"{GEN_LABEL.get(c['gen'],c['gen']):14}{DS_LABEL.get(c['dataset'],c['dataset']):10}"
              f"{rc['top_rel_causal_jaccard']:>12}{str(rc['rel_causal_spearman']):>10}"
              f"{cg['precision']:>8}{cg['recall']:>7}")


def table_selection(cells):
    print("\n=== SELECTION (avg over cells) ===")
    strats = ["full", "topk_relevance", "random_k", "causal_prune", "oracle_gold"]
    agg = {st: {"judge": [], "tokens": [], "gold_recall": [], "frags": []} for st in strats}
    for c in cells:
        for st, v in c["summary"]["selection"].items():
            agg[st]["judge"].append(v["judge_acc"]); agg[st]["tokens"].append(v["avg_tokens"])
            agg[st]["gold_recall"].append(v["gold_recall"]); agg[st]["frags"].append(v["avg_frags"])
    print(f"{'Strategy':16}{'Acc':>7}{'Tokens':>9}{'Frags':>7}{'GoldRec':>9}")
    for st in strats:
        a = agg[st]
        print(f"{st:16}{np.mean(a['judge']):>7.3f}{np.mean(a['tokens']):>9.1f}"
              f"{np.mean(a['frags']):>7.2f}{np.mean(a['gold_recall']):>9.3f}")
    return agg


def fig_capability_gap(cells):
    """Parametric leakage vs model capability, one point per generator (mean
    over datasets) for a clean trend, plus faint per-cell points."""
    pts = {}
    for c in cells:
        s = c["summary"]
        pts.setdefault(c["gen"], []).append(
            (s["acc_full_judge"], s["faithfulness_gap"]["frac_correct_parametric"]))
    fig, ax = plt.subplots(figsize=(6, 4.2))
    order = [g for g in CAP_ORDER if g in pts]
    xs_mean, ys_mean = [], []
    for gen in order:
        vs = pts[gen]
        ax.scatter([v[0] for v in vs], [v[1] for v in vs], s=22, alpha=0.25,
                   color="gray")
        mx, my = np.mean([v[0] for v in vs]), np.mean([v[1] for v in vs])
        xs_mean.append(mx); ys_mean.append(my)
        ax.scatter([mx], [my], s=110, label=GEN_LABEL.get(gen, gen), zorder=3)
        ax.annotate(GEN_LABEL.get(gen, gen), (mx, my), fontsize=7,
                    xytext=(4, 4), textcoords="offset points")
    # (no fitted trend line: with 5 points and a family confound a regression
    #  would be non-inferential; we show the points only)
    ax.set_xlabel("Answer accuracy with full context (model capability)")
    ax.set_ylabel("Parametric fraction of correct answers\n(correct without any context)")
    ax.set_title("Parametric reliance vs. capability: large but no clean trend")
    ax.grid(alpha=0.3); ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout(); fig.savefig(FIG / "capability_gap.pdf"); fig.savefig(FIG / "capability_gap.png", dpi=160)
    print("saved capability_gap")


def fig_rel_vs_causal(raw_glob="raw_*hotpotqa*gpt-4.1-mini*"):
    files = glob.glob(str(RES / (raw_glob + ".jsonl")))
    if not files:
        return
    rel, caus = [], []
    for line in Path(files[0]).read_text().splitlines():
        r = json.loads(line)
        rel += r["rel_scores"]; caus += r["causal_scores"]
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    ax.scatter(rel, caus, s=8, alpha=0.25)
    ax.set_xlabel("Relevance score (dense similarity to query)")
    ax.set_ylabel("Causal contribution (counterfactual)")
    ax.set_title("Relevance does not predict causal contribution")
    if len(rel) > 2:
        rho = np.corrcoef(rel, caus)[0, 1]
        ax.text(0.05, 0.92, f"Pearson r = {rho:.2f}", transform=ax.transAxes)
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(FIG / "rel_vs_causal.pdf"); fig.savefig(FIG / "rel_vs_causal.png", dpi=160)
    print("saved rel_vs_causal")


def fig_selection_pareto(cells=None):
    """Accuracy-vs-cost Pareto built from the SAME source as tab_methods
    (methods_aggregate): one colored point per method, y = strong-generator
    accuracy (Acc_s column), x = tokens (Tok column). Every point is a table cell."""
    agg, nfiles = methods_aggregate()
    if not nfiles:
        print("no methods_*.json yet; skip pareto"); return
    palette = {"full": "#1f77b4", "random_k": "#8c564b", "topk_relevance": "#ff7f0e",
               "contextcite": "#9467bd", "amortized_causal": "#2ca02c",
               "cfa_prune": "#d62728", "oracle_gold": "#17becf"}
    highlight = {"cfa_prune", "amortized_causal"}
    # labels placed AWAY from points: (dx, dy, ha) — most to the left (ha=right)
    off = {"full": (-10, 6, "right"), "random_k": (-10, -2, "right"),
           "topk_relevance": (0, -14, "center"), "contextcite": (10, 7, "left"),
           "amortized_causal": (0, 12, "center"), "cfa_prune": (0, 13, "center"),
           "oracle_gold": (-10, 8, "right")}
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for m in METHODS_ORDER:
        a = agg[m]
        if not a["acc_strong"] or not a["tokens"]:
            continue
        x = float(np.mean(a["tokens"])); y = float(np.mean(a["acc_strong"]))
        ax.scatter([x], [y], s=150 if m in highlight else 95, color=palette[m],
                   edgecolor="black" if m in highlight else "white",
                   linewidth=1.4 if m in highlight else 0.8, zorder=3)
        dx, dy, ha = off[m]
        ax.annotate(METHODS_SHORT[m], (x, y), fontsize=8.5, xytext=(dx, dy),
                    textcoords="offset points", ha=ha,
                    fontweight="bold" if m in highlight else "normal")
    ax.set_xlabel("Avg. context tokens (cost)"); ax.set_ylabel("Answer accuracy")
    ax.set_title("Accuracy vs context cost across selection methods")
    ax.set_xlim(20, 820); ax.set_ylim(0.30, 0.83)
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(FIG / "selection_pareto.pdf"); fig.savefig(FIG / "selection_pareto.png", dpi=160)
    print(f"saved selection_pareto (strong-gen, from {nfiles} method cells; matches tab_methods Acc_s/Tok)")


def _per_model(cells):
    """Aggregate cells per generator (mean over datasets), capability-ordered."""
    by = {}
    for c in cells:
        by.setdefault(c["gen"], []).append(c["summary"])
    rows = []
    for gen in [g for g in CAP_ORDER if g in by] + [g for g in by if g not in CAP_ORDER]:
        ss = by[gen]
        def m(f):
            return float(np.mean([f(s) for s in ss]))
        rows.append({
            "gen": gen, "label": GEN_LABEL.get(gen, gen), "k": len(ss),
            "acc": m(lambda s: s["acc_full_judge"]),
            "acc_cb": m(lambda s: s["acc_closedbook_judge"]),
            "param": m(lambda s: s["faithfulness_gap"]["frac_correct_parametric"]),
            "nocausal": m(lambda s: s["faithfulness_gap"]["frac_correct_no_causal_fragment"]),
            "jacc": m(lambda s: s["relevance_vs_causality"]["top_rel_causal_jaccard"]),
            "rho": m(lambda s: s["relevance_vs_causality"].get("rel_causal_spearman") or 0.0),
            "cgp": m(lambda s: s["causal_vs_gold"]["precision"]),
            "cgr": m(lambda s: s["causal_vs_gold"]["recall"]),
        })
    return rows


def emit_latex(cells):
    """Write results macros + compact (per-model) table fragments to both
    article2/ and article3/ (IEEE Access)."""
    outs = [ROOT / "article2", ROOT / "article3"]
    for o in outs:
        o.mkdir(exist_ok=True)
    pm = _per_model(cells)

    # --- compact faithfulness table (per model, mean over datasets) ---
    fa = [r"\begin{tabular}{lccccc}", r"\toprule",
          r"Model & Acc & Acc$_{\text{cb}}$ & \%Par & \%NoC & cgP\\", r"\midrule"]
    for r in pm:
        fa.append(f"{r['label']} & {r['acc']:.2f} & {r['acc_cb']:.2f} & "
                  f"{r['param']:.2f} & {r['nocausal']:.2f} & {r['cgp']:.2f}\\\\")
    fa += [r"\bottomrule", r"\end{tabular}"]
    fa_tex = "\n".join(fa)

    # --- compact relevance-vs-causality table (per model) ---
    rc = [r"\begin{tabular}{lccc}", r"\toprule",
          r"Model & relCausJ & Spearman & cgPrec\\", r"\midrule"]
    for r in pm:
        rc.append(f"{r['label']} & {r['jacc']:.2f} & {r['rho']:.2f} & {r['cgp']:.2f}\\\\")
    rc += [r"\bottomrule", r"\end{tabular}"]
    rc_tex = "\n".join(rc)
    # --- detailed per-cell table (for a full-width table*) ---
    cells_s = sorted(cells, key=lambda c: (CAP_ORDER.index(c["gen"]) if c["gen"] in CAP_ORDER else 9,
                                           c["dataset"]))
    fl = [r"\begin{tabular}{llcccccc}", r"\toprule",
          r"Model & Data & $n$ & Acc & Acc$_{\text{cb}}$ & \%Par & relCausJ & cgPrec\\", r"\midrule"]
    for c in cells_s:
        s = c["summary"]; fg = s["faithfulness_gap"]; rcc = s["relevance_vs_causality"]; cg = s["causal_vs_gold"]
        fl.append(f"{GEN_LABEL.get(c['gen'],c['gen'])} & {DS_LABEL.get(c['dataset'],c['dataset'])} & "
                  f"{s['n']} & {s['acc_full_judge']:.2f} & {s['acc_closedbook_judge']:.2f} & "
                  f"{fg['frac_correct_parametric']:.2f} & {rcc['top_rel_causal_jaccard']:.2f} & "
                  f"{cg['precision']:.2f}\\\\")
    fl += [r"\bottomrule", r"\end{tabular}"]
    fl_tex = "\n".join(fl)
    for o in outs:
        (o / "tab_faithfulness.tex").write_text(fa_tex, encoding="utf-8")
        (o / "tab_relcausal.tex").write_text(rc_tex, encoding="utf-8")
        (o / "tab_faithfulness_full.tex").write_text(fl_tex, encoding="utf-8")

    # --- selection table (avg over cells) ---
    strats = ["full", "topk_relevance", "random_k", "causal_prune", "oracle_gold"]
    names = {"full": "Full context", "topk_relevance": "Top-$k$ relevance",
             "random_k": "Random-$k$", "causal_prune": "Causal pruning (ours)",
             "oracle_gold": "Oracle gold"}
    agg = {st: {"judge": [], "tokens": [], "gold_recall": [], "frags": []} for st in strats}
    for c in cells:
        for st, v in c["summary"]["selection"].items():
            agg[st]["judge"].append(v["judge_acc"]); agg[st]["tokens"].append(v["avg_tokens"])
            agg[st]["gold_recall"].append(v["gold_recall"]); agg[st]["frags"].append(v["avg_frags"])
    sl = [r"\begin{tabular}{lcccc}", r"\toprule",
          r"Strategy & Acc & Frags & Tokens & GoldRec\\", r"\midrule"]
    for st in strats:
        a = agg[st]
        bold = (st == "causal_prune")
        nm = (r"\textbf{"+names[st]+"}") if bold else names[st]
        sl.append(f"{nm} & {np.mean(a['judge']):.3f} & {np.mean(a['frags']):.2f} & "
                  f"{np.mean(a['tokens']):.0f} & {np.mean(a['gold_recall']):.2f}\\\\")
    sl += [r"\bottomrule", r"\end{tabular}"]
    sl_tex = "\n".join(sl)
    for o in outs:
        (o / "tab_selection.tex").write_text(sl_tex, encoding="utf-8")

    # --- headline macros ---
    def avg(key_fn):
        return float(np.mean([key_fn(c["summary"]) for c in cells]))
    full_acc = avg(lambda s: s["selection"]["full"]["judge_acc"])
    cp = {k: avg(lambda s: s["selection"]["causal_prune"][k]) for k in ["judge_acc","avg_tokens"]}
    fc = {k: avg(lambda s: s["selection"]["full"][k]) for k in ["judge_acc","avg_tokens"]}
    tk = avg(lambda s: s["selection"]["topk_relevance"]["judge_acc"])
    param = avg(lambda s: s["faithfulness_gap"]["frac_correct_parametric"])
    jacc = avg(lambda s: s["relevance_vs_causality"]["top_rel_causal_jaccard"])
    macros = [
        "% auto-generated by exp/analyze.py",
        f"\\newcommand{{\\resParametric}}{{{param*100:.0f}}}",
        f"\\newcommand{{\\resRelCausJacc}}{{{jacc:.2f}}}",
        f"\\newcommand{{\\resCPacc}}{{{cp['judge_acc']:.3f}}}",
        f"\\newcommand{{\\resFullacc}}{{{fc['judge_acc']:.3f}}}",
        f"\\newcommand{{\\resTopkAcc}}{{{tk:.3f}}}",
        f"\\newcommand{{\\resCPtokens}}{{{cp['avg_tokens']:.0f}}}",
        f"\\newcommand{{\\resFulltokens}}{{{fc['avg_tokens']:.0f}}}",
        f"\\newcommand{{\\resTokenCut}}{{{fc['avg_tokens']/max(cp['avg_tokens'],1):.1f}}}",
        f"\\newcommand{{\\resCPretain}}{{{cp['judge_acc']/max(fc['judge_acc'],1e-9)*100:.0f}}}",
        f"\\newcommand{{\\resNcells}}{{{len(cells)}}}",
    ]
    macros_tex = "\n".join(macros) + "\n"
    for o in outs:
        (o / "resultmacros.tex").write_text(macros_tex, encoding="utf-8")
    print("emitted LaTeX: tab_faithfulness, tab_relcausal, tab_selection, resultmacros (article2+article3)")


METHODS_ORDER = ["full", "random_k", "topk_relevance", "contextcite",
                 "amortized_causal", "cfa_prune", "oracle_gold"]
METHODS_NAMES = {"full": "Full context", "random_k": "Random-$k$",
                 "topk_relevance": "Top-$k$ relevance",
                 "contextcite": "ContextCite~\\cite{cohenwang2024contextcite}",
                 "amortized_causal": r"\textbf{Amortized causal (ours)}",
                 "cfa_prune": r"\textbf{CFA pruning (ours)}", "oracle_gold": "Oracle gold"}
# short labels for the figure (no LaTeX)
METHODS_SHORT = {"full": "Full", "random_k": "Random-k", "topk_relevance": "Top-k rel.",
                 "contextcite": "ContextCite", "amortized_causal": "Amortized (ours)",
                 "cfa_prune": "CFA pruning (ours)", "oracle_gold": "Oracle gold"}


def methods_aggregate():
    """SINGLE source of truth for the methods comparison (tab_methods + the
    selection-Pareto figure). Reads methods_*.json, splits accuracy by weak vs
    strong generator, pools tokens/probe/gold-recall across the same cells.
    Returns {method: {acc_weak, acc_strong, tokens, probe, gr}} and the #cells."""
    files = [p for p in glob.glob(str(RES / "methods_*.json"))]

    def grp(gen):
        return "strong" if ("gpt-4.1-mini" in gen or "gpt-4.1_" in gen or gen.endswith("gpt-4.1")
                            or "gemma3-12b" in gen or "gemma3-27b" in gen) else "weak"
    agg = {m: {"acc_weak": [], "acc_strong": [], "tokens": [], "probe": [], "gr": []}
           for m in METHODS_ORDER}
    for p in files:
        d = json.loads(Path(p).read_text())
        g = grp(d["generator"].replace(":", "-"))
        for m, v in d["methods"].items():
            if m in agg:
                agg[m][f"acc_{g}"].append(v["acc"])
                agg[m]["tokens"].append(v["tokens"]); agg[m]["probe"].append(v["probe_calls"])
                agg[m]["gr"].append(v["gold_recall"])
    return agg, len(files)


def emit_methods_table():
    """Aggregate methods_*.json into a single tab_methods.tex (accuracy + cost),
    with the Tok column broken out per dataset (HotpotQA / 2Wiki / MuSiQue) so the
    headline comparison and the per-dataset token robustness live in one table.
    Acc/Probe/GR are pooled across cells; the pooled Tok the Pareto figure plots
    is the mean of the per-dataset Tok columns. Uses methods_aggregate() so the
    table and the Pareto figure stay consistent."""
    outs = [ROOT / "article2", ROOT / "article3"]
    agg, nfiles = methods_aggregate()
    by, ds_order = methods_perdataset()
    if not nfiles:
        print("no methods_*.json yet"); return
    present = [d for d in ds_order if any(by[d][m]["tokens"] for m in METHODS_ORDER)]
    tokhead = " & ".join(r"\shortstack{Tok\\(" + DS_LABEL[d] + r")}" for d in present)
    lines = [r"\begin{tabular}{lcc" + "c" * len(present) + "cc}", r"\toprule",
             r"Method & Acc$_{\text{w}}$ & Acc$_{\text{s}}$ & " + tokhead + r" & Probe & GR\\",
             r"\midrule"]
    for m in METHODS_ORDER:
        a = agg[m]
        if not (a["acc_weak"] or a["acc_strong"]):
            continue
        aw = f"{np.mean(a['acc_weak']):.2f}" if a["acc_weak"] else "--"
        as_ = f"{np.mean(a['acc_strong']):.2f}" if a["acc_strong"] else "--"
        toks = " & ".join(f"{np.mean(by[d][m]['tokens']):.0f}" if by[d][m]["tokens"] else "--"
                          for d in present)
        lines.append(f"{METHODS_NAMES[m]} & {aw} & {as_} & {toks} & "
                     f"{np.mean(a['probe']):.0f} & {np.mean(a['gr']):.2f}\\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    tex = "\n".join(lines)
    for o in outs:
        (o / "tab_methods.tex").write_text(tex, encoding="utf-8")
    print(f"emitted merged tab_methods.tex (acc + per-dataset Tok) from {nfiles} method cells")


def methods_perdataset():
    """Per-dataset view of the methods comparison: pool weak+strong cells within
    each dataset. Used to show the cost conclusion (Tok/Probe) holds on every
    dataset, not just on the pooled average."""
    ds_order = ["hotpotqa", "2wiki", "musique"]
    by = {d: {m: {"tokens": [], "probe": [], "gr": [], "acc": []} for m in METHODS_ORDER}
          for d in ds_order}
    for p in glob.glob(str(RES / "methods_*.json")):
        d = json.loads(Path(p).read_text())
        ds = d["dataset"]
        if ds not in by:
            continue
        for m, v in d["methods"].items():
            if m in by[ds]:
                by[ds][m]["tokens"].append(v["tokens"]); by[ds][m]["probe"].append(v["probe_calls"])
                by[ds][m]["gr"].append(v["gold_recall"]); by[ds][m]["acc"].append(v["acc"])
    return by, ds_order


def emit_methods_perdataset_table():
    """tab_methods_perdataset.tex: context Tokens per dataset + shared Probe,
    showing the token-cost ordering is stable across HotpotQA / 2Wiki / MuSiQue."""
    by, ds_order = methods_perdataset()
    present = [d for d in ds_order if any(by[d][m]["tokens"] for m in METHODS_ORDER)]
    if not present:
        print("no methods_*.json for per-dataset table"); return
    head = " & ".join([r"Tok (" + DS_LABEL[d] + ")" for d in present])
    lines = [r"\begin{tabular}{l" + "c" * len(present) + "c}", r"\toprule",
             r"Method & " + head + r" & Probe\\", r"\midrule"]
    for m in METHODS_ORDER:
        if not any(by[d][m]["tokens"] for d in present):
            continue
        toks = " & ".join(f"{np.mean(by[d][m]['tokens']):.0f}" if by[d][m]["tokens"] else "--"
                          for d in present)
        probe_vals = [v for d in present for v in by[d][m]["probe"]]
        pr = f"{np.mean(probe_vals):.0f}" if probe_vals else "--"
        lines.append(f"{METHODS_NAMES[m]} & {toks} & {pr}\\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    tex = "\n".join(lines)
    (ROOT / "article3" / "tab_methods_perdataset.tex").write_text(tex, encoding="utf-8")
    print(f"emitted tab_methods_perdataset.tex ({len(present)} datasets)")


def fig_feature_importance():
    f = RES / "selector_metrics.json"
    if not f.exists():
        return
    imp = json.loads(f.read_text()).get("feature_importance", {})
    if not imp:
        return
    items = sorted(imp.items(), key=lambda kv: kv[1])
    names = [k for k, _ in items]; vals = [v for _, v in items]
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    ax.barh(names, vals, color="#16a34a")
    ax.set_xlabel("Gradient-boosting feature importance")
    ax.set_title("What predicts causal contribution")
    fig.tight_layout(); fig.savefig(FIG / "feat_importance.pdf"); fig.savefig(FIG / "feat_importance.png", dpi=160)
    print("saved feat_importance")


def fig_perdataset(cells):
    """Grouped bars: parametric fraction by dataset, averaged over models."""
    ds_order = ["hotpotqa", "2wiki", "musique"]
    by = {d: [] for d in ds_order}
    for c in cells:
        if c["dataset"] in by:
            by[c["dataset"]].append(c["summary"]["faithfulness_gap"]["frac_correct_parametric"])
    labels = [DS_LABEL[d] for d in ds_order if by[d]]
    means = [np.mean(by[d]) for d in ds_order if by[d]]
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    ax.bar(labels, means, color=["#1d4ed8", "#0891b2", "#ea580c"])
    ax.set_ylabel("Mean parametric fraction\n(correct answers)")
    ax.set_title("Faithfulness gap shrinks on harder multi-hop tasks")
    for i, v in enumerate(means):
        ax.text(i, v + 0.01, f"{v:.2f}", ha="center", fontsize=9)
    ax.set_ylim(0, max(means) * 1.25)
    fig.tight_layout(); fig.savefig(FIG / "perdataset_param.pdf"); fig.savefig(FIG / "perdataset_param.png", dpi=160)
    print("saved perdataset_param")


def main():
    emit_methods_table()
    cells = load_cells()
    print(f"loaded {len(cells)} cells:", [(c['dataset'], c['gen'], c['n']) for c in cells])
    if not cells:
        print("no cells yet"); return
    table_faithfulness(cells)
    table_relcausal(cells)
    table_selection(cells)
    fig_capability_gap(cells)
    fig_rel_vs_causal()
    fig_selection_pareto(cells)
    fig_feature_importance()
    fig_perdataset(cells)
    emit_latex(cells)


if __name__ == "__main__":
    main()
