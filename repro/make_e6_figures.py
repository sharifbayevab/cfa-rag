"""Figures and LaTeX table bodies for round 2 (E6-E9).  New files only:
images/e6_within_gold_auc.{pdf,png}, images/e6_budget_curves.{pdf,png}, images/e6_pareto.{pdf,png},
tab_e6_within_gold.tex, tab_e6_incremental.tex, tab_e6_bridge.tex, tab_e6_selection.tex,
tab_e6_exact_fresh.tex, tab_e6_cost.tex, tab_e6_family_transfer.tex, tab_e6_full_family.tex
Table files hold rows only, with a comment line naming the columns (same convention as round 1).
"""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from e6_lib import NICE

HERE = Path(__file__).resolve().parent; PAPER = HERE.parent; OUT = HERE / "outputs"
C = {"blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "yellow": "#eda100", "violet": "#6250d6", "ink": "#0b0b0b", "muted": "#52514e", "grid": "#e4e3df"}
plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": C["muted"],
                     "axes.labelcolor": C["ink"], "xtick.color": C["muted"], "ytick.color": C["muted"], "pdf.fonttype": 42})
BUD = [128, 256, 384, 512, 768, 1024]; DAG = "$^\\dagger$"; DDAG = "$^\\ddagger$"
RUNS = [("Qwen3-4B (HF)", "e6_qwen_analysis.json", "e6_qwen_donors_analysis.json"), ("GPT-4.1-mini", "e6_gpt_analysis.json", None), ("Gemini-3.5-flash-lite", "e6_gemini_analysis.json", None)]
L = lambda p: json.load(open(OUT / p)) if p and (OUT / p).exists() else None
ci = lambda x, s=1, d=3, sign=False: (f"{s*x.get('mean', x.get('diff')):{'+' if sign else ''}.{d}f} [{s*x['ci95'][0]:{'+' if sign else ''}.{d}f}, {s*x['ci95'][1]:{'+' if sign else ''}.{d}f}]")
pc = lambda x: ci(x, 100, 1, True)


def save(fig, name):
    for ext in ("pdf", "png"): fig.savefig(PAPER / "images" / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def write(name, header, rows): (PAPER / name).write_text("".join(f"% {h}\n" for h in header) + "".join(r + " \\\\\n" for r in rows))


def within_gold(o):
    e6 = o["e6a_within_gold"]; rows = []
    for r in NICE:
        e = e6[f"{r}|sup"]; d = e["transfer-anscont_own"]; s = e["sum_anscont_transfer-anscont_own"]
        rows.append(f"{NICE[r]} & {e['n_informative']} & {ci(e['transfer'])} & {ci(e['anscont_own'])} & {e['anscont_donors']['mean']:.3f} & {e['anscont_gemma4b']['mean']:.3f} & "
                    f"{ci(e['sum_anscont_transfer'])} & {e['bge']['mean']:.3f} & {ci(d, sign=True)}{'*' if d['p_holm'] < .05 else ''} & {ci(s, sign=True)}{'*' if s['p_holm'] < .05 else ''}")
    write("tab_e6_within_gold.tex", ["Within-question AUC among gold passages, target: supporting passage (repro/e6_offline.py); * Holm p<0.05 over the seven readers",
          "reader | n questions | transferred CFA [CI] | containment of own answer [CI] | containment of donors' answers | containment of Gemma-3-4B answer | containment + transferred CFA [CI] | bge | transfer - containment [CI] | (containment + transfer) - containment [CI]"], rows)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6), sharey=True)
    ser = [("transfer", "Transferred CFA (other readers' deletions)", C["blue"], "o"), ("anscont_own", "Containment of the reader's own answer", C["orange"], "s"),
           ("anscont_donors", "Containment of the donors' answers", C["yellow"], "v"), ("sum_anscont_transfer", "Containment + transferred CFA", C["violet"], "P"), ("bge", "bge-reranker-v2-m3", C["aqua"], "^")]
    for ax, t, title in zip(axes, ("sup", "flip"), ("Target: supporting passage", "Target: any answer change")):
        for y, r in enumerate(NICE):
            e = e6[f"{r}|{t}"]
            for j, (k, lab, col, mk) in enumerate(ser):
                yy = y + (j - 2) * 0.15
                ax.plot(e[k]["ci95"], [yy, yy], color=col, lw=1.1, solid_capstyle="round", zorder=2)
                ax.plot(e[k]["mean"], yy, mk, ms=4.5, color=col, mec="white", mew=0.6, zorder=3, label=lab if y == 0 else None)
        ax.axvline(0.5, color=C["muted"], lw=0.8, ls=(0, (3, 3))); ax.set_xlim(0.25, 1.0); ax.set_title(title, fontsize=8, color=C["ink"])
        ax.set_xlabel("Within-question AUC among gold passages"); ax.grid(axis="x", color=C["grid"], lw=0.6); ax.set_axisbelow(True)
        ax.set_yticks(range(len(NICE))); ax.set_yticklabels(list(NICE.values()))
    axes[0].invert_yaxis(); axes[0].legend(loc="upper center", bbox_to_anchor=(1.05, -0.22), ncol=3, frameon=False, fontsize=7)
    save(fig, "e6_within_gold_auc")


def incremental(o):
    rows = []
    for t, tn in (("sup", "supporting"), ("flip", "any change")):
        for r in NICE:
            e = o["e6b_models"][f"{r}|{t}"]; d = e["bge+anscont+transfer - bge+anscont"]; q = e["bge+anscont+transfer - bge+anscont+qmean"]
            st = lambda x: "*" if x.get("p_holm", 1) < .05 else ""
            rows.append(f"{tn} & {NICE[r]} & {e['bge+anscont']['ap_pooled']['mean']:.3f} & {e['bge+anscont+transfer']['ap_pooled']['mean']:.3f} & {ci(d['d_ap_pooled'], sign=True)}{st(d['d_ap_pooled'])} & "
                        f"{ci(q['d_ap_pooled'], sign=True)}{st(q['d_ap_pooled'])} & {ci(d['d_auc_pooled'], sign=True)}{st(d['d_auc_pooled'])} & {ci(d['d_auc_within_question'], sign=True)}{st(d['d_auc_within_question'])} & {ci(d['d_auc_within_gold'], sign=True)}{st(d['d_auc_within_gold'])}")
    write("tab_e6_incremental.tex", ["Question-held-out logistic models (five folds): [bge, containment] vs [bge, containment, transferred CFA] (repro/e6_offline.py); * Holm p<0.05 over the seven readers",
          "target | reader | AP [bge, containment] | AP [+ transferred CFA] | delta AP pooled [CI] | delta AP against the question-mean control [CI] | delta AUC pooled [CI] | delta within-question AUC [CI] | delta within-gold AUC [CI]"], rows)
    rows = []
    for r in NICE:
        e = o["e6b_disagreement"][r]; s = o["e6b_strata"][r]["sup_within_gold_containment_uninformative"]
        rows.append(f"{NICE[r]} & " + " & ".join(f"{100*e[d]['share_supporting_not_containing_own_answer']:.1f} ({e[d]['n_supporting_not_containing_own_answer']}/{e[d]['n_supporting']})" for d in ("hotpotqa", "2wiki", "musique", "all")) +
                    f" & {e['all']['recall_in_top_ngold']['bge']:.3f} & {e['all']['recall_in_top_ngold']['anscont_bge']:.3f} & {e['all']['recall_in_top_ngold']['transfer_bge']:.3f} & {s['transfer']['n']} & {ci(s['transfer'])} & {ci(s['bge'])}")
    write("tab_e6_bridge.tex", ["Supporting passages that do not contain the reader's answer string (repro/e6_offline.py)",
          "reader | share % (count) HotpotQA | 2Wiki | MuSiQue | all | recall of these passages in the top-|gold| ranks: bge | containment + bge | transferred CFA + bge | n questions where containment does not separate gold passages | within-gold AUC there: transferred CFA [CI] | bge [CI]"], rows)


def family_transfer(o):
    rows = []
    for r, e in o["e9_leave_family_out"].items():
        t = e["sup"]; d = t["other_families_only-bge"]
        rows.append(f"{NICE[r]} & {', '.join(NICE[x] for x in e['donors']['other_families_only'])} & {ci(t['all_other_readers'])} & {ci(t['other_families_only'])} & "
                    f"{ci(t['same_family_only']) if t['same_family_only']['mean'] is not None else '--'} & {ci(t['gemma_only_main_grid'])} & {ci(t['bge'])} & {ci(d, sign=True)}{'*' if d['p_holm'] < .05 else ''} & {ci(t['other_families_only-all_other_readers'], sign=True)} & {ci(e['flip']['other_families_only'])}")
    write("tab_e6_family_transfer.tex", ["Transfer with donors from other model families only, within-gold AUC, target: supporting (repro/e6_offline.py); * Holm p<0.05 over six readers",
          "target reader | other-family donors | all five donors [CI] | other families only [CI] | same family only [CI] | two Gemma-3 readers of the main grid [CI] | bge [CI] | other families - bge [CI] | other families - all donors [CI] | other families only, target any change [CI]"], rows)


ARMS = [("bge", "bge-reranker-v2-m3"), ("anscont_bge", "Answer containment (own answer) + bge"), ("anscont_donors_bge", "Answer containment (donors' answers) + bge"),
        ("anscont_gemma4b_bge", "Answer containment (Gemma-3-4B answer) + bge"), ("cfa_transfer_bge", "Transferred CFA + bge"), ("cfa_exact_bge", "Exact CFA + bge"),
        ("sum_anscont_transfer_bge", "Containment + transferred CFA + bge"), ("sum_anscont_exact_bge", "Containment + exact CFA + bge"), ("goldans_oracle_bge", "Gold-answer containment + bge (oracle)")]
CONTR = [("anscont_bge-bge", "Containment + bge $-$ bge"), ("cfa_transfer_bge-bge", "Transferred CFA + bge $-$ bge"), ("cfa_transfer_bge-anscont_bge", "Transferred CFA + bge $-$ containment + bge"),
         ("cfa_exact_bge-anscont_bge", "Exact CFA + bge $-$ containment + bge"), ("sum_anscont_transfer_bge-anscont_bge", "Containment + transferred CFA $-$ containment"),
         ("sum_anscont_exact_bge-anscont_bge", "Containment + exact CFA $-$ containment"), ("cfa_transfer_bge-anscont_donors_bge", "Transferred CFA + bge $-$ donors' containment + bge"),
         ("anscont_gemma4b_bge-anscont_bge", "Gemma-3-4B containment $-$ own containment")]


def selection():
    rows = []
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5), squeeze=False)
    ser = [("goldans_oracle_bge", "Gold-answer containment + bge (oracle)", C["muted"], "D", (0, (2, 2))), ("sum_anscont_transfer_bge", "Containment + transferred CFA + bge", C["violet"], "P", "-"),
           ("cfa_transfer_bge", "Transferred CFA + bge", C["blue"], "o", "-"), ("anscont_bge", "Answer containment + bge (1 call)", C["orange"], "s", "-"),
           ("cfa_exact_bge", "Exact CFA + bge", C["yellow"], "v", "-"), ("bge", "bge-reranker-v2-m3", C["aqua"], "^", "-")]
    for ax, (reader, main, extra) in zip(axes[0], RUNS):
        a = L(main); x = L(extra)
        if a is None: ax.set_axis_off(); continue
        rows.append(f"\\multicolumn{{8}}{{l}}{{\\textit{{{reader}}}, full context {100*a['full']['mean']:.1f} at {a['full']['tokens']:.0f} tokens}}")
        for k, lab in ARMS:
            src = a if k in a["table"] else x if x and k in x["table"] else None
            if src is None: continue
            t = src["table"][k]; rows.append(f"{lab}{DAG if src is x else ''} & " + " & ".join(f"{100*t[str(b)]['mean']:.1f}" for b in BUD) + f" & {100*t['avg']['mean']:.1f}")
        for k, lab in CONTR:
            src = a if k in a["contrasts"] else x if x and k in x["contrasts"] else None
            if src is None: continue
            c = src["contrasts"][k]
            rows.append(f"{lab}{DAG if src is x else ''} & " + " & ".join(f"{100*c[str(b)]['diff']:+.1f}{'*' if c[str(b)].get('p_holm', 1) < .05 else ''}" for b in BUD) + f" & {pc(c['avg'])}{'*' if c['avg'].get('p_holm', 1) < .05 else ''}")
        ax.axhline(100 * a["full"]["mean"], color=C["ink"], lw=0.8, ls=(0, (4, 2)), label="Full context")
        for k, lab, col, mk, ls in ser:
            if k in a["table"]: ax.plot(BUD, [100 * a["table"][k][str(b)]["mean"] for b in BUD], ls=ls, lw=1.4 if ls == "-" else 1.0, color=col, marker=mk, ms=4.5, mec="white", mew=0.6, label=lab)
        ax.set_xscale("log", base=2); ax.set_xticks(BUD); ax.set_xticklabels([str(b) for b in BUD], fontsize=6.5); ax.minorticks_off()
        ax.set_title(reader, fontsize=8, color=C["ink"]); ax.set_xlabel("Context budget (tokens)"); ax.grid(axis="y", color=C["grid"], lw=0.6); ax.set_axisbelow(True)
    axes[0][0].set_ylabel("Cover-EM (%)"); h, l = axes[0][1].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.5, -0.17), ncol=4, frameon=False, fontsize=7); fig.tight_layout(w_pad=0.8)
    save(fig, "e6_budget_curves")
    write("tab_e6_selection.tex", ["Round-2 selection under six token budgets, cover-EM % (repro/e6_analyze.py); * Holm p<0.05 within the family of the run; dagger: separate Spark run with its own references",
          "method or contrast | 128 | 256 | 384 | 512 | 768 | 1024 | mean over budgets (contrasts: [95% CI])"], rows)


def exact_fresh():
    rows = []
    for reader, main, _ in RUNS[1:]:
        a = L(main)
        if a is None: continue
        p = a["probe"]; f = a["fixed"]; c = a["contrasts"]
        rows.append(f"\\multicolumn{{3}}{{l}}{{\\textit{{{reader}}}: flip rate {100*p['flip_rate']:.1f}\\% (earlier probes {100*p['flip_rate_stale']:.1f}\\%), label agreement with earlier probes {100*p['label_agreement_with_stale']:.1f}\\%, same full-context answer {100*p['full_answer_same_as_stale']:.1f}\\%}}")
        for k, lab in (("cfa_exact-ce", "Exact CFA $-$ CE"), ("cfa_exact_bge-bge", "Exact CFA + bge $-$ bge"), ("cfa_exact-cfa_transfer", "Exact CFA $-$ transferred CFA"), ("cfa_exact_bge-cfa_transfer_bge", "Exact CFA + bge $-$ transferred CFA + bge"),
                       ("cfa_exact_bge-cfa_exact_stale_bge", "Exact CFA + bge: fresh $-$ earlier probes"), ("cfa_exact_bge-anscont_bge", "Exact CFA + bge $-$ containment + bge"), ("gold_cfa_exact-gold_oracle", "Gold-first + exact CFA $-$ gold-first")):
            rows.append(f"{lab} & {100*a['table'][k.split('-')[0]]['avg']['mean']:.1f} & {pc(c[k]['avg'])}{'*' if c[k]['avg'].get('p_holm', 1) < .05 else ''}")
        pr = f["cfa_prune"]
        rows.append(f"CFA pruning: {pr['n_selected']:.2f} passages, {pr['tokens']:.0f} tokens, {pr['token_ratio_full_over_sel']:.2f}$\\times$ & {100*pr['mean']:.1f} & vs full {pc(pr['vs_full'])} (Holm p={pr['vs_full']['p_holm_all_arms']:.3f}); vs bge {pc(pr['vs_bge_match_prune'])}; vs containment + bge {pc(pr['vs_anscont_bge_match_prune'])} at equal tokens")
        ap = f["anscont_prune"]
        rows.append(f"Containment pruning: {ap['n_selected']:.2f} passages, {ap['tokens']:.0f} tokens, {ap['token_ratio_full_over_sel']:.2f}$\\times$ & {100*ap['mean']:.1f} & vs full {pc(ap['vs_full'])} (Holm p={ap['vs_full']['p_holm_all_arms']:.3f}); vs bge {pc(ap['vs_bge_match_acprune'])} at equal tokens")
    write("tab_e6_exact_fresh.tex", ["E7: exact CFA from fresh leave-one-out probes of the API readers (repro/e7_reprobe.py, repro/e6_api_sweep.py, repro/e6_analyze.py)",
          "contrast or arm | cover-EM % of the first arm (mean over budgets, or the fixed arm) | difference [95% CI], * Holm p<0.05"], rows)


LAB = {"bge": "bge", "ce": "CE", "anscont_bge": "containment", "anscont_donors_bge": "donors' containment", "anscont_gemma4b_bge": "Gemma containment", "cfa_exact_bge": "exact CFA",
       "contextcite": "ContextCite-32", "cfa_transfer_bge": "transferred CFA", "sum_anscont_transfer_bge": "containment + transferred CFA", "sum_anscont_exact_bge": "containment + exact CFA",
       "cfa_transfer_local2_bge": "CFA of two Gemma", "cfa_transfer_gemma4b_bge": "CFA of Gemma-4B"}


def cost():
    c = L("e8_cost.json"); rows = []
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6), squeeze=False); num = {k: i + 1 for i, k in enumerate(LAB)}
    for ax, (reader, r) in zip(axes[0], c["readers"].items()):
        if not r["arms"]: ax.set_axis_off(); continue
        rows.append(f"\\multicolumn{{6}}{{l}}{{\\textit{{{reader}}} ({r['donors']} donors), full context {100*r['full']['mean']:.1f}}}")
        seen = {}
        for k, lab in LAB.items():
            if k not in r["arms"]: continue
            a = r["arms"][k]
            rows.append(f"{lab}{' + bge' if k not in ('bge', 'ce', 'contextcite') else ''}{DDAG if a['run'] == 'r1' and reader.startswith('Qwen') else ''} & {a['target_calls']:.1f} & {a['donor_calls']:.1f} & {a['total_calls']:.1f} & {100*a['avg_cover']:.1f} [{100*a['avg_ci95'][0]:.1f}, {100*a['avg_ci95'][1]:.1f}] & {'yes' if a['pareto'] else ''}")
            x = a["total_calls"] + 1; y = 100 * a["avg_cover"]; col = C["violet"] if a["pareto"] else C["muted"]
            ax.plot(x, y, "o" if a["donor_calls"] == 0 else "s", ms=4.5, color=col, mec="white", mew=0.6, zorder=3)
            n = sum(1 for (xx, yy) in seen.values() if abs(np.log(xx / x)) < 0.25 and abs(yy - y) < 0.9); seen[k] = (x, y)
            ax.annotate(str(num[k]), (x, y), xytext=(4 + 7 * n, 1), textcoords="offset points", fontsize=6, color=C["ink"])
        par = sorted([(a["total_calls"] + 1, 100 * a["avg_cover"]) for k, a in r["arms"].items() if a["pareto"] and k in LAB])
        ax.step([p[0] for p in par] + [200], [p[1] for p in par] + [par[-1][1]], where="post", color=C["violet"], lw=0.9, zorder=1)
        ax.axhline(100 * r["full"]["mean"], color=C["ink"], lw=0.8, ls=(0, (4, 2)))
        ax.set_xscale("log"); ax.set_xlim(0.8, 200); ax.set_xticks([1, 2, 6, 15, 34, 73]); ax.set_xticklabels(["0", "1", "5", "14", "33", "72"], fontsize=6.5); ax.minorticks_off()
        ax.set_title(reader, fontsize=8, color=C["ink"]); ax.set_xlabel("LLM calls before the answer"); ax.grid(axis="y", color=C["grid"], lw=0.6); ax.set_axisbelow(True)
    axes[0][0].set_ylabel("Cover-EM, mean over budgets (%)"); fig.tight_layout(w_pad=1.0)
    key = "   ".join(f"{num[k]} {lab}" for k, lab in LAB.items())
    fig.text(0.5, -0.02, key[: key.index("   7 ")], ha="center", va="top", fontsize=6.3, color=C["ink"]); fig.text(0.5, -0.07, key[key.index("   7 ") + 3:], ha="center", va="top", fontsize=6.3, color=C["ink"])
    fig.text(0.5, -0.12, "circles: calls to the answering reader only; squares: calls to donor models; violet: Pareto-optimal; dashed: full context", ha="center", va="top", fontsize=6.3, color=C["muted"])
    save(fig, "e6_pareto")
    write("tab_e6_cost.tex", ["E8: LLM calls per question before the final answer against cover-EM averaged over six budgets (repro/e8_cost.py); double dagger: arm generated in the round-1 Spark run (same stack)",
          "arm | calls to the answering reader | calls to donor models | total | cover-EM % [95% CI] | Pareto-optimal"], rows)
    q = L("e6_qwen_r1_analysis.json"); rows = []
    for arm, lab in (("cfa_transfer_bge", "Transferred CFA + bge"), ("bge", "bge-reranker-v2-m3"), ("cfa_exact_bge", "Exact CFA + bge"), ("contextcite", "ContextCite-32")):
        for b in BUD:
            v = q["table"][arm][str(b)]["vs_full"]
            rows.append(f"{lab} & {b} & {100*q['table'][arm][str(b)]['mean']:.1f} & {pc(v)} & {v['p']:.4f} & {v['p_holm_six_budgets']:.3f} & {v['p_holm_all_arms']:.3f}")
    write("tab_e6_full_family.tex", [f"E8: budgeted selections against the full context, Qwen3-4B, round-1 sweep (repro/e6_analyze.py qwen_r1); full context {100*q['full']['mean']:.1f}; B=10000",
          f"arm | budget | cover-EM % | difference to full context [95% CI] | unadjusted p | Holm p over the six budgets of the arm | Holm p over all {q['full_context_family']['size_all_arms']} label-free selections of the run"], rows)


if __name__ == "__main__":
    o = L("e6_offline.json"); within_gold(o); incremental(o); family_transfer(o); selection(); exact_fresh(); cost()
    print("done")
