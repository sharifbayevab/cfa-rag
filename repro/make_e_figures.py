"""Figures and LaTeX tables for the 2026-11 experiments.  New files only:
images/e0_gold_internal_transfer.{pdf,png}, images/e1_budget_ladder.{pdf,png},
tab_e0_information.tex, tab_gold_internal_transfer.tex, tab_budget_ladder.tex,
tab_gold_internal_budget.tex, tab_target_ablation.tex, tab_kappa.tex
"""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent; PAPER = HERE.parent; OUT = HERE / "outputs"
C = {"blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "yellow": "#eda100", "violet": "#6250d6", "ink": "#0b0b0b", "muted": "#52514e", "grid": "#e4e3df"}
plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": C["muted"],
                     "axes.labelcolor": C["ink"], "xtick.color": C["muted"], "ytick.color": C["muted"], "pdf.fonttype": 42})
BUD = [128, 256, 384, 512, 768, 1024]
READER_NAMES = {"openai-gpt-4.1-nano": "GPT-4.1-nano", "openai-gpt-4o-mini": "GPT-4o-mini", "openai-gpt-4.1-mini": "GPT-4.1-mini", "openai-gpt-4.1": "GPT-4.1",
                "gemini-gemini-3.5-flash-lite": "Gemini-3.5-flash-lite", "ollama-qwen3-4b-instruct": "Qwen3-4B (Ollama)"}


def save(fig, name):
    for ext in ("pdf", "png"):
        p = PAPER / "images" / f"{name}.{ext}"
        fig.savefig(p, dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig_gold_internal():
    r = json.load(open(OUT / "e0c_signed_transfer.json"))
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.3), sharey=True)
    for ax, t, title in zip(axes, ("supporting", "flip"), ("Target: supporting passage (correct $\\to$ incorrect)", "Target: any answer change")):
        for y, rd in enumerate(READER_NAMES):
            e = r[f"{rd}|{t}"]
            ax.plot(e["transfer"]["ci95"], [y, y], color=C["blue"], lw=1.6, solid_capstyle="round", zorder=2)
            ax.plot(e["transfer"]["mean"], y, "o", ms=6, color=C["blue"], mec="white", mew=0.8, zorder=3, label="Transferred CFA (other five readers)" if y == 0 else None)
            ax.plot(e["bge"]["mean"], y, "s", ms=5.5, color=C["orange"], mec="white", mew=0.8, zorder=3, label="bge-reranker-v2-m3" if y == 0 else None)
            ax.plot(e["ce"]["mean"], y, "^", ms=6, color=C["aqua"], mec="white", mew=0.8, zorder=3, label="Cross-encoder" if y == 0 else None)
        ax.axvline(0.5, color=C["muted"], lw=0.8, ls=(0, (3, 3))); ax.set_xlim(0.25, 1.0); ax.set_title(title, fontsize=8, color=C["ink"])
        ax.set_xlabel("Within-question AUC among gold passages"); ax.grid(axis="x", color=C["grid"], lw=0.6); ax.set_axisbelow(True)
        ax.set_yticks(range(len(READER_NAMES))); ax.set_yticklabels(list(READER_NAMES.values())); ax.invert_yaxis() if ax is axes[0] else None
    axes[0].text(0.507, 0.5, "chance", fontsize=6.5, color=C["muted"], va="center", rotation=90, transform=axes[0].get_xaxis_transform())
    axes[0].legend(loc="upper center", bbox_to_anchor=(1.05, -0.25), ncol=3, frameon=False, fontsize=7.5)
    save(fig, "e0_gold_internal_transfer")


def fig_budget():
    panels = [(OUT / "e1_sweep_analysis.json", "Qwen3-4B-Instruct (new probes)"), (OUT / "e4_gpt41mini_analysis.json", "GPT-4.1-mini (no new probes)"),
              (OUT / "e4_gemini35fl_analysis.json", "Gemini-3.5-flash-lite (no new probes)")]
    panels = [(p, t) for p, t in panels if p.exists()]
    fig, axes = plt.subplots(1, len(panels), figsize=(2.6 * len(panels), 2.5), squeeze=False)
    series = [("gold_oracle", "Gold-first oracle", C["muted"], "D", (0, (2, 2))), ("cfa_exact_bge", "Exact CFA + bge", C["blue"], "o", "-"),
              ("cfa_transfer_bge", "Transferred CFA + bge", C["violet"], "P", "-"), ("contextcite", "ContextCite-32", C["yellow"], "v", "-"),
              ("bge", "bge-reranker-v2-m3", C["orange"], "s", "-"), ("ce", "Cross-encoder", C["aqua"], "^", "-")]
    for ax, (p, title) in zip(axes[0], panels):
        r = json.load(open(p))
        ax.axhline(100 * r["full"]["mean"], color=C["ink"], lw=0.8, ls=(0, (4, 2)), label="Full context (all passages)")
        for key, lab, col, mk, ls in series:
            if key not in r["table"]: continue
            ys = [100 * r["table"][key][str(b)]["mean"] for b in BUD]
            ax.plot(BUD, ys, ls=ls, lw=1.6 if key != "gold_oracle" else 1.0, color=col, marker=mk, ms=5, mec="white", mew=0.7, label=lab)
        ax.set_title(title, fontsize=8, color=C["ink"]); ax.set_xticks(BUD); ax.set_xticklabels(BUD, fontsize=7)
        ax.set_xlabel("Context budget (tokens)"); ax.grid(axis="y", color=C["grid"], lw=0.6); ax.set_axisbelow(True)
    axes[0][0].set_ylabel("Cover-EM (%)")
    h, l = max((a.get_legend_handles_labels() for a in axes[0]), key=lambda x: len(x[1]))
    fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.5, -0.2), ncol=4, frameon=False, fontsize=7.5)
    save(fig, "e1_budget_ladder")


def tex(name, lines): (PAPER / name).write_text("\n".join(lines) + "\n")
def ci(e, s=1.0, f="{:.3f}"): return f"{f.format(s*e['value'] if 'value' in e else s*e.get('mean', e.get('diff')))} [{f.format(s*e['ci95'][0])}, {f.format(s*e['ci95'][1])}]"


def tables():
    E = json.load(open(OUT / "e0_information_test.json"))["estimates"]
    rows = []
    for scope, nm in [("all", "All"), ("hotpotqa", "HotpotQA"), ("2wiki", "2Wiki"), ("musique", "MuSiQue")]:
        g = lambda m, k: E[f"{scope}|{m}|{k}"]["value"]
        d = E[f"{scope}|delta_cfa_vs_F9+gold+perm|wq_auc_gold"]
        rows.append(f"{nm} & {g('F9+gold','auc'):.3f} & {g('F9+gold+cfa','auc'):.3f} & {g('F9+gold','ap'):.3f} & {g('F9+gold+cfa','ap'):.3f} & "
                    f"{g('F9+gold','wq_auc_gold'):.3f} & {g('F9+gold+perm','wq_auc_gold'):.3f} & {g('F9+gold+cfa','wq_auc_gold'):.3f} & {d['value']:+.3f} [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}] \\\\")
    tex("tab_e0_information.tex", ["% Leave-one-generator-out information test (repro/e0_information_test.py)",
        "% columns: scope | pooled AUC F9+gold | +CFA | AP F9+gold | +CFA | within-question AUC among gold: F9+gold | +permuted CFA | +CFA | gain vs permuted [95% CI]"] + rows)
    T = json.load(open(OUT / "e0c_signed_transfer.json"))
    tex("tab_gold_internal_transfer.tex", ["% Training-free cross-reader transfer among gold passages (repro/e0c_signed_transfer.py)",
        "% reader | n questions | transferred CFA [CI] | bge | CE | transfer - bge [CI]   (target: supporting passage)"] +
        [f"{READER_NAMES[r]} & {T[r+'|supporting']['n_informative_questions']} & {ci(T[r+'|supporting']['transfer'])} & {T[r+'|supporting']['bge']['mean']:.3f} & "
         f"{T[r+'|supporting']['ce']['mean']:.3f} & {T[r+'|supporting']['transfer_minus_bge']['diff']:+.3f} [{T[r+'|supporting']['transfer_minus_bge']['ci95'][0]:+.3f}, {T[r+'|supporting']['transfer_minus_bge']['ci95'][1]:+.3f}] \\\\" for r in READER_NAMES])
    names = {"ce": "Cross-encoder", "bge": "bge-reranker-v2-m3", "contextcite": "ContextCite-32", "cfa_transfer": "Transferred CFA (CE ties)", "cfa_exact": "Exact CFA (CE ties)",
             "cfa_transfer_bge": "Transferred CFA + bge", "cfa_exact_bge": "Exact CFA + bge", "gold_oracle": "Gold-first oracle", "gold_cfa_transfer": "Gold-first + transferred CFA",
             "gold_cfa_exact": "Gold-first + exact CFA"}
    lad, gi = [], []
    for path, reader in [(OUT / "e1_sweep_analysis.json", "Qwen3-4B"), (OUT / "e4_gpt41mini_analysis.json", "GPT-4.1-mini"), (OUT / "e4_gemini35fl_analysis.json", "Gemini-3.5-flash-lite")]:
        if not path.exists(): continue
        r = json.load(open(path)); lad.append(f"\\multicolumn{{8}}{{l}}{{\\textit{{{reader}}} (full context {100*r['full']['mean']:.1f}, {r['full']['tokens']:.0f} tokens)}} \\\\")
        for k in ["ce", "bge", "contextcite", "cfa_transfer", "cfa_exact", "cfa_transfer_bge", "cfa_exact_bge", "gold_oracle"]:
            if k in r["table"]: lad.append(f"{names[k]} & " + " & ".join(f"{100*r['table'][k][str(b)]['mean']:.1f}" for b in BUD) + f" & {100*r['table'][k]['avg']['mean']:.1f} \\\\")
        for key in ["cfa_exact_bge-bge", "cfa_transfer_bge-bge", "cfa_exact-ce", "cfa_transfer-ce"]:
            c = r["contrasts"][key]; a, b = key.split("-")
            lad.append(f"$\\Delta$ {names[a]} $-$ {names[b]} & " + " & ".join(f"{100*c[str(x)]['diff']:+.1f}" + ("$^{*}$" if c[str(x)].get("p_holm", 1) < .05 else "") for x in BUD) +
                       f" & {100*c['avg']['diff']:+.1f} [{100*c['avg']['ci95'][0]:+.1f}, {100*c['avg']['ci95'][1]:+.1f}] \\\\")
        gi.append(f"\\multicolumn{{8}}{{l}}{{\\textit{{{reader}}}}} \\\\")
        for k in ["gold_oracle", "gold_cfa_transfer", "gold_cfa_exact"]:
            gi.append(f"{names[k]} & " + " & ".join(f"{100*r['table'][k][str(b)]['mean']:.1f}" for b in BUD) + f" & {100*r['table'][k]['avg']['mean']:.1f} \\\\")
        for key in ["gold_cfa_transfer-gold_oracle", "gold_cfa_exact-gold_oracle"]:
            c = r["contrasts"][key]
            gi.append(f"$\\Delta$ {names[key.split('-')[0]]} & " + " & ".join(f"{100*c[str(x)]['diff']:+.1f}" + ("$^{*}$" if c[str(x)].get("p_holm", 1) < .05 else "") for x in BUD) +
                      f" & {100*c['avg']['diff']:+.1f} [{100*c['avg']['ci95'][0]:+.1f}, {100*c['avg']['ci95'][1]:+.1f}] \\\\")
        if reader == "Qwen3-4B":
            tn = {"ce": "Cross-encoder (no training)", "tgt_ce": "Target: CE pseudo-labels", "tgt_gold": "Target: gold support", "tgt_cfa": "Target: CFA (five generators)",
                  "tgt_signed": "Target: signed support (five teachers)", "oof": "OOF classifier (manuscript selector)"}
            tex("tab_target_ablation.tex", ["% Same nine features and folds, different training targets; cover-EM (%) of Qwen3-4B at equal token budgets",
                "% arm | 128 | 256 | 384 | 512 | 768 | 1024 | mean"] + [f"{tn[k]} & " + " & ".join(f"{100*r['table'][k][str(b)]['mean']:.1f}" for b in BUD) + f" & {100*r['table'][k]['avg']['mean']:.1f} \\\\" for k in tn])
    hdr = "% arm | 128 | 256 | 384 | 512 | 768 | 1024 | mean over budgets (difference rows: [95% CI]); * Holm-adjusted p<0.05 within family"
    tex("tab_budget_ladder.tex", ["% Cover-EM (%) at equal context-token budgets (repro/analyze_sweep.py)", hdr] + lad)
    tex("tab_gold_internal_budget.tex", ["% Ordering inside the annotated support set: cover-EM (%) at equal token budgets", hdr] + gi)
    K = json.load(open(OUT / "e5_kappa.json")); kr = []
    for fn, lab in [("intervention_controls_qwen3_4b_n120.jsonl", "n=120 control (frozen)"), ("e1_loo_masks.jsonl", "n=450 (E1 stack)")]:
        if fn not in K: continue
        for s, nm in [("all", "All passages"), ("all|gold", "Gold passages"), ("all|nongold", "Non-gold passages"), ("hotpotqa", "HotpotQA"), ("2wiki", "2Wiki"), ("musique", "MuSiQue")]:
            e = K[fn][s]; kr.append(f"{lab} & {nm} & {100*e['agreement']:.1f} & {e['kappa']:.3f} [{e['kappa_ci95'][0]:.3f}, {e['kappa_ci95'][1]:.3f}] \\\\")
    tex("tab_kappa.tex", ["% Deletion vs position-preserving mask labels: raw agreement (%) and Cohen's kappa [95% CI] (repro/e5_kappa.py)"] + kr)


if __name__ == "__main__":
    fig_gold_internal(); fig_budget(); tables(); print("ok")
