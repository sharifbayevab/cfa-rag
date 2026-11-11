"""Derive the manuscript table bodies from the generated tab_*.tex rows and analysis JSON.

No number is typed here: rows are re-labelled, re-grouped or column-filtered copies of
tab_budget_ladder.tex, tab_gold_internal_budget.tex, tab_gold_internal_transfer.tex,
tab_e0_information.tex, tab_kappa.tex, tab_target_ablation.tex, and the per-dataset
contrasts are read from outputs/*_analysis.json.
"""
import json, re
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def rows(name):
    return [l.rstrip() for l in (ROOT / name).read_text().splitlines() if l.strip() and not l.startswith("%")]


def minus(s):
    """Typeset ASCII minus signs of numbers as math minus."""
    return re.sub(r"(?<![\w$.])-(?=\d)", r"$-$", s)


def group(line):
    m = re.match(r"\\multicolumn\{8\}\{l\}\{\\textit\{(.+?)\}(.*)\} \\\\", line)
    return m.group(1), m.group(2).strip()


# ---- Table: cover-EM levels and paired differences at equal token budgets
ladder, goldb = rows("tab_budget_ladder.tex"), rows("tab_gold_internal_budget.tex")
gold_extra, gold_delta, cur = {}, {}, None
for l in goldb:
    if l.startswith("\\multicolumn"):
        cur = group(l)[0]
    elif l.startswith("Gold-first + transferred CFA"):
        gold_extra[cur] = l
    elif l.startswith("$\\Delta$ Gold-first + transferred CFA"):
        gold_delta[cur] = l
LEVEL = {"Transferred CFA (CE ties)": "Transferred CFA", "Exact CFA (CE ties)": "Exact CFA",
         "Gold-first oracle": "Gold-first (annotation)"}
DELTA = {"$\\Delta$ Exact CFA + bge $-$ bge-reranker-v2-m3": "Exact CFA + bge $-$ bge",
         "$\\Delta$ Transferred CFA + bge $-$ bge-reranker-v2-m3": "Transferred CFA + bge $-$ bge",
         "$\\Delta$ Exact CFA (CE ties) $-$ Cross-encoder": "Exact CFA $-$ cross-encoder",
         "$\\Delta$ Transferred CFA (CE ties) $-$ Cross-encoder": "Transferred CFA $-$ cross-encoder"}
ORDER = ["Transferred CFA $-$ cross-encoder", "Transferred CFA + bge $-$ bge",
         "Exact CFA $-$ cross-encoder", "Exact CFA + bge $-$ bge"]
lev, dif, cur, first = [], [], None, True
blocks = {}
for l in ladder:
    if l.startswith("\\multicolumn"):
        cur = group(l); blocks[cur] = {"lev": [], "dif": {}}
        continue
    label, rest = l.split(" & ", 1)
    if label.startswith("$\\Delta$"):
        blocks[cur]["dif"][DELTA[label]] = rest
    else:
        blocks[cur]["lev"].append(LEVEL.get(label, label) + " & " + rest)
for i, ((name, note), b) in enumerate(blocks.items()):
    sep = "" if i == 0 else "\\midrule\n"
    lev.append(sep + f"\\multicolumn{{8}}{{l}}{{\\textit{{{name}}} {note}}} \\\\")
    lev += b["lev"]
    lev.append(gold_extra[name].replace("Gold-first + transferred CFA", "Gold-first + transferred CFA"))
    dif.append(sep + f"\\multicolumn{{8}}{{l}}{{\\textit{{{name}}}}} \\\\")
    dif += [k + " & " + b["dif"][k] for k in ORDER]
    dif.append(gold_delta[name].replace("$\\Delta$ Gold-first + transferred CFA", "Gold-first + transferred CFA $-$ gold-first"))
(ROOT / "tab_paper_levels.tex").write_text(minus("\n".join(lev)) + "\n")
(ROOT / "tab_paper_diffs.tex").write_text(minus("\n".join(dif)) + "\n")

# ---- Table: gold-internal transfer (drop the interval of the transferred score; the figure shows it)
out = []
for l in rows("tab_gold_internal_transfer.tex"):
    c = l[:-2].strip().split(" & ")
    c[2] = c[2].split(" [")[0]
    out.append(" & ".join(c) + " \\\\")
(ROOT / "tab_paper_goldtransfer.tex").write_text(minus("\n".join(out)) + "\n")

# ---- Table: learned information test, within-question columns only
out = []
for l in rows("tab_e0_information.tex"):
    c = l[:-2].strip().split(" & ")
    out.append(" & ".join([c[0], c[5], c[6], c[7], c[8]]) + " \\\\")
(ROOT / "tab_paper_e0.tex").write_text(minus("\n".join(out)) + "\n")

# ---- Table: kappa, the two samples side by side
k = {}
for l in rows("tab_kappa.tex"):
    c = l[:-2].strip().split(" & ")
    k.setdefault(c[1], {})["120" if "120" in c[0] else "450"] = (c[2], c[3])
out = [f"{s} & {v['450'][0]} & {v['450'][1]} & {v['120'][0]} & {v['120'][1].split(' [')[0]} \\\\" for s, v in k.items()]
(ROOT / "tab_paper_kappa.tex").write_text("\n".join(out) + "\n")

# ---- Table: target ablation with short labels
SHORT = {"Cross-encoder (no training)": "Cross-encoder, no training", "Target: CE pseudo-labels": "CE pseudo-labels",
         "Target: gold support": "Gold support", "Target: CFA (five generators)": "Deletion labels",
         "Target: signed support (five teachers)": "Signed support", "OOF classifier (manuscript selector)": "Deletion-label classifier"}
out = []
for l in rows("tab_target_ablation.tex"):
    label, rest = l.split(" & ", 1)
    out.append(SHORT[label] + " & " + rest)
(ROOT / "tab_paper_targets.tex").write_text("\n".join(out) + "\n")

# ---- Table: per-dataset mean differences over the six budgets
def ci(x):
    return f"{100 * x['diff']:+.1f} [{100 * x['ci95'][0]:+.1f}, {100 * x['ci95'][1]:+.1f}]"

out = []
for name, f in (("Qwen3-4B", "e1_sweep_analysis.json"), ("GPT-4.1-mini", "e4_gpt41mini_analysis.json"),
                ("Gemini-3.5-flash-lite", "e4_gemini35fl_analysis.json")):
    c = json.load(open(HERE / "outputs" / f))["contrasts"]
    for key, lab in (("cfa_transfer-ce", "transferred CFA $-$ cross-encoder"), ("cfa_transfer_bge-bge", "transferred CFA + bge $-$ bge")):
        d = c[key]["by_dataset_avg"]
        out.append(f"{name}, {lab} & {ci(d['hotpotqa'])} & {ci(d['2wiki'])} & {ci(d['musique'])} \\\\")
(ROOT / "tab_paper_bydataset.tex").write_text(minus("\n".join(out)) + "\n")
print("wrote tab_paper_*.tex")
