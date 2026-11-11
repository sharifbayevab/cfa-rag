"""Rebuild corrected CFA tables, statistics and figures from immutable caches.

No model calls, downloads, pickle loading, or writes outside this manuscript.
Run from any directory: python3 /path/to/repro/rebuild.py
"""
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/cfa-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
PAPER = HERE.parent
INPUT = HERE / "inputs"
OUTPUT = HERE / "outputs"
MODELS = ["ollama-gemma3-4b", "openai-gpt-4.1-nano", "ollama-gemma3-12b",
          "openai-gpt-4.1-mini", "openai-gpt-4.1"]
LABELS = dict(zip(MODELS, ["Gemma3-4B", "GPT-4.1-nano", "Gemma3-12B", "GPT-4.1-mini", "GPT-4.1"]))
DATASETS = ["hotpotqa", "2wiki", "musique"]
DS_LABEL = dict(zip(DATASETS, ["HotpotQA", "2Wiki", "MuSiQue"]))


def verify_inputs():
    for item in json.loads((HERE / "input_manifest.json").read_text()):
        actual = hashlib.sha256((INPUT / item["file"]).read_bytes()).hexdigest()
        if actual != item["sha256"]:
            raise ValueError(f"Input checksum differs: {item['file']}")


def load_grid():
    cells = {}
    for ds in DATASETS:
        for gen in MODELS:
            path = INPUT / f"raw_{ds}_{gen}_n150.jsonl"
            rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
            assert len(rows) == 150
            for row in rows:
                row["dataset"], row["generator"] = ds, gen
                row["group"] = ds + ":" + row["qid"]
            cells[ds, gen] = rows
    for ds in DATASETS:
        assert all({r["qid"] for r in cells[ds, g]} == {r["qid"] for r in cells[ds, MODELS[0]]} for g in MODELS)
    return cells


def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if a | b else 1.0


def random_jaccard(m, s, k):
    return sum(math.comb(s, x) * math.comb(m-s, k-x) / math.comb(m, k) *
               (x/(s+k-x) if s+k-x else 1)
               for x in range(max(0, s+k-m), min(s, k)+1))


def write_table(name, spec, header, rows):
    lines = [r"\begin{tabular}{"+spec+"}", r"\toprule", header+r"\\", r"\midrule"]
    lines += [" & ".join(row)+r"\\" for row in rows]
    lines += [r"\bottomrule", r"\end{tabular}"]
    (PAPER / name).write_text("\n".join(lines)+"\n")


def save_figure(fig, name):
    fig.tight_layout()
    fig.savefig(PAPER / "images" / (name+".pdf"), bbox_inches="tight")
    plt.close(fig)


def main():
    verify_inputs()
    OUTPUT.mkdir(exist_ok=True)
    cells = load_grid()
    rows = [r for rr in cells.values() for r in rr]
    # Resample questions within dataset, retaining all five generators together.
    grouped = defaultdict(dict)
    for r in rows:
        grouped[r["dataset"]].setdefault(r["qid"], {})[r["generator"]] = r
    assert all(len(g)==5 for ds in grouped.values() for g in ds.values())
    arrays = {}
    for ds in DATASETS:
        perq = []
        for qid in sorted(grouped[ds]):
            rr = [grouped[ds][qid][g] for g in MODELS]
            correct = np.array([r["cover_full"] for r in rr])
            par = np.array([r["judge_cb"] for r in rr])
            j = np.array([jaccard(r["causal_idxs"], r["toprel_idxs"]) for r in rr])
            null = np.array([random_jaccard(len(r["rel_scores"]), len(r["causal_idxs"]), len(r["toprel_idxs"])) for r in rr])
            delta = np.array([r["selection"]["causal_prune"]["cover"]-r["selection"]["full"]["cover"] for r in rr])
            perq.append([delta.mean(), (correct*par).sum(), correct.sum(), j.mean(), null.mean()])
        arrays[ds] = np.array(perq)
    rng = np.random.default_rng(0)
    sampled = np.zeros((10000, 5))
    for ds in DATASETS:
        a = arrays[ds]
        sampled += a[rng.integers(0, len(a), size=(len(sampled), len(a)))].mean(1)/3
    point = np.concatenate(list(arrays.values())).mean(0)
    delta_ci = np.percentile(sampled[:, 0], [2.5, 97.5])
    par_ci = np.percentile(sampled[:, 1]/sampled[:, 2], [2.5,97.5])
    j_ci = np.percentile(sampled[:,3], [2.5,97.5])
    excess_ci = np.percentile(sampled[:,3]-sampled[:,4], [2.5,97.5])
    stats = {"question_model_observations": len(rows), "unique_questions": 450,
             "bootstrap": {"unit": "question", "strata": DATASETS, "generators": "fixed, kept together", "seed": 0, "resamples": 10000},
             "delta_cover": float(point[0]), "delta_ci95": delta_ci.tolist(),
             "delta_lower_one_sided95": float(np.percentile(sampled[:,0],5)),
             "pooled_closedbook_given_correct": float(point[1]/point[2]), "pooled_closedbook_ci95": par_ci.tolist(),
             "jaccard_all": float(point[3]), "jaccard_ci95": j_ci.tolist(),
             "jaccard_chance": float(point[4]), "jaccard_excess_ci95": excess_ci.tolist()}
    stats["jaccard_nonempty"] = float(np.mean([jaccard(r["causal_idxs"],r["toprel_idxs"]) for r in rows if r["causal_idxs"]]))
    stats["jaccard_nonempty_chance"] = float(np.mean([random_jaccard(len(r["rel_scores"]),len(r["causal_idxs"]),len(r["toprel_idxs"])) for r in rows if r["causal_idxs"]]))
    words_full = np.array([r["selection"]["full"]["n_tokens"] for r in rows])
    words_prune = np.array([r["selection"]["causal_prune"]["n_tokens"] for r in rows])
    m = np.array([len(r["rel_scores"]) for r in rows])
    stats["final_context_word_ratio"] = float(words_full.sum()/words_prune.sum())
    stats["total_probe_and_final_context_word_ratio"] = float((m*words_full+words_prune).sum()/words_full.sum())
    stats["empty_set_rate_all"] = float(np.mean([not r["causal_idxs"] for r in rows]))
    details, model_table, rel_table = [], [], []
    def summary(rr):
        correct = [r for r in rr if r["cover_full"]]
        return [np.mean([r["cover_full"] for r in rr]), np.mean([r["judge_cb"] for r in rr]),
                np.mean([r["judge_cb"] for r in correct]), np.mean([not r["causal_idxs"] for r in correct]),
                np.mean([len(set(r["causal_idxs"])&set(r["gold_idxs"]))/len(r["causal_idxs"]) if r["causal_idxs"] else 0 for r in correct])]
    for g in MODELS:
        ss=[]
        for ds in DATASETS:
            rr=cells[ds,g];s=summary(rr);ss.append(s)
            j=np.mean([jaccard(r["causal_idxs"],r["toprel_idxs"]) for r in rr])
            details.append([LABELS[g],DS_LABEL[ds],"150"]+[f"{v:.2f}" for v in s[:3]]+[f"{j:.2f}",f"{s[4]:.2f}"])
        model_table.append([LABELS[g]]+[f"{v:.2f}" for v in np.mean(ss,axis=0)])
        rr=[r for ds in DATASETS for r in cells[ds,g]]
        rel_table.append([LABELS[g],f"{np.mean([jaccard(r['causal_idxs'],r['toprel_idxs']) for r in rr]):.3f}",
                          f"{np.mean([random_jaccard(len(r['rel_scores']),len(r['causal_idxs']),len(r['toprel_idxs'])) for r in rr]):.3f}"])
    write_table("tab_faithfulness.tex","lccccc",r"Model & Acc & Acc$_{\rm cb}$ & Par & NoC & cgP",model_table)
    write_table("tab_faithfulness_full.tex","llcccccc",r"Model & Dataset & $n$ & Acc & Acc$_{\rm cb}$ & Par & Jaccard & cgP",details)
    write_table("tab_relcausal.tex","lcc","Model & Observed & Chance",rel_table)
    quality=[]
    for scope in DATASETS+["all"]:
        rr=[r for r in rows if scope=="all" or r["dataset"]==scope]
        full=np.mean([r["selection"]["full"]["cover"] for r in rr]);prune=np.mean([r["selection"]["causal_prune"]["cover"] for r in rr])
        wf=np.mean([r["selection"]["full"]["n_tokens"] for r in rr]);wp=np.mean([r["selection"]["causal_prune"]["n_tokens"] for r in rr])
        quality.append([DS_LABEL.get(scope,"Pooled"),f"{full:.3f}",f"{prune:.3f}",f"{100*(prune-full):+.2f}",f"{wf:.0f}",f"{wp:.0f}"])
    write_table("tab_selection.tex","lccccc",r"Dataset & Full & Pruned & $\Delta$ (pp) & W$_{\rm full}$ & W$_{\rm prune}$",quality)
    stats["quality_secondary"]={}
    for metric in ["em","f1"]:
        stats["quality_secondary"][metric]={k:float(np.mean([r["selection"][k][metric] for r in rows])) for k in ["full","causal_prune"]}
    # Historical six-cell method table: explicitly exclude the non-held-out learned selector.
    methods=[json.loads(p.read_text()) for p in sorted(INPUT.glob("methods_*.json"))]
    assert len(methods)==6
    names={"full":"Full context","random_k":"Random-4","topk_relevance":"Dense top-4",
           "contextcite":"Mask-surrogate", "cfa_prune":"CFA pruning", "oracle_gold":"Gold-only"}
    method_rows=[]
    for method,name in names.items():
        weak=[d["methods"][method]["acc"] for d in methods if d["generator"].startswith("ollama:")]
        strong=[d["methods"][method]["acc"] for d in methods if d["generator"].startswith("openai:")]
        w=[np.mean([d["methods"][method]["tokens"] for d in methods if d["dataset"]==ds]) for ds in DATASETS]
        probes=np.mean([d["methods"][method]["probe_calls"] for d in methods])
        gr=np.mean([d["methods"][method]["gold_recall"] for d in methods])
        method_rows.append([name,f"{np.mean(weak):.2f}",f"{np.mean(strong):.2f}"]+[f"{x:.0f}" for x in w]+[f"{probes:.1f}",f"{gr:.2f}"])
    write_table("tab_methods.tex","lccccccc",r"Method & Acc$_w$ & Acc$_s$ & Words (HotpotQA) & Words (2Wiki) & Words (MuSiQue) & Probes & GR",method_rows)
    fig,ax=plt.subplots(figsize=(6.5,3.5))
    for method,name in names.items():
        rr=[d["methods"][method] for d in methods if d["generator"].startswith("openai:")]
        x=np.mean([d["tokens"] for d in rr]);y=np.mean([d["acc"] for d in rr])
        ax.scatter(x,y,s=55)
        offsets={"contextcite":(8,-12),"cfa_prune":(-10,10),"full":(-65,8),"random_k":(6,7)}
        ax.annotate(name,(x,y),xytext=offsets.get(method,(6,8)),textcoords="offset points",fontsize=8)
    ax.set(xlabel="Final context length (whitespace words)",ylabel="Cover-EM accuracy",xlim=(0,1200),ylim=(0.20,0.78))
    ax.grid(alpha=.2);save_figure(fig,"selection_pareto")
    # Use paired selection full rather than a different baseline generation in open-corpus table.
    oc=[]
    for ds in DATASETS:
        rr=[json.loads(l) for l in (INPUT/f"raw_oc_{ds}_openai-gpt-4.1-mini_n150.jsonl").read_text().splitlines() if l.strip()]
        correct=[r for r in rr if r["cover_full"]]
        oc.append([DS_LABEL[ds]]+[f"{np.mean([r['selection'][k]['cover'] for r in rr]):.3f}" for k in ["full","causal_prune","topk_relevance"]]+[f"{np.mean([r['judge_cb'] for r in correct]):.2f}",f"{np.mean([jaccard(r['causal_idxs'],r['toprel_idxs']) for r in rr]):.3f}"])
    write_table("tab_opencorpus.tex","lccccc","Dataset & Full & Pruned & Dense-4 & Par & Jaccard",oc)
    # All figures derive from the same rows and use conservative terminology.
    fig,ax=plt.subplots(figsize=(5.5,3.4))
    for g in MODELS:
        s=np.mean([summary(cells[ds,g]) for ds in DATASETS],axis=0)
        ax.scatter(s[0],s[2],s=55);ax.annotate(LABELS[g],(s[0],s[2]),xytext=(4,5),textcoords="offset points",fontsize=8)
    ax.set(xlabel="Full-context cover-EM (mean across datasets)",ylabel="Closed-book answerability among correct",xlim=(.28,.83),ylim=(.19,.50));ax.grid(alpha=.2)
    save_figure(fig,"capability_gap")
    fig,ax=plt.subplots(figsize=(5.5,3.3))
    rates=[[summary(cells[ds,g])[2] for g in MODELS] for ds in DATASETS]
    ax.bar([DS_LABEL[d] for d in DATASETS],np.mean(rates,axis=1),yerr=np.std(rates,axis=1),capsize=4,color=["#2456a6","#088ca5","#dc772b"])
    ax.set(ylabel="Closed-book answerability among correct",ylim=(0,.65));save_figure(fig,"perdataset_param")
    rr=cells["hotpotqa","openai-gpt-4.1-mini"]
    rel=np.concatenate([r["rel_scores"] for r in rr]);score=np.concatenate([r["causal_scores"] for r in rr])
    stats["pearson_gold_assisted_score"] = float(np.corrcoef(rel,score)[0,1])
    fig,ax=plt.subplots(figsize=(5.5,3.4));ax.scatter(rel,score,s=6,alpha=.25)
    ax.text(.03,.93,f"Pearson r = {stats['pearson_gold_assisted_score']:.2f}",transform=ax.transAxes)
    ax.set(xlabel="Dense relevance",ylabel="Auxiliary gold-assisted shift score");save_figure(fig,"rel_vs_causal")
    macro={"revDelta":f"{100*point[0]:+.2f}","revDeltaLo":f"{100*delta_ci[0]:+.2f}","revDeltaHi":f"{100*delta_ci[1]:+.2f}",
           "revLower":f"{100*np.percentile(sampled[:,0],5):+.2f}","revPar":f"{100*point[1]/point[2]:.1f}",
           "revParLo":f"{100*par_ci[0]:.1f}","revParHi":f"{100*par_ci[1]:.1f}",
           "revJacc":f"{point[3]:.3f}","revChance":f"{point[4]:.3f}","revJLo":f"{j_ci[0]:.3f}","revJHi":f"{j_ci[1]:.3f}",
           "revJExLo":f"{excess_ci[0]:.3f}","revJExHi":f"{excess_ci[1]:.3f}",
           "revWords":f"{stats['final_context_word_ratio']:.2f}","revTotalWords":f"{stats['total_probe_and_final_context_word_ratio']:.2f}",
           "revPearson":f"{stats['pearson_gold_assisted_score']:.2f}","revNonemptyJ":f"{stats['jaccard_nonempty']:.3f}","revNonemptyChance":f"{stats['jaccard_nonempty_chance']:.3f}"}
    (PAPER/"revisionmacros.tex").write_text("% Generated by repro/rebuild.py\n"+"\n".join("\\newcommand{\\"+k+"}{"+v+"}" for k,v in macro.items())+"\n")
    (OUTPUT/"statistics.json").write_text(json.dumps(stats,indent=2)+"\n")
    fresh_plan = INPUT / "fresh_selection_plan.jsonl"
    fresh_results = INPUT / "fresh_qa_qwen3_4b_n450.jsonl"
    if fresh_plan.exists() and fresh_results.exists():
        subprocess.run(
            [
                sys.executable,
                str(PAPER / "experiments" / "spark" / "analyze_results.py"),
                "--plan", str(fresh_plan),
                "--results", str(fresh_results),
                "--out-dir", str(OUTPUT),
                "--latex-dir", str(PAPER),
                "--batch16-rows", "232",
                "--full-logit-rows", "249",
                "--cache-disabled-start", "306",
                "--cache-disabled-end", "316",
            ],
            check=True,
        )
    controls_input = INPUT / "intervention_controls_qwen3_4b_n120.jsonl"
    if controls_input.exists():
        subprocess.run(
            [
                sys.executable,
                str(PAPER / "experiments" / "spark" / "analyze_intervention_controls.py"),
                "--input", str(controls_input),
                "--output", str(OUTPUT / "intervention_controls_analysis.json"),
            ],
            check=True,
        )
    efficiency_input = INPUT / "efficiency_sweep_qwen3_4b_n450.jsonl"
    if efficiency_input.exists():
        subprocess.run(
            [
                sys.executable,
                str(PAPER / "experiments" / "spark" / "analyze_efficiency_sweep.py"),
                "--input", str(efficiency_input),
                "--output", str(OUTPUT / "efficiency_sweep_analysis.json"),
            ],
            check=True,
        )
    semantic_input = INPUT / "semantic_label_validation.json"
    if semantic_input.exists():
        shutil.copyfile(semantic_input, OUTPUT / "semantic_label_validation.json")
    subprocess.run([sys.executable, str(HERE / "render_revision_outputs.py")], check=True)
    subprocess.run([sys.executable, str(HERE / "make_diagrams.py")], check=True)
    print(json.dumps(stats,indent=2))


if __name__=="__main__":
    main()
