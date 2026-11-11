"""E0c. Training-free cross-reader transfer on the signed grid (six readers, stored answers).

For each target reader, score every passage by the mean deletion-flip of the other
five readers and ask how well that score ranks the target reader's own labels
*among the annotated gold passages of the same question* (within-question AUC,
questions with both classes among gold passages).  Baselines: cross-encoder,
bge-reranker-v2-m3, dense relevance.  Targets: flip (label-free) and supporting
(deletion turns a correct answer incorrect).  No training, no generator calls.
Dataset-stratified question bootstrap, B=2000, seed 0.
"""
import json, sys
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from cfa_lib import boot_indices, load_features, load_pools, load_signed, paired, mean_ci

READERS = ["openai-gpt-4.1-nano", "openai-gpt-4o-mini", "openai-gpt-4.1-mini", "openai-gpt-4.1", "gemini-gemini-3.5-flash-lite", "ollama-qwen3-4b-instruct"]


def auc(y, s):
    y = np.asarray(y, bool); n1 = y.sum(); n0 = len(y) - n1
    if n1 == 0 or n0 == 0: return np.nan
    r = rankdata(s); return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def main():
    pools = load_pools(); feats, _, names = load_features(); bge = json.load(open(HERE / "outputs/e3_bge_reranker_scores.json"))["scores"]
    S = {r: load_signed(r) for r in READERS}; qs = sorted(pools); ds = np.array([q.split(":")[0] for q in qs]); bi = boot_indices(ds, 2000)
    res = {}
    for target in READERS:
        for tname in ("flip", "supporting"):
            per = {k: [] for k in ("transfer", "ce", "bge", "dense")}; rate = []
            for q in qs:
                fr = S[target][q]["fragments"]; gold = [f["idx"] for f in fr if f["is_gold"]]
                y = np.array([(fr[i]["del_flip"] > 0) if tname == "flip" else (fr[i]["del_role"] == "supporting") for i in gold]); rate.append(y.mean())
                sc = {"transfer": np.array([np.mean([S[o][q]["fragments"][i]["del_flip"] for o in READERS if o != target]) for i in gold]),
                      "ce": feats[q][gold, names.index("ce")], "bge": np.array(bge[q])[gold], "dense": feats[q][gold, names.index("rel")]}
                for k in per: per[k].append(auc(y, sc[k]))
            per = {k: np.array(v) for k, v in per.items()}; valid = ~np.isnan(per["transfer"])
            def nm(a, idx):  # nan-aware bootstrap of the mean over informative questions
                boots = np.array([np.nanmean(a[i]) for i in idx]); return {"mean": float(np.nanmean(a)), "ci95": [float(x) for x in np.percentile(boots, [2.5, 97.5])]}
            entry = {"n_informative_questions": int(valid.sum()), "label_rate_among_gold": float(np.mean(rate))}
            for k in per: entry[k] = nm(per[k], bi)
            for k in ("ce", "bge"):
                d = per["transfer"] - per[k]; boots = np.array([np.nanmean(d[i]) for i in bi])
                entry[f"transfer_minus_{k}"] = {"diff": float(np.nanmean(d)), "ci95": [float(x) for x in np.percentile(boots, [2.5, 97.5])]}
            res[f"{target}|{tname}"] = entry
            print(f"{target:30s} {tname:10s} n={entry['n_informative_questions']:3d} transfer {entry['transfer']['mean']:.3f} [{entry['transfer']['ci95'][0]:.3f},{entry['transfer']['ci95'][1]:.3f}] "
                  f"ce {entry['ce']['mean']:.3f} bge {entry['bge']['mean']:.3f} dense {entry['dense']['mean']:.3f} | vs bge {entry['transfer_minus_bge']['diff']:+.3f} [{entry['transfer_minus_bge']['ci95'][0]:+.3f},{entry['transfer_minus_bge']['ci95'][1]:+.3f}]")
    (HERE / "outputs/e0c_signed_transfer.json").write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    main()
