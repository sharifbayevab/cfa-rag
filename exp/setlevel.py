# -*- coding: utf-8 -*-
"""Set-level ablation to bound the single-fragment LOO under-crediting of
redundant evidence (the confound that inflates the "ungrounded"/%NoC statistic).

For each correct answer we additionally ablate the ENTIRE gold supporting set at
once and regenerate. If the answer flips, the gold set is collectively
load-bearing even when no single fragment was -- so a single-fragment %NoC label
was an artifact of redundancy. We report the original %NoC and the
set-corrected %NoC (correct answers that flip under neither single-fragment nor
gold-set removal, i.e. genuinely ungrounded / parametric).
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

from exp import metrics
from exp.data import load_samples
from exp.generators import make_generator
from exp.rag import answer

RES = Path(__file__).resolve().parent.parent / "results"


def run(dataset, generator_spec, n, seed):
    gen = make_generator(generator_spec)
    raw = {json.loads(l)["qid"]: json.loads(l)
           for l in (RES / f"raw_{dataset}_{generator_spec.replace(':','-')}_n{n}.jsonl").read_text().splitlines() if l.strip()}
    samples = {s["qid"]: s for s in load_samples(dataset, n=n, seed=seed)}

    correct = single_ung = set_grounded_among_ung = corrected_ung = 0
    for qid, r in raw.items():
        if r["judge_full"] <= 0.5:        # cover-EM correct only
            continue
        s = samples.get(qid)
        if s is None:
            continue
        correct += 1
        gold = [i for i, f in enumerate(s["fragments"]) if f.is_gold]
        is_single_ung = len(r["causal_idxs"]) == 0
        if not gold:
            if is_single_ung:
                single_ung += 1; corrected_ung += 1
            continue
        # ablate the whole gold set, regenerate
        idxs = [i for i in range(len(s["fragments"])) if i not in set(gold)]
        ans_noG = answer(gen, s["question"], s["fragments"], idxs)
        set_flip = metrics.exact_match(ans_noG, r["ans_full"]) < 0.5
        if is_single_ung:
            single_ung += 1
            if set_flip:
                set_grounded_among_ung += 1   # was an LOO artifact
            else:
                corrected_ung += 1            # genuinely ungrounded
    c = max(correct, 1)
    return {
        "dataset": dataset, "generator": generator_spec, "n_correct": correct,
        "orig_NoC": round(single_ung / c, 3),
        "setlevel_grounded_among_ungrounded": round(set_grounded_among_ung / max(single_ung, 1), 3),
        "corrected_NoC": round(corrected_ung / c, 3),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", nargs="*", default=[
        "hotpotqa:openai:gpt-4.1-mini:200", "2wiki:openai:gpt-4.1-mini:200"])
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()
    out = []
    for cell in args.cells:
        ds, gen_a, gen_b, n = cell.split(":")
        gen = f"{gen_a}:{gen_b}"
        r = run(ds, gen, int(n), args.seed)
        print(json.dumps(r)); out.append(r)
    (RES / "setlevel.json").write_text(json.dumps(out, indent=2))
    # macros
    import numpy as np
    om = np.mean([x["orig_NoC"] for x in out]); cm = np.mean([x["corrected_NoC"] for x in out])
    art = np.mean([x["setlevel_grounded_among_ungrounded"] for x in out])
    macros = [
        f"\\newcommand{{\\slOrigNoC}}{{{om:.2f}}}",
        f"\\newcommand{{\\slCorrNoC}}{{{cm:.2f}}}",
        f"\\newcommand{{\\slArtifact}}{{{art*100:.0f}}}",
    ]
    for o in [Path(__file__).resolve().parent.parent/"article2", Path(__file__).resolve().parent.parent/"article3"]:
        (o/"setlevelmacros.tex").write_text("\n".join(macros)+"\n")
    print("orig_NoC=%.3f corrected_NoC=%.3f artifact_frac=%.3f" % (om, cm, art))


if __name__ == "__main__":
    main()
