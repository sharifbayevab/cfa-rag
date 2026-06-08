# -*- coding: utf-8 -*-
"""Dataset loaders -> unified samples for the attribution experiments.

A sample is:
  {qid, question, golds:[str], fragments:[Fragment(idx,title,text,is_gold)], dataset}

We use the *distractor* multi-hop settings (HotpotQA, 2WikiMultiHopQA, MuSiQue)
where each question ships with a candidate paragraph pool (a few gold + several
distractors), so context selection runs over this pool without a retriever.
"""
from __future__ import annotations

import random

from datasets import load_dataset

from exp.rag import Fragment


def _hotpotqa(split, n, seed):
    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split=split, trust_remote_code=True)
    idxs = list(range(len(ds)))
    random.Random(seed).shuffle(idxs)
    out = []
    for i in idxs[:n]:
        ex = ds[i]
        ctx = ex["context"]
        gold_titles = set(ex["supporting_facts"]["title"])
        frags = []
        for j, (title, sents) in enumerate(zip(ctx["title"], ctx["sentences"])):
            frags.append(Fragment(idx=j, title=title, text=" ".join(sents).strip(),
                                   is_gold=title in gold_titles))
        out.append({"qid": str(ex["id"]), "question": ex["question"],
                    "golds": [ex["answer"]], "fragments": frags, "dataset": "hotpotqa"})
    return out


def _twowiki(split, n, seed):
    # 2WikiMultiHopQA mirror with the same context schema as HotpotQA
    for name in ["scholarly-shadows-syndicate/2WikiMultihopQA_with_q_gpt35",
                 "voidful/2WikiMultihopQA"]:
        try:
            ds = load_dataset(name, split=split, trust_remote_code=True)
            break
        except Exception:
            ds = None
    if ds is None:
        return []
    idxs = list(range(len(ds)))
    random.Random(seed).shuffle(idxs)
    out = []
    for i in idxs[:n]:
        ex = ds[i]
        ctx = ex["context"]
        titles = ctx["title"] if isinstance(ctx, dict) else [c[0] for c in ctx]
        sents = ctx["content"] if isinstance(ctx, dict) and "content" in ctx else (
            ctx["sentences"] if isinstance(ctx, dict) else [c[1] for c in ctx])
        sf = ex.get("supporting_facts", {})
        gold_titles = set(sf["title"]) if isinstance(sf, dict) else set(s[0] for s in sf)
        frags = [Fragment(idx=j, title=t, text=" ".join(s) if isinstance(s, list) else str(s),
                          is_gold=t in gold_titles) for j, (t, s) in enumerate(zip(titles, sents))]
        out.append({"qid": str(ex.get("id", i)), "question": ex["question"],
                    "golds": [ex["answer"]], "fragments": frags, "dataset": "2wiki"})
    return out


def _musique(split, n, seed):
    # MuSiQue-Ans: each question has paragraphs with is_supporting flags
    ds = load_dataset("dgslibisey/MuSiQue", split=split, trust_remote_code=True)
    idxs = list(range(len(ds)))
    random.Random(seed).shuffle(idxs)
    out = []
    for i in idxs[:n]:
        ex = ds[i]
        paras = ex["paragraphs"]
        frags = [Fragment(idx=j, title=p.get("title", ""), text=p.get("paragraph_text", ""),
                          is_gold=bool(p.get("is_supporting", False)))
                 for j, p in enumerate(paras)]
        out.append({"qid": str(ex.get("id", i)), "question": ex["question"],
                    "golds": [ex["answer"]], "fragments": frags, "dataset": "musique"})
    return out


LOADERS = {
    "hotpotqa": (_hotpotqa, "validation"),
    "2wiki": (_twowiki, "validation"),
    "musique": (_musique, "validation"),
}


def load_samples(dataset: str, n: int = 100, seed: int = 13):
    fn, split = LOADERS[dataset]
    return fn(split, n, seed)


if __name__ == "__main__":
    import sys
    name = sys.argv[1] if len(sys.argv) > 1 else "hotpotqa"
    s = load_samples(name, n=2)
    print(f"{name}: {len(s)} samples")
    ex = s[0]
    print("Q:", ex["question"][:90], "| A:", ex["golds"])
    print("fragments:", len(ex["fragments"]), "| gold:",
          [f.idx for f in ex["fragments"] if f.is_gold])
    print("frag0:", ex["fragments"][0].title, "->", ex["fragments"][0].text[:80])
