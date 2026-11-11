"""E5 (b). Semantic-equivalence check of the answer-change label.

Builds a frozen sample of 300 (full answer, leave-one-out answer) pairs from the E1
run: 100 per dataset, SHA-256 ordered, half with a changed normalized string and half
unchanged.  Writes a blank two-annotator sheet (CSV) for human labelling and, as an
automatic proxy only, asks an LLM judge (Gemini; the OpenAI key was not valid on the
run date) whether the two answers are equivalent (--judge).  The proxy is NOT a human annotation.
"""
import csv, hashlib, json, os, sys, time
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from cfa_lib import jl, normalize

JUDGE_PROMPT = ("Two short answers to the same question are given. Decide whether they are semantically equivalent "
                "answers to the question (same entity, value, or yes/no meaning; ignore wording, articles, extra explanation).\n"
                "Question: {q}\nAnswer A: {a}\nAnswer B: {b}\nReply with exactly one word: SAME or DIFFERENT.")


def sample():
    rows = jl(HERE / "inputs_e1/e1_loo.jsonl"); out = []
    for ds in sorted({r["dataset"] for r in rows}):
        cand = []
        for r in rows:
            if r["dataset"] != ds: continue
            for i, a in enumerate(r["delete_answers"]):
                cand.append({"pair_id": f"{r['question_group']}|{i}", "dataset": ds, "question": r["question"], "answer_full": r["full_answer"],
                             "answer_deleted": a, "changed_by_rule": int(normalize(a) != normalize(r["full_answer"]))})
        cand.sort(key=lambda c: hashlib.sha256(c["pair_id"].encode()).hexdigest())
        out += [c for c in cand if c["changed_by_rule"]][:50] + [c for c in cand if not c["changed_by_rule"]][:50]
    return out


def main():
    pairs = sample()
    with open(HERE / "outputs/e5_equivalence_annotation_sheet.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["pair_id", "dataset", "question", "answer_full", "answer_deleted", "annotator1_same(1/0)", "annotator2_same(1/0)"])
        order = sorted(pairs, key=lambda c: hashlib.sha256(("shuffle" + c["pair_id"]).encode()).hexdigest())
        for c in order: w.writerow([c["pair_id"], c["dataset"], c["question"], c["answer_full"], c["answer_deleted"], "", ""])
    print("sheet rows", len(pairs))
    if "--judge" not in sys.argv: return
    for line in open(".env"):
        if line.startswith("GOOGLE_API_KEY="): os.environ["GOOGLE_API_KEY"] = line.split("=", 1)[1].strip().strip('"').strip("'")
    import logging
    from google import genai
    from google.genai import types as gt
    from concurrent.futures import ThreadPoolExecutor
    logging.getLogger("google_genai").setLevel(logging.ERROR); logging.getLogger("google_genai.types").setLevel(logging.ERROR)
    client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"], http_options=gt.HttpOptions(timeout=60000))
    JUDGE = "gemini-3.5-flash"
    def judge(c):
        for attempt in range(5):
            try:
                r = client.models.generate_content(model=JUDGE, contents=JUDGE_PROMPT.format(q=c["question"], a=c["answer_full"], b=c["answer_deleted"]),
                                                   config=gt.GenerateContentConfig(temperature=0.0, max_output_tokens=256))
                t = (r.text or "").upper()
                if "DIFF" in t or "SAME" in t: return int("DIFF" in t)
            except Exception: time.sleep(2 * (attempt + 1))
        return None
    with ThreadPoolExecutor(8) as ex: verdict = list(ex.map(judge, pairs))
    for c, v in zip(pairs, verdict): c["judge_different"] = v
    ok = [c for c in pairs if c["judge_different"] is not None]
    a = np.array([c["changed_by_rule"] for c in ok]); b = np.array([c["judge_different"] for c in ok])
    po = np.mean(a == b); pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    res = {"n": len(ok), "judge": JUDGE + " (automatic proxy, not human)", "agreement": float(po), "kappa": float((po - pe) / (1 - pe)),
           "rule_changed_but_judge_same": float(np.mean(b[a == 1] == 0)), "rule_same_but_judge_different": float(np.mean(b[a == 0] == 1)),
           "by_dataset": {d: {"changed_but_same": float(np.mean([1 - c["judge_different"] for c in ok if c["dataset"] == d and c["changed_by_rule"]]))} for d in sorted({c["dataset"] for c in ok})},
           "pairs": pairs}
    (HERE / "outputs/e5_equivalence_judge.json").write_text(json.dumps(res, indent=1, ensure_ascii=False) + "\n")
    print({k: v for k, v in res.items() if k != "pairs"})


if __name__ == "__main__":
    main()
