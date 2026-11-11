"""E3. Score every (question, passage) pair of the frozen pools with BAAI/bge-reranker-v2-m3 (CPU, deterministic)."""
import json, hashlib
from pathlib import Path
import torch
from sentence_transformers import CrossEncoder
HERE = Path(__file__).resolve().parent
MODEL = "BAAI/bge-reranker-v2-m3"
torch.manual_seed(0)
ce = CrossEncoder(MODEL, device="cpu", max_length=512)
out = {}
rows = [json.loads(x) for x in open(HERE / "inputs_e1/pools_all.jsonl")]
for n, r in enumerate(rows, 1):
    pairs = [(r["question"], f"{f['title']}: {f['text']}") for f in r["fragments"]]
    out[r["question_group"]] = [float(x) for x in ce.predict(pairs, batch_size=16, show_progress_bar=False)]
    if n % 25 == 0: print(n, flush=True)
payload = {"model": MODEL, "input": "(question, 'title: text')", "max_length": 512, "device": "cpu",
           "pools_sha256": hashlib.sha256((HERE / "inputs_e1/pools_all.jsonl").read_bytes()).hexdigest(), "scores": out}
(HERE / "outputs/e3_bge_reranker_scores.json").write_text(json.dumps(payload) + "\n")
print("done")
