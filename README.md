# Counterfactual Fragment Attribution (CFA) — experiments

Code and cached results for measuring the **causal contribution** of each
retrieved fragment in retrieval-augmented generation (RAG), by black-box
leave-one-out ablation: remove a fragment, regenerate, and see whether the
answer changes.

## Contents

```
exp/        # experiment code (data, generators, attribution/CFA, methods, selector, analysis, stats)
results/    # cached generations (raw_*.jsonl), per-cell summaries, methods / selector /
            # set-level / open-corpus outputs, trained selector
requirements.txt
run_repro.sh
```

## Setup
```bash
pip install -r requirements.txt
```

## Reproduce the reported numbers (offline, from cached generations)
```bash
bash run_repro.sh        # rescore -> analyze -> stats_ci -> ablation
```
Re-derives all aggregate numbers (tables, macros, figures) from `results/`.

## Regenerate from scratch
Needs an OpenAI key in `.env` and/or a local [Ollama](https://ollama.com) server:
```bash
# distractor-pool cells
python -m exp.run_experiment --dataset hotpotqa --generator openai:gpt-4.1-mini --n 200 --no-judge
# open-corpus retrieve-then-read
python -m exp.opencorpus     --dataset hotpotqa --generator openai:gpt-4.1-mini --pool-n 1000 --eval-n 150 --k 10
# method comparison (CFA pruning, amortized, ContextCite, ...)
python -m exp.run_methods    --dataset hotpotqa --generator ollama:gemma3:4b --n 50
```

## Metric note
Correctness is **cover-EM** (the gold answer string appears in the prediction),
computed locally by `exp/metrics.py`. **No LLM judge** is used for any reported
number. Confidence intervals use a bootstrap (`exp/stats_ci.py`), including a
cell-clustered variant.

## License
Code released under the [MIT License](LICENSE). Datasets (HotpotQA,
2WikiMultiHopQA, MuSiQue) are loaded via 🤗 `datasets` under their respective licenses.
