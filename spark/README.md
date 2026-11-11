# Fresh downstream evidence on NVIDIA DGX Spark

`build_selection_plan.py` reconstructs the leakage-free five-fold models and
creates one selection plan per unique question. All comparison selectors retain
the same number of passages as the OOF all-feature selector.

`run_fresh_qa.py` evaluates a held-out open reader on full context, the OOF
selector, dense relevance, cross-encoder relevance, a CE-rank-plus-length
predictor, and ContextCite. The ContextCite implementation follows the official
algorithm and code: 32 Bernoulli-0.5 masks, fixed-response token probability,
sequence logit transform, standardized Lasso with alpha 0.01, and attribution
ranking. Short answers are treated as a single response statement. The primary
comparison matches every baseline to the OOF selector's per-question budget; a
fixed top-four sensitivity analysis is generated from the same ContextCite
probes.

The run checkpoints every question to JSONL and can resume after interruption.
The final summary uses a dataset-stratified paired bootstrap over questions.

The Spark run uses the pinned model revision recorded in the output metadata and
the NVIDIA PyTorch 25.09 container. After the first download, `resume_full.sh`
runs Hugging Face in strict offline mode, removes only the named experiment
container, and resumes `results/fresh_qa.jsonl`. The ContextCite forward pass
applies Qwen's language-model head only to the final response states; this
returns the same response-token logits without materializing unused logits for
every prompt token.

After all 450 rows are present, validate and summarize them with:

```bash
python analyze_results.py \
  --batch16-rows 232 \
  --full-logit-rows 249 \
  --cache-disabled-start 306 \
  --cache-disabled-end 316 \
  --out-dir results
```

The analyzer rejects malformed or duplicate rows, mismatched frozen questions,
changed passage selections or budgets, and deviations from the 32-mask
ContextCite protocol. It writes the complete JSON analysis, paired bootstrap
contrasts, LaTeX table rows, and manuscript macros.

`REVISION_PROTOCOL.md` freezes the compact reviewer-response experiments.
`run_intervention_controls.py` compares deletion against an exact
token-position-preserving mask and jointly intervenes on the annotated support
set. `analyze_semantic_labels.py` repeats the label analysis with a pinned BGE
embedding-equivalence rule. `build_efficiency_plan.py` creates inner-CV
thresholds and frozen method rankings; `run_efficiency_sweep.py` evaluates
these rankings at equal 256/512/768-token budgets on all 450 questions.
