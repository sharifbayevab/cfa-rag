# CFA analysis package

This directory reproduces the numbers, tables and figures reported in the paper.
It operates offline on frozen copies of the experiment caches. Scripts that call
reader APIs read keys from a local `.env` file, which is not part of the repository.

Run from the repository root:

```bash
python3 repro/rebuild.py
python3 repro/selector_oof.py
python3 repro/selector_diagnostics.py
python3 repro/render_revision_outputs.py
python3 -m unittest discover -s repro -p 'test_*.py'
```

Dependencies: Python 3, NumPy, scikit-learn, Matplotlib. Installed dependency
versions are recorded in `outputs/environment.json`. Table bodies (`tab_*.tex`)
and figures (`images/`) are written to the working directory.

## Inputs and provenance

`input_manifest.json` records SHA-256 hashes for all frozen inputs. `inputs/`
contains the original 15-cell grid (150 questions/cell), three retrieved-pool
cells, the historical six-cell method aggregates, four available per-question
method caches, the cached feature matrix, repeatability checks, the 120-question
intervention-control run, the 450-question equal-token sweep and the frozen
semantic-sensitivity analysis.

The provenance paths identify where these files were read; reproduction uses
only local `inputs/`, not those absolute source paths. No pickled estimator is
loaded. The original generators are not called.

`make_diagrams.py` regenerates the two vector overview figures from manuscript
constants: the black-box deletion procedure and the exact-to-OOF system
architecture. `rebuild.py` invokes it automatically.

## Corrected estimands

- There are 450 unique questions, not 2250 independent questions.
- Bootstrap sampling is stratified by dataset and clusters each question's
  full five-generator block. The five generators are fixed.
- Relevance Jaccard includes all rows and is compared with an exact
  hypergeometric random-set expectation conditional on `(m, |S|, |R|)`.
- `Par` is closed-book answerability conditional on a correct attribution
  baseline, not proof of exclusive parametric generation.
- The main pruning comparison uses paired selection records and covers all
  2250 question–generator observations. Its full-context reference was generated
  separately from the attribution baseline. The open-corpus table uses the
  same convention, eliminating the duplicate incompatible Full columns.
- Context length is a whitespace word count. `m*W + W_selected` sums the full,
  leave-one-out, and final context inputs. It is not a bill or latency estimate.
- The historical `contextcite` cache key denotes the **answer-agreement
  Mask-surrogate**: ridge regression on F1 between freshly generated answers.
  It is not the original ContextCite algorithm, which uses fixed-answer
  probabilities and Lasso. Legacy keys are preserved only for data provenance.

## Learned selector and fresh downstream evaluation

`selector_oof.py` reconstructs question/dataset/generator/fragment alignment
from the raw grid, checks every label and available ID against the feature
matrix, and computes five-fold **question-held-out** predictions.
All copies of a question across generators remain in the same fold.
Cross-generator results additionally exclude the target generator from training.
The seed is fixed to zero. The code never saves or evaluates a global
in-sample checkpoint as if it were held out.

Outputs:

- `outputs/selector_folds.json`: complete training and test question IDs.
- `outputs/selector_oof_selections.jsonl`: probabilities and selected passage
  indices for each held-out question–generator pair, with fold IDs.
- `outputs/selector_metrics.json`: grouped AUC/AP, feature baselines, transfer
  results, and a checksum of the selection manifest.
- `tab_predictor.tex` and `selmacros.tex`: only independently evaluated
  classification metrics.

The probability threshold is fixed at 0.5; empty selections fall back to the
highest predicted probability. The saved rows explicitly have
`qa_status="not_generated"`; those historical OOF rows remain label-prediction
records rather than downstream QA results. The separate fresh experiment freezes
one selection per unique question and generates new answers with the previously
unseen `Qwen/Qwen3-4B-Instruct-2507` reader.
The historical `amortized_causal` rows are deliberately excluded from the
method table and plotted comparison, because their available evaluation path
reused training questions. They remain in immutable inputs for auditing.

The fresh run compares the frozen OOF selector against equal-budget dense,
cross-encoder, CE-rank-plus-length, and original ContextCite selection. A second
panel fixes every selector at four passages. ContextCite uses 32 deterministic
Bernoulli masks, fixed-response sequence probability, a logit transform, and
standardized Lasso with alpha 0.01. The raw answers, selected indices, model
revision, environment versions, and paired question-bootstrap analysis are
retained in `inputs/fresh_qa_qwen3_4b_n450.jsonl`,
`inputs/fresh_selection_plan.jsonl`, and `outputs/fresh_qa_analysis.json`.

## Revision controls and diagnostics

`inputs/intervention_controls_qwen3_4b_n120.jsonl` contains a prespecified,
SHA-256-sorted sample of 40 questions per dataset. It compares stable-ID
deletion with an exact-token-length neutral mask, and also removes or masks the
complete annotated supporting set. `analyze_intervention_controls.py` reports
question-clustered confidence intervals.

`inputs/efficiency_sweep_qwen3_4b_n450.jsonl` evaluates all selectors at 256,
512 and 768 context-token budgets and evaluates OOF probability thresholds.
The calibrated operating point is selected only within the training portion of
each outer question-held-out fold. `analyze_efficiency_sweep.py` validates the
frozen plan and computes paired question-bootstrap contrasts.

`inputs/semantic_label_validation.json` records the sensitivity analysis over
3,436 deletion/mask answer pairs using pinned `BAAI/bge-base-en-v1.5`
embeddings. `selector_diagnostics.py` produces paired confidence intervals,
Brier/ECE calibration results and leave-one-feature-out ablations.
`render_revision_outputs.py` regenerates the revision macros and the two
publication figures: budget/calibration curves and intervention-control
confidence intervals.

## Remaining historical limitations

Two of the six method cells have only aggregate caches in the supplied inputs;
the method table therefore remains descriptive, without new significance or
equivalence claims. Individual counterfactual answer strings and complete
serving snapshots were not retained in the original primary grid. No code can
recover those missing observations from the aggregate sensitivity labels.
The paper explicitly distinguishes these limitations from the
quantities that can be reproduced offline. The fresh downstream and control
experiments still use one reader and supplied candidate pools. Semantic-label
validation is embedding-based rather than human-annotated. The results therefore
do not establish cross-reader or open-corpus selection superiority.

## Round 2 (2026-11): answer-containment baseline, fresh API probes, cost

Scripts (all new; round-1 scripts and outputs are unchanged):

| Script | What it does | Output |
|---|---|---|
| `e6_lib.py` | containment features and the shared round-2 ranking definitions | |
| `e6_offline.py` | E6a within-gold AUC of containment baselines; E6b strata, question-held-out logistic models, bridge-passage analysis; E9 leave-family-out transfer (no generator calls) | `outputs/e6_offline.json` |
| `e6_build_plan.py` | frozen ranking plans for the Spark runs | `inputs_e6/e6_plan.jsonl`, `inputs_e6/e6_plan_donors.jsonl` |
| `e1_spark/sweep.py --plan e6_plan.jsonl` | Qwen3-4B answers (Spark, same stack as round 1; every answer of a run, including its references and the full context, is generated in that run) | `inputs_e6/e6_sweep.jsonl`, `inputs_e6/e6_sweep_donors.jsonl` |
| `e7_reprobe.py` | fresh leave-one-out probes of GPT-4.1-mini and Gemini-3.5-flash-lite; `--repeat` measures the repeat-call noise floor | `inputs_e6/e7_loo_{gpt,gemini}.jsonl`, `outputs/e7_*_cache.jsonl`, `outputs/e7_*_repeat.json` |
| `e6_api_sweep.py` | containment arms and fresh exact-CFA arms on the API readers (reuses the E4 cache of the same reader) | `outputs/e6_e4_*_answers.jsonl`, `outputs/e6_e4_*_cache.jsonl` |
| `e6_analyze.py qwen|qwen_donors|gpt|gemini|qwen_r1` | contrasts, Holm families, full-context family, containment strata | `outputs/e6_*_analysis.json` (+ `.txt`) |
| `e8_cost.py` | calls per question against cover-EM, Pareto set | `outputs/e8_cost.json` |
| `make_e6_figures.py` | `images/e6_*.pdf`, `tab_e6_*.tex` | |

Containment: the normalized answer string, padded with spaces, is a substring of the
normalized passage (title + text), with `cfa_lib.normalize`.  "Own answer" is the
target reader's full-context answer (one call).  SHA-256 of the round-2 inputs:
`inputs_e6/manifest.json`.

```bash
python3 repro/e6_offline.py
python3 repro/e6_build_plan.py            # then Spark: sweep.py --plan e6_plan.jsonl / e6_plan_donors.jsonl
python3 repro/e7_reprobe.py --reader gpt ; python3 repro/e7_reprobe.py --reader gemini
python3 repro/e7_reprobe.py --reader gpt --repeat ; python3 repro/e7_reprobe.py --reader gemini --repeat
python3 repro/e6_api_sweep.py --reader gpt ; python3 repro/e6_api_sweep.py --reader gemini
for r in qwen qwen_donors gpt gemini qwen_r1; do python3 repro/e6_analyze.py $r; done
python3 repro/e8_cost.py ; python3 repro/make_e6_figures.py
```
