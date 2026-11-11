# Frozen revision experiments

This protocol was fixed before inspecting the downstream outcomes.

## Position and interaction controls

- Reader: `Qwen/Qwen3-4B-Instruct-2507`, the same pinned fresh reader used in
  the main downstream experiment.
- Sample: 40 questions per benchmark (120 total), selected by sorting each
  benchmark's frozen question identifiers by SHA-256 and taking the first 40.
- Single-fragment conditions: full prompt, deletion of each fragment with the
  original numeric IDs retained, and replacement of the exact title-and-text
  token span by one repeated neutral token ID.
- The mask condition has exactly the same prompt-token length as the full
  prompt and leaves all later token positions unchanged.
- Interaction condition: delete or token-mask the complete annotated
  supporting set jointly.
- Primary label: normalized answer change, matching the paper's CFA label.
- Semantic sensitivity analysis: BGE-base-en-v1.5 cosine equivalence at the
  fixed thresholds 0.80, 0.85, and 0.90.

## Selection efficiency controls

- Questions: all 450 frozen questions (150 per benchmark).
- Token budgets: 256, 512, and 768 tokenizer tokens in the rendered context.
- Methods: OOF CFA, dense relevance, cross-encoder relevance, CE-rank plus
  length, and the already-computed 32-mask ContextCite ranking.
- Selection: scan each method's ranking, retain a candidate if the resulting
  context remains within budget, then restore source order for the reader.
- Threshold curve: OOF thresholds 0.1, 0.2, 0.3, 0.4, and 0.5.
- Calibrated threshold: for every outer question-held-out fold, select the
  threshold that maximizes passage-label F1 on predictions from a four-fold
  inner question-held-out split. Test answers are never used for calibration.
- Inference: greedy decoding, 48-token output cap, stable passage IDs.
- Statistics: dataset-stratified paired question bootstrap, 10,000 draws.

The generated JSONL files are append-only checkpoints. Analysis scripts reject
duplicate question identifiers and preserve the raw per-question outcomes.
