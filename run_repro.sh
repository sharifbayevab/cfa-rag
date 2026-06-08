#!/usr/bin/env bash
# Re-derive paper numbers from the cached generations in results/ (fully offline).
set -e
python3 -m exp.rescore            # cover-EM summaries from cached answers
python3 -m exp.analyze            # tables + figures + macros (-> article3/)
python3 -m exp.stats_ci           # bootstrap CIs (incl. cell-clustered)
python3 -m exp.ablation_threshold # tau sweep
echo "Done. Regenerated article3/*macros.tex, article3/tab_*.tex, article3/images/ from results/."
echo "Note: exp.setlevel and the run_* / opencorpus scripts call a generator (API/Ollama) and are not part of the offline repro."
