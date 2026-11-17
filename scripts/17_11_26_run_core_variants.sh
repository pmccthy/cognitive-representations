#!/bin/bash
# Train the core variants (CORE_VARIANTS in cogreps/variants.py) and the feedforward
# ff_history control (1 and 2 trials of history) with 50 seeds on the 2AFC reversal task at
# several reward probabilities, with the Bayes and Q-learning baselines, then plot each
# condition as it finishes.
#
# Run from the repo root, in the cogtasks environment:
#     bash scripts/17_11_26_run_core_variants.sh 2>&1 | tee results/run_core_variants.log
#
# Resumable: if it's interrupted, run the same command again and finished agents are skipped.
# caffeinate stops the Mac sleeping while it runs (keep it plugged in).

set -e  # stop at the first error

SEEDS=$(seq 0 49)
CORE=$(python -c "from cogreps.variants import CORE_VARIANTS; print(' '.join(CORE_VARIANTS))")

for P_REWARD in 1.0 0.8 0.6 0.5; do
    OUT="results/2afc_core_p${P_REWARD/./}"   # results/2afc_core_p10, _p08, _p06, _p05
    echo "=== p_reward = $P_REWARD -> $OUT"
    caffeinate -i python scripts/10_11_26_train_2afc_variants.py \
        --variants $CORE ff_history --n-back 1 2 \
        --seeds $SEEDS --p-reward "$P_REWARD" --out "$OUT"
    python scripts/10_11_26_plot_2afc_variants.py --results "$OUT"
done
