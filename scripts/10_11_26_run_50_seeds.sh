#!/bin/bash
# Train every variant (including the ff_history windows) with 50 seeds, on the 2AFC
# reversal task with certain (p_reward = 1) and probabilistic (p_reward = 0.8) rewards,
# then plot both.
#
# Run from the repo root, in the cogtasks environment:
#     bash scripts/10_11_26_run_50_seeds.sh 2>&1 | tee results/run_50_seeds.log
#
# Resumable: if it's interrupted, run the same command again and finished agents are skipped.
# caffeinate stops the Mac sleeping while it runs (keep it plugged in).

set -e  # stop at the first error

SEEDS=$(seq 0 49)

for P_REWARD in 1.0 0.8; do
    OUT="results/2afc_50seeds_p${P_REWARD/./}"   # results/2afc_50seeds_p10 and ..._p08
    echo "=== p_reward = $P_REWARD -> $OUT"
    caffeinate -i python scripts/10_11_26_train_2afc_variants.py \
        --seeds $SEEDS --p-reward "$P_REWARD" --out "$OUT"
    python scripts/10_11_26_plot_2afc_variants.py --results "$OUT"
done
