"""
Train every RNN variant on the 2AFC reversal task and summarise how well each learns.
Author: patrick.mccarthy@dpag.ox.ac.uk

Run from the repo root, in the cogtasks environment:
    python scripts/train_2afc_variants.py
    python scripts/train_2afc_variants.py --n-updates 5000 --seeds 0 1 2
    python scripts/train_2afc_variants.py --p-reward 0.8 --out results/2afc_variants_p0.8

Variants train in parallel, one per CPU core. Prints a summary table and saves, in --out:
    task.json         the task settings used
    summary.csv       one row per agent (variant x seed), plus the Bayes and Q-learning baselines
    history.csv       reward and loss terms at every training update
    test_trials.csv   every trial of the frozen-weight test sessions, for every agent
    models/           trained weights, one file per agent
Plot the results with scripts/plot_2afc_variants.py.
"""

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from cogreps.agents import BayesAgent, QLearner
from cogreps.tasks import TwoAFC
from cogreps.training import run_episode, train_rnn
from cogreps.variants import VARIANTS, make_agent

# Chance is 50% correct choices. An agent counts as having learned once its reward per trial,
# smoothed over 50 updates, passes what 60% correct would earn (0.6 x p_reward)
LEARNED_LEVEL = 0.6


# ---------------------------------------------------------------------- measures

def inference_trials(trials):
    """Inference signature: after a reversal, once the agent has met it on one stimulus,
    is its first choice for the *other* stimulus correct? (True/False per reversal)

    An agent that infers the rule switches both stimuli after one error (Bayes: ~1.0).
    One that learns each stimulus separately keeps the old answer (Q-learning: low).
    """
    results = []
    for i in range(1, len(trials)):
        if trials.rule[i] == trials.rule[i - 1]:
            continue  # no reversal before trial i
        for j in range(i + 1, min(i + 15, len(trials))):
            if trials.rule[j] != trials.rule[i]:
                break  # another reversal came first
            if trials.stimulus[j] != trials.stimulus[i]:
                results.append(trials.action[j] == trials.correct_action[j])
                break
    return results


def evaluate(agent, task, n_episodes=20, seed=10_000):
    """Run the agent as it is (RNN weights frozen) on long test sessions of the same task.

    Returns reward per trial, fraction of correct choices, the inference score,
    and a table of every test trial.
    """
    env = TwoAFC(**{**task, "n_trials": 500})
    sessions = []
    for episode in range(n_episodes):
        steps = run_episode(agent, env, seed=seed + episode)
        trials = steps[steps.phase == "STIM"].reset_index(drop=True)
        trials = trials[["trial", "stimulus", "rule", "action", "correct_action", "reward"]].copy()
        trials["correct"] = trials.action == trials.correct_action
        trials["episode"] = episode
        sessions.append(trials)
    inferred = [x for t in sessions for x in inference_trials(t)]
    all_trials = pd.concat(sessions, ignore_index=True)
    return (float(all_trials.reward.mean()), float(all_trials.correct.mean()),
            float(np.mean(inferred)), all_trials)


def updates_to_learn(history, p_reward, window=50):
    """First update at which reward per trial (averaged over `window` updates) passes the
    learned level (see LEARNED_LEVEL)."""
    smoothed = history.reward_per_trial.rolling(window).mean()
    above = np.flatnonzero(smoothed.to_numpy() > LEARNED_LEVEL * p_reward)
    return int(above[0]) if len(above) else np.nan


# ---------------------------------------------------------------------- one training run

def train_one(variant, seed, task, n_updates, batch_size, lr, out):
    torch.set_num_threads(1)  # runs are parallel across processes instead
    start = time.time()
    make_env = partial(TwoAFC, **task)
    agent = make_agent(variant, make_env(), seed=seed, lr=lr)
    history = train_rnn(agent, make_env, n_updates, batch_size, seed=seed * 1_000_000, log_every=0)
    torch.save(agent.model.state_dict(), out / "models" / f"{variant}_seed{seed}.pt")

    reward, correct, inference, trials = evaluate(agent, task)
    summary = {
        "agent": variant,
        "seed": seed,
        "updates_to_learn": updates_to_learn(history, task["p_reward"]),
        "train_reward_last100": history.reward_per_trial.tail(100).mean(),
        "test_reward": reward,
        "test_correct": correct,
        "test_inference": inference,
        "minutes": (time.time() - start) / 60,
    }
    for table in (history, trials):
        table["agent"], table["seed"] = variant, seed
    return summary, history, trials


# ---------------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--n-updates", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--n-trials", type=int, default=50, help="trials per training episode")
    parser.add_argument("--p-reversal", type=float, default=0.05, help="reversal probability per trial")
    parser.add_argument("--p-reward", type=float, default=1.0, help="reward probability for correct choices")
    parser.add_argument("--out", default="results/2afc_variants")
    args = parser.parse_args()
    out = Path(args.out)
    (out / "models").mkdir(parents=True, exist_ok=True)

    # The task every agent is trained and tested on
    task = dict(n_trials=args.n_trials, rule=None, p_reversal=args.p_reversal, p_reward=args.p_reward)
    (out / "task.json").write_text(json.dumps(task, indent=2))

    # Baselines: the ideal observer, and Q-learning (learning as it goes over the test sessions)
    env = TwoAFC(**task)
    rows, histories, all_trials = [], [], []
    for name, agent in [("bayes (ideal)", BayesAgent(env, seed=0)),
                        ("q_learning", QLearner(env, alpha=0.3, beta=10, seed=0))]:
        reward, correct, inference, trials = evaluate(agent, task)
        rows.append({"agent": name, "test_reward": reward, "test_correct": correct,
                     "test_inference": inference})
        trials["agent"] = name
        all_trials.append(trials)

    # RNN variants, in parallel
    jobs = [(v, s) for v in args.variants for s in args.seeds]
    print(f"Training {len(jobs)} agents ({args.n_updates} updates each) on {args.workers} workers...")
    with ProcessPoolExecutor(max_workers=min(args.workers, len(jobs))) as pool:
        futures = [pool.submit(train_one, v, s, task, args.n_updates, args.batch_size, args.lr, out)
                   for v, s in jobs]
        for future in as_completed(futures):
            summary, history, trials = future.result()
            print(f"  done: {summary['agent']} (seed {summary['seed']}), "
                  f"test reward {summary['test_reward']:.3f}, {summary['minutes']:.1f} min")
            rows.append(summary)
            histories.append(history)
            all_trials.append(trials)

    # Save and print
    columns = ["agent", "seed", "updates_to_learn", "train_reward_last100",
               "test_reward", "test_correct", "test_inference", "minutes"]
    table = pd.DataFrame(rows, columns=columns)
    table.to_csv(out / "summary.csv", index=False)
    pd.concat(histories).to_csv(out / "history.csv", index=False)
    pd.concat(all_trials).to_csv(out / "test_trials.csv", index=False)

    print(f"\nTask: {task}")
    print(f"Training: {args.n_updates} updates x {args.batch_size} episodes, lr {args.lr}")
    print(f"updates_to_learn: first update with smoothed reward > {LEARNED_LEVEL * task['p_reward']:.2f}"
          f" (= {LEARNED_LEVEL:.0%} correct; chance is 50%)\n")
    print(table.round(3).to_string(index=False))

    # Mean and spread across seeds, for the RNN variants
    rnn = table.dropna(subset=["seed"])
    if rnn.seed.nunique() > 1:
        stats = rnn.groupby("agent")[["updates_to_learn", "test_correct", "test_inference"]]
        print("\nAcross seeds (mean ± sd; updates_to_learn counts only seeds that learned):")
        print(stats.agg(["mean", "std"]).round(3).to_string())
        print(rnn.groupby("agent").updates_to_learn.apply(lambda x: f"{x.notna().sum()}/{len(x)} seeds learned").to_string())
    print(f"\nSaved to {out}/  (plot with: python scripts/plot_2afc_variants.py)")


if __name__ == "__main__":
    main()
