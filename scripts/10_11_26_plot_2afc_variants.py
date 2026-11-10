"""
Plot the results of scripts/10_11_26_train_2afc_variants.py.
Author: patrick.mccarthy@dpag.ox.ac.uk

Run from the repo root:
    python scripts/10_11_26_plot_2afc_variants.py
    python scripts/10_11_26_plot_2afc_variants.py --results results/2afc_variants

Saves to <results>/figures/:
    learning_curves.png   reward per trial over training, for each RNN variant (mean over seeds)
    learning_curves_by_seed.png  the same, one panel per variant with every seed shown
    test_summary.png      frozen-weight test performance of every agent and seed
    reversals.png         correct choices and reward around reversals (frozen-weight test)
    reward_over_session.png  cumulative reward over a test session
    example_sessions.png  trial-by-trial behaviour of each agent in one test session
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cogreps.plotting import smooth, use_style
from cogreps.variants import VARIANTS

BASELINES = ["bayes (ideal)", "q_learning"]
BASELINE_STYLE = {"bayes (ideal)": dict(color="black", ls="--"),
                  "q_learning": dict(color="grey", ls="--")}


def agent_order(names):
    """Baselines first, then variants in the order they're defined in variants.py."""
    known = BASELINES + list(VARIANTS)
    return [n for n in known if n in names] + sorted(set(names) - set(known))


def colours(agents):
    """Fixed colour per agent, so every figure matches."""
    cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    variants = [a for a in agents if a not in BASELINES]
    styles = {a: dict(color=cycle[i % len(cycle)], ls="-") for i, a in enumerate(variants)}
    return {**BASELINE_STYLE, **styles}


# ---------------------------------------------------------------------- figures

def plot_learning_curves(history, summary, style, path, p_reward, window=50):
    """Reward per trial at each training update (mean over seeds), with test baselines."""
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for agent in agent_order(history.agent.unique()):
        curve = history[history.agent == agent].groupby("update").reward_per_trial.mean()
        ax.plot(curve.index, smooth(curve.values, min(window, len(curve))), label=agent, **style[agent])
    for agent in BASELINES:
        reward = summary.loc[summary.agent == agent, "test_reward"]
        if len(reward):
            ax.axhline(reward.iloc[0], label=agent, lw=1.5, **style[agent])
    ax.axhline(0.5 * p_reward, color="lightgrey", lw=1, zorder=0)  # chance
    ax.set(xlabel="Training update", ylabel="Reward per trial", ylim=(0.5 * p_reward - 0.1, 1.0))
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.savefig(path)
    plt.close(fig)


def plot_learning_curves_by_seed(history, summary, style, path, p_reward, window=50):
    """One panel per variant: each seed as a thin line, the mean across seeds in bold."""
    agents = agent_order(history.agent.unique())
    n_cols = 3
    n_rows = int(np.ceil(len(agents) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.5 * n_cols, 4 * n_rows), sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()
    for ax, agent in zip(axes, agents):
        data = history[history.agent == agent]
        for _, run in data.groupby("seed"):
            run = run.sort_values("update")
            ax.plot(run["update"], smooth(run.reward_per_trial.to_numpy(), min(window, len(run))),
                    color=style[agent]["color"], lw=1, alpha=min(0.6, 5 / data.seed.nunique()))
        mean = data.groupby("update").reward_per_trial.mean()
        ax.plot(mean.index, smooth(mean.values, min(window, len(mean))), color=style[agent]["color"], lw=3)
        for baseline in BASELINES:  # test performance of the baselines, for reference
            reward = summary.loc[summary.agent == baseline, "test_reward"]
            if len(reward):
                ax.axhline(reward.iloc[0], lw=1.2, **style[baseline])
        ax.axhline(0.5 * p_reward, color="lightgrey", lw=1, zorder=0)  # chance
        ax.set_title(f"{agent} ({data.seed.nunique()} seeds)", fontsize=14)
        ax.set_ylim(0.5 * p_reward - 0.1, 1.0)
    for ax in axes[len(agents):]:
        ax.axis("off")
    fig.supxlabel("Training update")
    fig.supylabel("Reward per trial")
    fig.savefig(path)
    plt.close(fig)


def plot_test_summary(summary, style, path):
    """Frozen-weight test performance: one dot per seed, a bar at the mean, baselines as lines."""
    rnn = summary.dropna(subset=["seed"])
    agents = agent_order(rnn.agent.unique())
    measures = [m for m in ["test_correct", "test_inference"] if m in summary]
    labels = {"test_correct": "P(correct choice)", "test_inference": "Inference score"}
    fig, axes = plt.subplots(1, len(measures), figsize=(7 * len(measures), 5.5))
    rng = np.random.default_rng(0)
    for ax, measure in zip(np.atleast_1d(axes), measures):
        for x, agent in enumerate(agents):
            values = rnn.loc[rnn.agent == agent, measure].to_numpy()
            jitter = rng.uniform(-0.12, 0.12, len(values))
            ax.scatter(x + jitter, values, color=style[agent]["color"], s=40, zorder=3)
            ax.plot([x - 0.3, x + 0.3], [values.mean()] * 2, color=style[agent]["color"], lw=3)
        for baseline in BASELINES:
            value = summary.loc[summary.agent == baseline, measure]
            if len(value):
                ax.axhline(value.iloc[0], lw=1.5, label=baseline, **style[baseline])
        ax.axhline(0.5, color="lightgrey", lw=1, zorder=0)  # chance
        ax.set_xticks(range(len(agents)), agents, rotation=40, ha="right", fontsize=13)
        ax.set(ylabel=labels[measure], ylim=(0.4, 1.02))
    np.atleast_1d(axes)[-1].legend(loc="lower right")
    fig.savefig(path)
    plt.close(fig)


def reversal_windows(trials, before=5, after=15):
    """Trials around each reversal, as rows of [before + after] values per variable.

    A reversal at trial i means the rule changed between trials i-1 and i. Reversals with
    another reversal inside the window are skipped, so each window shows one switch.
    """
    windows = {"correct": [], "reward": []}
    for _, session in trials.groupby(["seed", "episode"]):
        rule = session.rule.to_numpy()
        switches = np.flatnonzero(np.diff(rule)) + 1
        for i in switches:
            if i - before < 0 or i + after > len(rule):
                continue
            if np.count_nonzero(np.diff(rule[i - before:i + after])) > 1:
                continue  # another reversal inside the window
            for key in windows:
                windows[key].append(session[key].to_numpy(dtype=float)[i - before:i + after])
    return {k: np.array(v) for k, v in windows.items()}


def plot_reversals(trials, style, path, p_reward, before=5, after=15):
    """Probability of a correct choice, and of reward, on trials around a reversal."""
    x = np.arange(-before, after)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)
    for agent in agent_order(trials.agent.unique()):
        windows = reversal_windows(trials[trials.agent == agent], before, after)
        if not len(windows["correct"]):
            continue
        for ax, key in zip(axes, ["correct", "reward"]):
            mean = windows[key].mean(0)
            sem = windows[key].std(0) / np.sqrt(len(windows[key]))
            ax.plot(x, mean, label=agent, **style[agent])
            ax.fill_between(x, mean - sem, mean + sem, color=style[agent]["color"], alpha=0.15, lw=0)
    for ax, ylabel, chance in zip(axes, ["P(correct choice)", "P(reward)"], [0.5, 0.5 * p_reward]):
        ax.axvline(-0.5, color="lightgrey", lw=1, zorder=0)  # reversal happens here
        ax.axhline(chance, color="lightgrey", lw=1, zorder=0)
        ax.set(xlabel="Trials from reversal", ylabel=ylabel, ylim=(0, 1.02))
    axes[1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.savefig(path)
    plt.close(fig)


def plot_reward_over_session(trials, style, path, p_reward):
    """Cumulative reward over a test session, averaged over sessions (the ideal agent sets the ceiling)."""
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for agent in agent_order(trials.agent.unique()):
        data = trials[trials.agent == agent]
        per_session = data.pivot_table(index="trial", columns=["seed", "episode"], values="reward")
        cumulative = per_session.cumsum().mean(axis=1)
        ax.plot(cumulative.index + 1, cumulative.values, label=agent, **style[agent])
    n = trials.trial.max() + 1
    ax.plot([0, n], [0, n * 0.5 * p_reward], color="lightgrey", lw=1, zorder=0)  # chance
    ax.set(xlabel="Trial in session", ylabel="Cumulative reward")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.savefig(path)
    plt.close(fig)


def plot_example_sessions(trials, style, path, n_trials=150, episode=0, window=5):
    """One test session per agent: correct choices (smoothed), rewarded trials, and reversals."""
    agents = agent_order(trials.agent.unique())
    fig, axes = plt.subplots(len(agents), 1, figsize=(14, 1.9 * len(agents)), sharex=True)
    for ax, agent in zip(np.atleast_1d(axes), agents):
        data = trials[(trials.agent == agent) & (trials.episode == episode)]
        data = data[data.seed == data.seed.min()]  # first seed only
        data = data.sort_values("trial").head(n_trials)
        t = data.trial.to_numpy()

        # Reversals (rule changes) as vertical lines
        for i in np.flatnonzero(np.diff(data.rule.to_numpy())) + 1:
            ax.axvline(t[i] - 0.5, color="lightgrey", lw=1.5, zorder=0)
        # Rewarded trials as ticks along the top, correct choices smoothed below
        rewarded = t[data.reward.to_numpy() > 0]
        ax.plot(rewarded, np.full(len(rewarded), 1.12), "|", color=style[agent]["color"], ms=6)
        ax.plot(t, smooth(data.correct.to_numpy(dtype=float), window), color=style[agent]["color"])
        ax.set(ylim=(-0.05, 1.25), yticks=[0, 1])
        ax.set_title(agent, loc="left", fontsize=12, pad=2)
    np.atleast_1d(axes)[-1].set_xlabel("Trial in session")
    fig.supylabel("P(correct), smoothed")
    fig.savefig(path)
    plt.close(fig)


# ---------------------------------------------------------------------- main

def first_existing(folder, *names):
    """The first of these files that exists in the folder (None if none do)."""
    for name in names:
        if (folder / name).exists():
            return folder / name
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", default="results/2afc_variants")
    args = parser.parse_args()
    results = Path(args.results)
    figures = results / "figures"
    figures.mkdir(exist_ok=True)

    use_style()
    summary = pd.read_csv(results / "summary.csv")
    history = pd.read_csv(first_existing(results, "history.csv.gz", "history.csv"))
    style = colours(agent_order(summary.agent.unique()))
    task_path = results / "task.json"
    p_reward = json.loads(task_path.read_text())["p_reward"] if task_path.exists() else 1.0

    plot_learning_curves(history, summary, style, figures / "learning_curves.png", p_reward)
    plot_learning_curves_by_seed(history, summary, style, figures / "learning_curves_by_seed.png", p_reward)
    plot_test_summary(summary, style, figures / "test_summary.png")
    trials_path = first_existing(results, "test_trials.csv.gz", "test_trials.csv")
    if trials_path is not None:
        trials = pd.read_csv(trials_path)
        trials["seed"] = trials.seed.fillna(0)  # baselines have no seed
        plot_reversals(trials, style, figures / "reversals.png", p_reward)
        plot_reward_over_session(trials, style, figures / "reward_over_session.png", p_reward)
        plot_example_sessions(trials, style, figures / "example_sessions.png")
    else:
        print("No test_trials.csv (rerun train_2afc_variants.py); only learning curves plotted.")
    print(f"Figures saved to {figures}/")


if __name__ == "__main__":
    main()
