"""
Running and training agents.
Author: patrick.mccarthy@dpag.ox.ac.uk
"""

import pandas as pd
import torch

from .agents import TrainingDiverged


def run_episode(agent, env, seed=None):
    """Run one episode and return it as a table with one row per step.

    Each row holds the observation the agent acted on, its action, the reward that action
    earned, the environment's ground truth at that step, and whatever the agent records
    (e.g. its belief or hidden state). Tabular agents learn as it runs; RNN agents don't.
    """
    obs, info = env.reset(seed=seed)
    agent.reset()
    rows = []
    done = False
    while not done:
        action = agent.act(obs)
        next_obs, reward, terminated, truncated, next_info = env.step(action)
        done = terminated or truncated
        rows.append({
            "obs": env.OBSERVATIONS[obs],
            "action": env.ACTIONS[action],
            "reward": reward,
            **info,              # ground truth when the action was chosen
            **agent.record(),    # agent's internal variables at this step
        })
        agent.observe(obs, action, reward, next_obs, done)
        obs, info = next_obs, next_info
    return pd.DataFrame(rows)


def train_rnn(agent, make_env, n_updates, batch_size=32, seed=0, log_every=100, continuous=False,
              checkpoint=None, checkpoint_every=None):
    """Train an RNNAgent: each update collects a batch of episodes and takes one gradient step.

    Args:
        make_env: function returning a new environment
        n_updates: number of gradient steps
        batch_size: episodes per step
        continuous: each update's episodes continue the previous update's sessions (same
            environments, network state carried over) instead of starting new sessions
        checkpoint: function called as checkpoint(n_done, agent) before training (n_done = 0),
            after every checkpoint_every updates, and at the end, e.g. to test or save the
            network part-way through training. It must not change the agent's training state.
    Returns:
        A table with the reward per trial, trials per episode and each loss term at every update.
        If training diverges (non-finite loss or gradients), it stops there: the table ends at
        the last good update, history.attrs["diverged_at"] is the update that failed (else
        None) and history.attrs["diverged_because"] says why. The agent keeps the last good weights.
    """
    envs = [make_env() for _ in range(batch_size)]
    outcome_ids = trial_end_ids(envs[0])
    history = []
    diverged_at, diverged_because = None, None
    if checkpoint:
        checkpoint(0, agent)
    for update in range(n_updates):
        seeds = [seed + update * batch_size + i for i in range(batch_size)]  # new episodes each time
        try:
            batch = agent.collect_episodes(envs, seeds, continue_session=continuous and update > 0)
            losses = agent.update(batch)
        except TrainingDiverged as error:
            diverged_at, diverged_because = update, str(error)
            break

        n_trials = count_trials(batch, outcome_ids)
        reward_per_trial = float(batch["rewards"].sum() / n_trials)
        # Trials per episode too: an agent stuck making invalid actions (two-step task) can earn
        # a good reward per trial while completing very few trials
        history.append({"update": update, "reward_per_trial": reward_per_trial,
                        "trials_per_episode": float(n_trials) / batch_size, **losses})
        if log_every and (update % log_every == 0 or update == n_updates - 1):
            print(f"update {update:5d}   reward per trial {reward_per_trial:.3f}")
        n_done = update + 1
        if checkpoint and ((checkpoint_every and n_done % checkpoint_every == 0) or n_done == n_updates):
            checkpoint(n_done, agent)
    history = pd.DataFrame(history)
    history.attrs.update(diverged_at=diverged_at, diverged_because=diverged_because)
    return history


def trial_end_ids(env):
    """Integer ids of the observations that end a trial, if the env lists them (env.OUTCOMES)."""
    return torch.tensor([env.OBSERVATIONS.index(o) for o in getattr(env, "OUTCOMES", [])])


def count_trials(batch, outcome_ids):
    """Number of completed trials in a batch.

    With no outcome list (TwoAFC), each step where the action counts is one trial. Otherwise
    (TwoStepTask, where trials vary in length) a trial ends at each step that moves into an
    outcome observation from a non-outcome one.
    """
    if len(outcome_ids) == 0:
        return batch["mask"].sum()
    ends = torch.isin(batch["next_obs"], outcome_ids) & ~torch.isin(batch["obs"], outcome_ids)
    return ends.sum().clamp(min=1)
