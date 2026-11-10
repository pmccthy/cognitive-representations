"""
Running and training agents.
Author: patrick.mccarthy@dpag.ox.ac.uk
"""

import pandas as pd


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
        next_obs, reward, done, truncated, next_info = env.step(action)
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


def train_rnn(agent, make_env, n_updates, batch_size=32, seed=0, log_every=100):
    """Train an RNNAgent: each update collects a batch of episodes and takes one gradient step.

    Args:
        make_env: function returning a new environment
        n_updates: number of gradient steps
        batch_size: episodes per step
    Returns:
        A table with the reward per trial and each loss term at every update.
    """
    envs = [make_env() for _ in range(batch_size)]
    history = []
    for update in range(n_updates):
        seeds = [seed + update * batch_size + i for i in range(batch_size)]  # new episodes each time
        batch = agent.collect_episodes(envs, seeds)
        losses = agent.update(batch)

        reward_per_trial = float(batch["rewards"].sum() / batch["mask"].sum())
        history.append({"update": update, "reward_per_trial": reward_per_trial, **losses})
        if log_every and (update % log_every == 0 or update == n_updates - 1):
            print(f"update {update:5d}   reward per trial {reward_per_trial:.3f}")
    return pd.DataFrame(history)
