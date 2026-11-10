"""
Named network variants.
Author: patrick.mccarthy@dpag.ox.ac.uk

Each variant is a set of settings for RNNAgent, not a class:
    model:   ModelConfig settings (readout, gradient flow, prediction heads)
    weights: how much each loss term counts (terms left out are off)
Add a variant by adding an entry; anything can be overridden when making an agent.
"""

from .agents import RNNAgent

# Reinforcement-learning terms shared by every variant. Wang et al. 2018 used critic = 0.05;
# 0.5 learned the 2AFC reversal task much faster in our tests.
RL = {"actor": 1.0, "critic": 0.5, "entropy": 0.05}

VARIANTS = {
    # End-to-end meta-RL: linear policy/value straight off the RNN (Wang et al. 2018)
    "meta_rl": dict(
        model=dict(readout_hidden=None),
        weights=RL,
    ),
    # End-to-end RL through a feedforward ("basal ganglia") readout layer
    "rl_readout": dict(
        model=dict(readout_hidden=64, readout_obs=True),
        weights=RL,
    ),
    # RNN trained only to predict the next observation; RL trains the readout alone
    # (Blanco-Pozo et al. 2024)
    "ssl_obs": dict(
        model=dict(readout_hidden=64, readout_obs=True, detach_readout=True, predict=("obs",)),
        weights={**RL, "obs": 1.0},
    ),
    # As above, but the RNN also predicts the reward each action earns
    "ssl_obs_reward": dict(
        model=dict(readout_hidden=64, readout_obs=True, detach_readout=True,
                   predict=("obs", "reward")),
        weights={**RL, "obs": 1.0, "reward": 1.0},
    ),
    # Prediction and RL both shape the RNN
    "ssl_obs_reward_plus_rl": dict(
        model=dict(readout_hidden=64, readout_obs=True, detach_readout=False,
                   predict=("obs", "reward")),
        weights={**RL, "obs": 1.0, "reward": 1.0},
    ),
    # Control: the RNN keeps its random initial weights; only the readout learns
    "untrained_core": dict(
        model=dict(readout_hidden=64, readout_obs=True, freeze_core=True),
        weights=RL,
    ),
}


def make_agent(variant, env, model=None, weights=None, **kwargs):
    """Build an RNNAgent for a named variant.

    `model` and `weights` dicts update the variant's own settings, e.g.
    make_agent("ssl_obs", env, model={"hidden_size": 32}, seed=0)
    Other keyword arguments (inputs, gamma, lr, seed) go straight to RNNAgent.
    """
    spec = VARIANTS[variant]
    return RNNAgent(env,
                    model={**spec["model"], **(model or {})},
                    weights={**spec["weights"], **(weights or {})},
                    **kwargs)
