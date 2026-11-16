"""
Named network variants.
Author: patrick.mccarthy@dpag.ox.ac.uk

Each variant is a set of settings for RNNAgent, not a class:
    model:   ModelConfig settings (readout, gradient flow, prediction heads)
    weights: how much each loss term counts (terms left out are off)
    inputs:  (optional) which input signals the network gets
Add a variant by adding an entry; anything can be overridden when making an agent.
"""

from .agents import RNNAgent

# Reinforcement-learning terms shared by every variant. Wang et al. 2018 used critic = 0.05;
# 0.5 learned the 2AFC reversal task much faster in our tests.
RL = {"actor": 1.0, "critic": 0.5, "entropy": 0.05}

VARIANTS = {
    ## No BG
    # RL gradients backpropagated to the RNN, no prediction loss (Wang style)
    "pfc_rl": dict(
        model=dict(readout_hidden=None, detach_readout=False),
        weights=RL,
    ),
    # RL gradients backpropagated to the RNN, next-observation prediction loss
    "pfc_nextobs_rl": dict(
        model=dict(readout_hidden=None, detach_readout=False, predict=("obs",)),
        weights={**RL, "obs": 1.0},
    ),  
    # RL gradients not backpropagated to RNN, no prediction loss
    "pfc_readout_rl": dict(),
    # RL gradients not backpropagated to RNN, next-observation prediction loss
    "pfc_nextobs_readout_rl": dict(),

    ## BG with current observation input
    # RL gradients backpropagated to RNN, no prediction loss
    "pfc_bg_obs_e2e_rl": dict(
        model=dict(readout_hidden=64, readout_obs=True),
        weights=RL,
    ),
    # RL gradients backpropagated to RNN, next-observation prediction loss 
    "pfc_nextobs_bg_obs_e2e_rl": dict(
        model=dict(readout_hidden=64, readout_obs=True, predict=("obs",)),
        weights={**RL, "obs": dict()},
    ),
    # RL gradients not backpropagated to RNN, no prediction loss
    "pfc_bg_obs_rl": dict(
        model=dict(readout_hidden=64, readout_obs=True, detach_readout=True),
        weights=RL,
    ),
    # RL gradients not backpropagated to RNN, next-observation prediction loss (Blanco-Pozo style)
    "pfc_nextobs_bg_obs_rl": dict(
        model=dict(readout_hidden=64, readout_obs=True, detach_readout=True, predict=("obs",)),
        weights={**RL, "obs": 1.0},
    ),

    ## BG without current observation input
    # RL gradients backpropagated to RNN, no prediction loss
    "pfc_bg_e2e_rl": dict(
        model=dict(readout_hidden=64, readout_obs=False),
        weights=RL,
    ),
    # RL gradients backpropagated to RNN, next-observation prediction loss
    "pfc_bg_e2e_rl": dict(
        model=dict(readout_hidden=64, readout_obs=False),
        weights=RL,
    ),
    # RL gradients not backpropagated to RNN, no prediction loss
    "pfc_bg_obs_rl": dict(
        model=dict(readout_hidden=64, readout_obs=False, detach_readout=True),
        weights=RL,
    ),
    # RL gradients not backpropagated to RNN, next-observation prediction loss (Blanco-Pozo style)
    "pfc_nextobs_bg_obs_rl": dict(
        model=dict(readout_hidden=64, readout_obs=False, detach_readout=True, predict=("obs",)),
        weights={**RL, "obs": 1.0},
    ),

    ## Control: no recurrence. A feedforward network given the current stimulus and an
    # explicit window of the last n_back trials (set n_back when making the agent)
    "ff_history": dict(
        model=dict(core="none", readout_hidden=64),
        weights=RL,
        inputs=("obs", "history"),
    ),
}

CORE_VARIANTS = {
    "pfc_rl", # Wang-style 
    "pfc_readout_rl", # control for Wang to see effect of backpropagating RL losses to PFC recurrent layer
    "pfc_nextobs_bg_obs_rl", # Blanco-Pozo style
    "pfc_nextobs_bg_rl", # control for Blanco-Pozo to see effect of adding current observation to BG layer
    "pfc_nextobs_bg_obs_e2e_rl", # control for Blanco-Pozo to see effect of backpropagating RL losses to the PFC recurrent layer
    "pfc_bg_obs_rl" # control for Blanco-Pozo to see effect of removing next-observation prediction loss
    "pfc_bg_obs_e2e_rl", # double control for Blanco-Pozo to see effect of backpropagating RL losses to the PFC recurrent layer and without self-supervised 

}

def make_agent(variant, env, model=None, weights=None, **kwargs):
    """Build an RNNAgent for a named variant.

    `model` and `weights` dicts update the variant's own settings, e.g.
    make_agent("ssl_obs", env, model={"hidden_size": 32}, seed=0)
    Other keyword arguments (inputs, n_back, gamma, lr, seed) go straight to RNNAgent.
    """
    spec = VARIANTS[variant]
    if "inputs" in spec:
        kwargs = {"inputs": spec["inputs"], **kwargs}
    return RNNAgent(env,
                    model={**spec["model"], **(model or {})},
                    weights={**spec["weights"], **(weights or {})},
                    **kwargs)
