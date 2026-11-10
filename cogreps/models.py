"""
Neural network models.
Author: patrick.mccarthy@dpag.ox.ac.uk

One configurable recurrent model covers every network variant. A variant is a ModelConfig,
not a class: which heads exist, what the readout sees, and where gradients may flow.
"""

from dataclasses import dataclass

import torch
from torch import nn

# Recurrent core types, selected by name in the config
CORES = {"rnn": nn.RNN, "gru": nn.GRU, "lstm": nn.LSTM}

# Prediction targets the model can have heads for. Each head reads the RNN state and the
# action just taken, (h_t, a_t), and predicts:
#   "obs":    the next observation o_{t+1}
#   "reward": the reward r_t earned by a_t
#   "action": the next action a_{t+1}
PREDICTION_TARGETS = ("obs", "reward", "action")


@dataclass
class ModelConfig:
    input_dim: int                     # length of the input vector (built by the agent)
    obs_dim: int                       # number of observation types
    n_actions: int
    core: str = "gru"                  # "rnn", "gru" or "lstm"
    hidden_size: int = 64
    readout_hidden: int | None = 64    # None: linear policy/value off the RNN; int: one ReLU layer
    readout_obs: bool = False          # also feed the current observation to the readout
    detach_readout: bool = False       # True: RL losses train the readout only, never the RNN
    freeze_core: bool = False          # True: RNN weights are never trained (untrained-core control)
    predict: tuple = ()                # prediction heads, any of PREDICTION_TARGETS


class RecurrentModel(nn.Module):
    """RNN core, with a policy/value readout and optional prediction heads.

        inputs ──► RNN core ──► hidden ──┬──► readout ──► policy logits, value
                                         └──► prediction heads (given the action taken)
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        H = cfg.hidden_size

        # Recurrent core
        self.core = CORES[cfg.core](cfg.input_dim, H, batch_first=True)
        if cfg.freeze_core:
            self.core.requires_grad_(False)

        # Readout: identity (linear policy/value straight off the RNN) or one hidden ReLU layer
        readout_in = H + (cfg.obs_dim if cfg.readout_obs else 0)
        if cfg.readout_hidden is None:
            self.readout = nn.Identity()
            readout_out = readout_in
        else:
            self.readout = nn.Sequential(nn.Linear(readout_in, cfg.readout_hidden), nn.ReLU())
            readout_out = cfg.readout_hidden

        # Policy (action logits) and value (expected future reward) from the readout
        self.policy = nn.Linear(readout_out, cfg.n_actions)
        self.value = nn.Linear(readout_out, 1)

        # Prediction heads, one per target, reading the RNN state and the action taken
        out_dims = {"obs": cfg.obs_dim, "reward": 1, "action": cfg.n_actions}
        for target in cfg.predict:
            if target not in PREDICTION_TARGETS:
                raise ValueError(f"Unknown prediction target {target!r}")
        self.pred = nn.ModuleDict(
            {t: nn.Linear(H + cfg.n_actions, out_dims[t]) for t in cfg.predict}
        )

    def forward(self, x, obs_onehot=None, h=None):
        """Run the model over a sequence.

        Args:
            x: inputs, [batch, time, input_dim]
            obs_onehot: current observations, [batch, time, obs_dim] (only needed if readout_obs)
            h: initial recurrent state (None = zeros)

        Returns:
            dict with "hidden" [B, T, H], "logits" [B, T, n_actions], "value" [B, T],
            and "h", the final recurrent state for continuing the sequence.
        """
        hidden, h = self.core(x, h)

        # Stop RL gradients reaching the RNN if requested
        readout_in = hidden.detach() if self.cfg.detach_readout else hidden
        if self.cfg.readout_obs:
            readout_in = torch.cat([readout_in, obs_onehot], dim=-1)
        features = self.readout(readout_in)

        return {
            "hidden": hidden,
            "logits": self.policy(features),
            "value": self.value(features).squeeze(-1),
            "h": h,
        }

    def predictions(self, hidden, action_onehot):
        """Prediction-head outputs (logits) for each target, given the actions taken."""
        z = torch.cat([hidden, action_onehot], dim=-1)
        return {target: head(z) for target, head in self.pred.items()}
