"""
Loss terms for training recurrent agents.
Author: patrick.mccarthy@dpag.ox.ac.uk

Every term is something to minimise. A variant chooses which terms to use and how much
each counts through a weights dict, e.g. {"actor": 1, "critic": 0.05, "entropy": 0.05, "obs": 1}.

All functions take a batch of whole episodes as [batch, time] tensors:
    obs, next_obs   observation before and after each step (integers)
    actions         action taken at each step (integers)
    rewards         reward returned for each action
    mask            1 where the action counts (stimulus steps), 0 where it is ignored
"""

import torch
import torch.nn.functional as F


def discounted_returns(rewards, gamma):
    """Sum of discounted future rewards from each step to the end of the episode."""
    returns = torch.zeros_like(rewards)
    running = torch.zeros_like(rewards[:, 0])
    for t in reversed(range(rewards.shape[1])):
        running = rewards[:, t] + gamma * running
        returns[:, t] = running
    return returns


def masked_mean(x, mask):
    return (x * mask).sum() / mask.sum().clamp(min=1)


def actor_critic_losses(out, batch, gamma):
    """Actor-critic terms, using each step's full discounted return as the target.

    actor:   make actions with better-than-expected returns more likely
    critic:  make the value estimate predict the return
    entropy: negative policy entropy (minimising it keeps the policy exploratory)
    The actor and entropy terms only use steps where the action counts.
    """
    returns = discounted_returns(batch["rewards"], gamma)
    advantage = (returns - out["value"]).detach()  # how much better than expected

    log_probs = F.log_softmax(out["logits"], dim=-1)
    log_prob_taken = log_probs.gather(-1, batch["actions"].unsqueeze(-1)).squeeze(-1)
    entropy = -(log_probs.exp() * log_probs).sum(-1)

    return {
        "actor": -masked_mean(log_prob_taken * advantage, batch["mask"]),
        "critic": 0.5 * ((returns - out["value"]) ** 2).mean(),
        "entropy": -masked_mean(entropy, batch["mask"]),
    }


def prediction_losses(preds, batch):
    """Prediction terms, one per head the model has.

    obs:    cross-entropy for predicting the next observation
    reward: binary cross-entropy for predicting the reward the action earns
    action: cross-entropy for predicting the next action (only where that action counts)
    """
    losses = {}
    if "obs" in preds:
        logits = preds["obs"]
        losses["obs"] = F.cross_entropy(logits.flatten(0, 1), batch["next_obs"].flatten())
    if "reward" in preds:
        losses["reward"] = F.binary_cross_entropy_with_logits(
            preds["reward"].squeeze(-1), batch["rewards"]
        )
    if "action" in preds:
        # Step t predicts a_{t+1}, so the last step has no target
        logits = preds["action"][:, :-1]
        target = batch["actions"][:, 1:]
        ce = F.cross_entropy(logits.flatten(0, 1), target.flatten(), reduction="none")
        losses["action"] = masked_mean(ce, batch["mask"][:, 1:].flatten())
    return losses


def weighted_total(terms, weights):
    """Weighted sum of the loss terms named in `weights` (a weight of 0 switches a term off)."""
    total = 0.0
    for name, w in weights.items():
        if w:
            total = total + w * terms[name]
    return total
