"""
Agents for interacting with task environments.
Author: patrick.mccarthy@dpag.ox.ac.uk

All agents share one interface, so the same loop (training.run_episode) can run any of them:
    reset()                                      start of an episode
    act(obs) -> action                           choose an action
    observe(obs, action, reward, next_obs, done) take in the result of a step
    record() -> dict                             internal variables to log at this step
"""

import numpy as np
import torch
import torch.nn.functional as F

from .losses import actor_critic_losses, prediction_losses, weighted_total
from .models import ModelConfig, RecurrentModel


class Agent:
    """Base class defining the shared interface."""

    def reset(self):
        pass

    def act(self, obs):
        raise NotImplementedError

    def observe(self, obs, action, reward, next_obs, done):
        pass

    def record(self):
        return {}


def softmax(x, beta):
    p = np.exp(beta * (x - x.max()))
    return p / p.sum()


# ---------------------------------------------------------------------- tabular agents

class QLearner(Agent):
    """Tabular Q-learning: one learned value per (observation, action) pair.

    With gamma = 0 each update is Q[o, a] += alpha * (r - Q[o, a]), which suits the 2AFC,
    where each choice's reward arrives immediately. Values are kept across episodes.
    After a reversal it must relearn each stimulus separately, so it shows no inference.
    """

    def __init__(self, env, alpha=0.1, beta=5.0, gamma=0.0, seed=None):
        self.alpha, self.beta, self.gamma = alpha, beta, gamma
        self.Q = np.zeros((env.observation_space.n, env.action_space.n))
        self.rng = np.random.default_rng(seed)
        self.learning = True  # set False to evaluate with values frozen

    def act(self, obs):
        # Softmax choice: higher beta = more often picks the highest-valued action
        return int(self.rng.choice(self.Q.shape[1], p=softmax(self.Q[obs], self.beta)))

    def observe(self, obs, action, reward, next_obs, done):
        if not self.learning:
            return
        future = 0.0 if done else self.gamma * self.Q[next_obs].max()
        self.Q[obs, action] += self.alpha * (reward + future - self.Q[obs, action])

    def record(self):
        return {"Q": self.Q.copy()}


class BayesAgent(Agent):
    """Ideal observer for the TwoAFC: tracks the probability that each rule is in force.

    It knows the task's structure (the two rules, the reversal and reward probabilities) but
    never sees the true rule. After each choice it updates its belief with Bayes' rule given
    the outcome, then allows for a possible reversal before the next trial. It has nothing
    to train: it is a reference for the best achievable behaviour and for the belief
    (`belief` = P(rule 1)) that networks could be decoded against.
    """

    def __init__(self, env, p_reversal=None, p_reward=None, beta=None, seed=None):
        # Task structure; the true parameters are used unless others are given
        self.rules = env.RULES
        self.obs_labels = env.OBSERVATIONS
        self.action_labels = env.ACTIONS
        self.initial_rule = env.initial_rule
        self.p_reversal = env.p_reversal if p_reversal is None else p_reversal
        self.p_reward = env.p_reward if p_reward is None else p_reward
        self.beta = beta  # None: always pick the more probable correct action
        self.rng = np.random.default_rng(seed)

    def reset(self):
        # Prior over the rule at the start of a session
        self.belief = 0.5 if self.initial_rule is None else float(self.initial_rule == 1)

    def p_correct(self, stimulus):
        """Probability that each action is correct for this stimulus, given the belief."""
        p_rule = {0: 1 - self.belief, 1: self.belief}
        return np.array([
            sum(p for rule, p in p_rule.items() if self.rules[rule][stimulus] == action)
            for action in self.action_labels
        ])

    def act(self, obs):
        stimulus = self.obs_labels[obs]
        if stimulus == "OUTCOME":
            return 0  # action is ignored on outcome steps
        p = self.p_correct(stimulus)
        if self.beta is None:
            best = np.flatnonzero(p == p.max())
            return int(self.rng.choice(best))  # break ties at random
        return int(self.rng.choice(len(p), p=softmax(p, self.beta)))

    def observe(self, obs, action, reward, next_obs, done):
        stimulus = self.obs_labels[obs]
        if stimulus == "OUTCOME":
            return

        # 1) Bayes' rule: how likely was this outcome under each rule?
        rewarded = reward > 0
        likelihood = {}
        for rule in (0, 1):
            correct = self.rules[rule][stimulus] == self.action_labels[action]
            if correct:
                likelihood[rule] = self.p_reward if rewarded else 1 - self.p_reward
            else:
                likelihood[rule] = 0.0 if rewarded else 1.0  # wrong choices are never rewarded
        numerator = self.belief * likelihood[1]
        self.belief = numerator / (numerator + (1 - self.belief) * likelihood[0])

        # 2) The rule may reverse before the next trial
        self.belief = self.belief * (1 - self.p_reversal) + (1 - self.belief) * self.p_reversal

    def record(self):
        return {"belief": self.belief}


# ---------------------------------------------------------------------- recurrent agent

class RNNAgent(Agent):
    """Agent built on a RecurrentModel, trained on batches of whole episodes.

    Which inputs it receives, which heads the model has and which losses it is trained
    with are all set by arguments, so one class covers every variant (see variants.py).

    Args:
        env: an environment, used for the sizes of the observation and action spaces
        model: dict of ModelConfig settings (everything except the sizes)
        weights: dict of loss weights, e.g. {"actor": 1, "critic": 0.05, "entropy": 0.05}
        inputs: which signals to feed the network, in order, from
            "obs" (current observation), "prev_action", "prev_reward"
        gamma: discount factor for future reward (per step). 0 = each action is judged only
            by its immediate reward, which is all the 2AFC needs; tasks where reward comes
            after later steps (e.g. the two-step task) need gamma > 0
        lr: learning rate
    """

    def __init__(self, env, model, weights, inputs=("obs", "prev_action", "prev_reward"),
                 gamma=0.0, lr=1e-3, seed=None):
        # Sizes as plain ints (gymnasium stores them as numpy ints, which torch rejects)
        self.n_obs = int(env.observation_space.n)
        self.n_actions = int(env.action_space.n)
        self.inputs, self.weights, self.gamma = inputs, weights, gamma

        # Which observations the agent's action counts on (the env ignores actions on OUTCOME)
        self.counts = torch.tensor([label != "OUTCOME" for label in env.OBSERVATIONS])

        if seed is not None:
            torch.manual_seed(seed)  # for the initial weights
        self.generator = torch.Generator().manual_seed(seed if seed is not None else 0)

        sizes = {"obs": self.n_obs, "prev_action": self.n_actions, "prev_reward": 1}
        cfg = ModelConfig(input_dim=sum(sizes[k] for k in inputs),
                          obs_dim=self.n_obs, n_actions=self.n_actions, **model)
        self.model = RecurrentModel(cfg)
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.Adam(trainable, lr=lr)

    # ---- inputs

    def one_hot(self, x, n):
        """One-hot encoding, where -1 (e.g. no previous action yet) gives all zeros."""
        return F.one_hot(x.clamp(min=0), n).float() * (x >= 0).unsqueeze(-1)

    def build_input(self, obs, prev_action, prev_reward):
        """Concatenate the chosen input signals. Arguments are tensors of the same shape."""
        parts = {
            "obs": self.one_hot(obs, self.n_obs),
            "prev_action": self.one_hot(prev_action, self.n_actions),
            "prev_reward": prev_reward.float().unsqueeze(-1),
        }
        return torch.cat([parts[k] for k in self.inputs], dim=-1)

    # ---- step-by-step interface (evaluation, recording)

    def reset(self):
        self.h = None
        self.prev_action = torch.tensor([-1])  # no previous action at the start
        self.prev_reward = torch.tensor([0.0])

    @torch.no_grad()
    def act(self, obs):
        obs = torch.tensor([obs])
        x = self.build_input(obs, self.prev_action, self.prev_reward)[:, None]  # [1, 1, D]
        out = self.model(x, self.one_hot(obs, self.n_obs)[:, None], self.h)
        self.h = out["h"]
        self.last_hidden = out["hidden"][0, 0].numpy()
        probs = F.softmax(out["logits"][0, 0], dim=-1)
        return int(torch.multinomial(probs, 1, generator=self.generator))

    def observe(self, obs, action, reward, next_obs, done):
        # Becomes the "previous action/reward" input on the next step. An ignored action
        # (outcome step) is passed on as "no action" (-1), so the input only shows real choices.
        self.prev_action = torch.tensor([action if self.counts[obs] else -1])
        self.prev_reward = torch.tensor([float(reward)])

    def record(self):
        return {"hidden": self.last_hidden}

    # ---- batch training

    def sequence_inputs(self, batch):
        """Inputs for whole episodes: step t sees a_{t-1} and r_{t-1} (none at t = 0)."""
        choices = torch.where(batch["mask"].bool(), batch["actions"], -1)  # ignored actions -> -1
        prev_action = torch.cat([torch.full_like(choices[:, :1], -1), choices[:, :-1]], dim=1)
        prev_reward = torch.cat([torch.zeros_like(batch["rewards"][:, :1]),
                                 batch["rewards"][:, :-1]], dim=1)
        x = self.build_input(batch["obs"], prev_action, prev_reward)
        return x, self.one_hot(batch["obs"], self.n_obs)

    @torch.no_grad()
    def collect_episodes(self, envs, seeds):
        """Run one episode in each environment in parallel, sampling from the policy.

        All episodes must have the same length (true for the TwoAFC). Returns a batch
        of [n_envs, time] tensors in the format losses.py expects.
        """
        obs = torch.tensor([env.reset(seed=s)[0] for env, s in zip(envs, seeds)])
        prev_action = torch.full((len(envs),), -1)
        prev_reward = torch.zeros(len(envs))
        h = None
        steps = {"obs": [], "next_obs": [], "actions": [], "rewards": [], "mask": []}
        done = False
        while not done:
            x = self.build_input(obs, prev_action, prev_reward)[:, None]
            out = self.model(x, self.one_hot(obs, self.n_obs)[:, None], h)
            h = out["h"]
            actions = torch.multinomial(F.softmax(out["logits"][:, 0], -1), 1,
                                        generator=self.generator).squeeze(-1)

            results = [env.step(int(a)) for env, a in zip(envs, actions)]
            next_obs = torch.tensor([r[0] for r in results])
            rewards = torch.tensor([r[1] for r in results], dtype=torch.float)
            terminated = [r[2] for r in results]
            assert all(terminated) or not any(terminated), "episodes must end together"
            done = terminated[0]

            # The action counts on stimulus steps; on outcome steps it is ignored
            counts = self.counts[obs]

            for key, value in zip(steps, (obs, next_obs, actions, rewards, counts.float())):
                steps[key].append(value)
            obs, prev_reward = next_obs, rewards
            prev_action = torch.where(counts, actions, -1)  # ignored actions -> "no action"

        return {key: torch.stack(values, dim=1) for key, values in steps.items()}

    def update(self, batch):
        """One gradient step on a batch of episodes. Returns the value of each loss term."""
        x, obs_onehot = self.sequence_inputs(batch)
        out = self.model(x, obs_onehot)

        terms = actor_critic_losses(out, batch, self.gamma)
        if self.model.pred:
            action_onehot = self.one_hot(batch["actions"], self.n_actions)
            terms.update(prediction_losses(self.model.predictions(out["hidden"], action_onehot),
                                           batch))

        loss = weighted_total(terms, self.weights)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        return {name: float(value) for name, value in terms.items()}
