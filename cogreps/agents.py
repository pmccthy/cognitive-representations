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


class TwoAFCInference(Agent):
    """Bayesian inference for the TwoAFC (the ideal observer): tracks the probability that each
    rule is in force.

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


# ---------------------------------------------------------------------- two-step task agents
#
# Baselines for the two-step task: the model-free, model-based and Bayesian inference strategies
# of Blanco-Pozo et al. (2024), Methods. Each gives a value to the two first-step choices (LEFT,
# RIGHT) and samples its choice from a softmax of those values. Outside the CHOICE state there is
# only one valid action (poke the lit port, or CENTER after an outcome), which all of them take.
# They learn once per trial, on the step where the outcome is revealed, and are reset at the
# start of each episode.

OTHER = {"LEFT": "RIGHT", "RIGHT": "LEFT", "UP": "DOWN", "DOWN": "UP"}


def sample_choice(values, beta, rng):
    """LEFT or RIGHT, sampled from softmax(beta * value). values = {"LEFT": .., "RIGHT": ..}."""
    p = softmax(np.array([values["LEFT"], values["RIGHT"]]), beta)
    return ("LEFT", "RIGHT")[rng.choice(2, p=p)]


def only_valid_action(env, obs):
    """Outside CHOICE: poke the lit port, or CENTER after an outcome."""
    state = env.OBSERVATIONS[obs]
    port = state.removesuffix("-ACTIVE") if state.endswith("-ACTIVE") else "CENTER"
    return env.ACTIONS.index(port)


def is_outcome_step(env, obs, next_obs):
    """True on the step where poking the lit port revealed the trial's outcome."""
    return env.OBSERVATIONS[obs].endswith("-ACTIVE") and env.OBSERVATIONS[next_obs] in env.OUTCOMES


def p_port(env, choice):
    """P(second-step port | choice) under the task's transition rule."""
    common = env.TRANSITIONS[env.transition_rule][choice]
    return {common: env.p_common, OTHER[common]: 1 - env.p_common}


class TwoStepModelFree(Agent):
    """Model-free strategy: values each choice by the rewards that followed it.

    After each trial (choice c, port s reached, reward r):
        Q(c) <- (1 - alpha) Q(c) + alpha ((1 - lam) V(s) + lam r)
        V(s) <- (1 - alpha) V(s) + alpha r
    lam (the eligibility trace) sets whether Q(c) moves towards the reward itself (lam = 1) or
    towards the value of the port reached (lam = 0). V(s) is used before its own update. The
    transition structure is never used.

    Asymmetric version (paper): alpha_neg is the learning rate after omissions (None = alpha),
    and with forget > 0 the choice not made and the port not visited decay towards 0.5.

    Defaults maximise reward on the two-step task with 20-40 trial blocks (grid search,
    18/11/26; the optimum is flat, and beta is shared across the three baselines).
    """

    def __init__(self, env, alpha=0.8, beta=20.0, lam=1.0, alpha_neg=None, forget=0.0, seed=None):
        self.env = env
        self.alpha, self.beta, self.lam = alpha, beta, lam
        self.alpha_neg = alpha if alpha_neg is None else alpha_neg
        self.forget = forget
        self.rng = np.random.default_rng(seed)

    def reset(self):
        self.Q = {"LEFT": 0.5, "RIGHT": 0.5}  # choice values
        self.V = {"UP": 0.5, "DOWN": 0.5}     # second-step port values

    def act(self, obs):
        if self.env.OBSERVATIONS[obs] == "CHOICE":
            self.choice = sample_choice(self.Q, self.beta, self.rng)
            return self.env.ACTIONS.index(self.choice)
        return only_valid_action(self.env, obs)

    def observe(self, obs, action, reward, next_obs, done):
        if not is_outcome_step(self.env, obs, next_obs):
            return
        c, s, r = self.choice, self.env.ACTIONS[action], float(reward)
        a = self.alpha if r > 0 else self.alpha_neg
        self.Q[c] = (1 - a) * self.Q[c] + a * ((1 - self.lam) * self.V[s] + self.lam * r)
        self.V[s] = (1 - a) * self.V[s] + a * r

        # Forgetting: what wasn't experienced this trial decays towards 0.5
        self.Q[OTHER[c]] += self.forget * (0.5 - self.Q[OTHER[c]])
        self.V[OTHER[s]] += self.forget * (0.5 - self.V[OTHER[s]])

    def record(self):
        return {"q_left": self.Q["LEFT"], "q_right": self.Q["RIGHT"],
                "v_up": self.V["UP"], "v_down": self.V["DOWN"]}


class TwoStepModelBased(Agent):
    """Model-based strategy: learns each port's value and values each choice through the
    known transition probabilities.

    After each trial (port s reached, reward r):
        V(s) <- (1 - alpha) V(s) + alpha r
    and the choice values are
        Q(c) = sum_s P(s | c) V(s)

    Asymmetric version (paper): alpha_neg is the learning rate after omissions (None = alpha),
    and with forget > 0 the port not visited decays towards forget_to (0.5 in the paper; 0 in
    one of its variants).

    Defaults maximise reward on the two-step task with 20-40 trial blocks (grid search,
    18/11/26; the optimum is flat, and beta is shared across the three baselines).
    """

    def __init__(self, env, alpha=0.7, beta=20.0, alpha_neg=None, forget=0.0, forget_to=0.5, seed=None):
        self.env = env
        self.alpha, self.beta = alpha, beta
        self.alpha_neg = alpha if alpha_neg is None else alpha_neg
        self.forget, self.forget_to = forget, forget_to
        self.rng = np.random.default_rng(seed)

    def reset(self):
        self.V = {"UP": 0.5, "DOWN": 0.5}  # second-step port values

    def choice_values(self):
        return {c: sum(p * self.V[s] for s, p in p_port(self.env, c).items()) for c in ("LEFT", "RIGHT")}

    def act(self, obs):
        if self.env.OBSERVATIONS[obs] == "CHOICE":
            self.choice = sample_choice(self.choice_values(), self.beta, self.rng)
            return self.env.ACTIONS.index(self.choice)
        return only_valid_action(self.env, obs)

    def observe(self, obs, action, reward, next_obs, done):
        if not is_outcome_step(self.env, obs, next_obs):
            return
        s, r = self.env.ACTIONS[action], float(reward)
        a = self.alpha if r > 0 else self.alpha_neg
        self.V[s] = (1 - a) * self.V[s] + a * r
        self.V[OTHER[s]] += self.forget * (self.forget_to - self.V[OTHER[s]])

    def record(self):
        Q = self.choice_values()
        return {"q_left": Q["LEFT"], "q_right": Q["RIGHT"], "v_up": self.V["UP"], "v_down": self.V["DOWN"]}


class TwoStepInference(Agent):
    """Bayesian inference strategy for the two-step task: tracks the probability that UP is
    the good port (the counterpart of TwoAFCInference).

    It knows the task's structure (transition probabilities, reward probabilities, how often
    the block switches) but never sees the true block. After each trial it updates its belief
    with Bayes' rule given which port it reached and whether it was rewarded, then allows for a
    possible block switch before the next trial. Choices are valued by their expected reward:
        P(reward | s) = belief * P(reward | s, UP good) + (1 - belief) * P(reward | s, DOWN good)
        Q(c) = sum_s P(s | c) P(reward | s)
    `belief` = P(UP is the good port) is what networks could be decoded against.

    Args (the true task parameters are used unless others are given):
        p_reversal: probability that the block switches after a trial. From the task: 1 / mean
            block length with block_len, otherwise its p_reversal (1/30 if neither is set). With
            block_len this assumes a constant switch probability, so it can't use the block
            timing (no switch in a block's first 20 trials) and isn't quite the ideal observer.
        p_good, p_bad: reward probabilities at the good and bad port
        beta: softmax inverse temperature. Reward is within 0.002 per trial of beta = 100 at
            the default.
        asymmetric: the paper's asymmetric version, in which omissions carry no information
            about the block (it sees omission at either port as the same observation), so only
            rewards update the belief
    """

    def __init__(self, env, p_reversal=None, p_good=None, p_bad=None, beta=20.0, asymmetric=False,
                 seed=None):
        self.env = env
        if p_reversal is not None:
            self.p_reversal = p_reversal
        elif env.block_len is not None:
            self.p_reversal = 2 / sum(env.block_len)  # 1 / mean block length
        else:
            self.p_reversal = env.p_reversal if env.p_reversal > 0 else 1 / 30
        self.p_good = env.BLOCKS["UP"]["UP"] if p_good is None else p_good
        self.p_bad = env.BLOCKS["UP"]["DOWN"] if p_bad is None else p_bad
        self.beta, self.asymmetric = beta, asymmetric
        self.rng = np.random.default_rng(seed)

    def reset(self):
        self.belief = 0.5  # P(UP is the good port) at the start of a session

    def p_reward(self, port, good_port):
        """P(reward at this port | which port is good)."""
        return self.p_good if port == good_port else self.p_bad

    def choice_values(self):
        """Expected reward of each choice, given the belief."""
        p_reward = {s: self.belief * self.p_reward(s, "UP") + (1 - self.belief) * self.p_reward(s, "DOWN")
                    for s in ("UP", "DOWN")}
        return {c: sum(p * p_reward[s] for s, p in p_port(self.env, c).items()) for c in ("LEFT", "RIGHT")}

    def act(self, obs):
        if self.env.OBSERVATIONS[obs] == "CHOICE":
            self.choice = sample_choice(self.choice_values(), self.beta, self.rng)
            return self.env.ACTIONS.index(self.choice)
        return only_valid_action(self.env, obs)

    def observe(self, obs, action, reward, next_obs, done):
        if not is_outcome_step(self.env, obs, next_obs):
            return
        port, rewarded = self.env.ACTIONS[action], reward > 0

        # 1) Bayes' rule: how likely was this outcome if UP were good, and if DOWN were good?
        if rewarded or not self.asymmetric:
            likelihood = {}
            for good_port in ("UP", "DOWN"):
                p = self.p_reward(port, good_port)
                likelihood[good_port] = p if rewarded else 1 - p
            numerator = self.belief * likelihood["UP"]
            self.belief = numerator / (numerator + (1 - self.belief) * likelihood["DOWN"])

        # 2) The block may switch before the next trial
        self.belief = self.belief * (1 - self.p_reversal) + (1 - self.belief) * self.p_reversal

    def record(self):
        Q = self.choice_values()
        return {"belief": self.belief, "q_left": Q["LEFT"], "q_right": Q["RIGHT"]}


# ---------------------------------------------------------------------- recurrent agent

class TrainingDiverged(RuntimeError):
    """Raised by RNNAgent.update when the loss or gradients stop being finite. The weights are
    left as they were before that update. `terms` holds the loss terms at that point."""

    def __init__(self, message, terms=None):
        super().__init__(message)
        self.terms = terms or {}


class RNNAgent(Agent):
    """Agent built on a RecurrentModel, trained on batches of whole episodes.

    Which inputs it receives, which heads the model has and which losses it is trained
    with are all set by arguments, so one class covers every variant (see variants.py).

    Args:
        env: an environment, used for the sizes of the observation and action spaces
        model: dict of ModelConfig settings (everything except the sizes)
        weights: dict of loss weights, e.g. {"actor": 1, "critic": 0.05, "entropy": 0.05}
        inputs: which signals to feed the network, in order, from
            "obs" (current observation), "prev_action", "prev_reward", and
            "history" (stimulus, choice and reward of each of the last n_back trials)
        n_back: number of past trials in the "history" input
        gamma: discount factor for future reward (per step). 0 = each action is judged only
            by its immediate reward, which is all the 2AFC needs; tasks where reward comes
            after later steps (e.g. the two-step task) need gamma > 0
        lr: learning rate. With lr_bg set, this is the PFC learning rate only.
        lr_bg: separate learning rate for the "BG" readout, as in Blanco-Pozo et al. (2024),
            who trained PFC and BG with separate optimisers (lr 0.01 and 0.05). None = one
            learning rate for the whole model. When set, the parameters are split in two:
                PFC: the recurrent core and the prediction heads   (learning rate lr)
                BG:  the readout layer and the policy and value heads   (learning rate lr_bg)
            and gradients are clipped separately in each, so the two parts train as if they
            had their own optimisers. (In variants without a readout layer, BG is just the
            policy and value heads.)
        bootstrap: for episodes that are cut off rather than ended by the task, add the
            estimated value of the state after the last step to the returns (as in
            Blanco-Pozo et al. 2024), since the task would have gone on.
    """

    def __init__(self, env, model, weights, inputs=("obs", "prev_action", "prev_reward"),
                 n_back=1, gamma=0.0, lr=1e-3, lr_bg=None, bootstrap=False, seed=None):
        # Sizes as plain ints (gymnasium stores them as numpy ints, which torch rejects)
        self.n_obs = int(env.observation_space.n)
        self.n_actions = int(env.action_space.n)
        self.inputs, self.weights, self.gamma = inputs, weights, gamma
        self.bootstrap = bootstrap
        self.carry = None  # network state at the end of the last collection (continuing sessions)
        self.n_back = n_back
        self.trial_size = self.n_obs + self.n_actions + 1  # one past trial: stimulus, choice, reward

        # Which observations the agent's action counts on (the env ignores actions on OUTCOME)
        self.counts = torch.tensor([label != "OUTCOME" for label in env.OBSERVATIONS])

        if seed is not None:
            torch.manual_seed(seed)  # for the initial weights
        self.generator = torch.Generator().manual_seed(seed if seed is not None else 0)

        sizes = {"obs": self.n_obs, "prev_action": self.n_actions, "prev_reward": 1,
                 "history": n_back * self.trial_size}
        cfg = ModelConfig(input_dim=sum(sizes[k] for k in inputs),
                          obs_dim=self.n_obs, n_actions=self.n_actions, **model)
        self.model = RecurrentModel(cfg)
        self.lr, self.lr_bg = lr, lr_bg
        self.optimizer = torch.optim.Adam(self.param_groups())

    def param_groups(self):
        """Trainable parameters for the optimiser: one group, or PFC and BG groups (see lr_bg)."""
        def trainable(*modules):
            return [p for m in modules for p in m.parameters() if p.requires_grad]

        if self.lr_bg is None:
            return [{"name": "all", "params": trainable(self.model), "lr": self.lr}]
        m = self.model
        groups = [{"name": "pfc", "params": trainable(m.core, m.pred), "lr": self.lr},
                  {"name": "bg", "params": trainable(m.readout, m.policy, m.value), "lr": self.lr_bg}]
        n_grouped = sum(len(g["params"]) for g in groups)
        assert n_grouped == len(trainable(m)), "some trainable parameters are in neither PFC nor BG"
        return [g for g in groups if g["params"]]  # e.g. no PFC parameters in the feedforward control

    # ---- inputs

    def one_hot(self, x, n):
        """One-hot encoding, where -1 (e.g. no previous action yet) gives all zeros."""
        return F.one_hot(x.clamp(min=0), n).float() * (x >= 0).unsqueeze(-1)

    def build_input(self, obs, prev_action, prev_reward, history):
        """Concatenate the chosen input signals.

        obs, prev_action and prev_reward have the same shape ([batch] or [batch, time]);
        history has two extra dimensions at the end, [..., n_back, trial_size].
        """
        parts = {
            "obs": lambda: self.one_hot(obs, self.n_obs),
            "prev_action": lambda: self.one_hot(prev_action, self.n_actions),
            "prev_reward": lambda: prev_reward.float().unsqueeze(-1),
            "history": lambda: history.flatten(-2),
        }
        return torch.cat([parts[k]() for k in self.inputs], dim=-1)

    def empty_history(self, batch_size):
        """History before any trials: all zeros."""
        return torch.zeros(batch_size, self.n_back, self.trial_size)

    def push_history(self, history, obs, action, reward, counts):
        """Add this step to the front of the trial history if it was a choice (counts = True).

        history: [batch, n_back, trial_size], most recent trial first; the oldest drops off.
        """
        trial = torch.cat([self.one_hot(obs, self.n_obs), self.one_hot(action, self.n_actions),
                           reward.float().unsqueeze(-1)], dim=-1)
        shifted = torch.cat([trial[:, None], history[:, :-1]], dim=1)
        return torch.where(counts[:, None, None], shifted, history)

    # ---- step-by-step interface (evaluation, recording)

    def reset(self):
        self.h = None
        self.prev_action = torch.tensor([-1])  # no previous action at the start
        self.prev_reward = torch.tensor([0.0])
        self.history = self.empty_history(1)

    @torch.no_grad()
    def act(self, obs):
        obs = torch.tensor([obs])
        x = self.build_input(obs, self.prev_action, self.prev_reward, self.history)[:, None]  # [1, 1, D]
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
        self.history = self.push_history(self.history, torch.tensor([obs]), torch.tensor([action]),
                                         torch.tensor([float(reward)]), self.counts[[obs]])

    def record(self):
        return {"hidden": self.last_hidden}

    # ---- batch training

    def sequence_inputs(self, batch):
        """Inputs for whole episodes: step t sees a_{t-1} and r_{t-1} and the trials completed
        before step t. At t = 0 these come from batch["start"]: nothing at the start of a
        session, or what came before when a session is continued."""
        start = batch["start"]
        choices = torch.where(batch["mask"].bool(), batch["actions"], -1)  # ignored actions -> -1
        prev_action = torch.cat([start["prev_action"][:, None], choices[:, :-1]], dim=1)
        prev_reward = torch.cat([start["prev_reward"][:, None], batch["rewards"][:, :-1]], dim=1)

        # Trial history at each step, rebuilt exactly as it was while acting
        history = start["history"]
        histories = []
        for t in range(batch["obs"].shape[1]):
            histories.append(history)
            history = self.push_history(history, batch["obs"][:, t], batch["actions"][:, t],
                                        batch["rewards"][:, t], batch["mask"][:, t].bool())
        history = torch.stack(histories, dim=1)  # [B, T, n_back, trial_size]

        x = self.build_input(batch["obs"], prev_action, prev_reward, history)
        return x, self.one_hot(batch["obs"], self.n_obs)

    def start_state(self, n):
        """Network state and previous inputs at the start of a session (nothing yet)."""
        return {"h": None, "prev_action": torch.full((n,), -1), "prev_reward": torch.zeros(n),
                "history": self.empty_history(n)}

    @torch.no_grad()
    def collect_episodes(self, envs, seeds, continue_session=False):
        """Run one episode in each environment in parallel, sampling from the policy.

        Episodes may end at different steps: finished ones are padded to the length of the
        longest and batch["valid"] marks the real steps. With continue_session=True the
        environments aren't reset: each episode continues its session from where the previous
        collection stopped, with the network's state and previous inputs carried over (as in
        Blanco-Pozo et al. 2024). Returns [n_envs, time] tensors in the format losses.py
        expects, plus:
            start        network state and previous inputs at the first step
            final_value  value of the state after each episode's last step (for bootstrapping)
            truncated    whether each episode was cut off rather than ended by the task
        """
        n = len(envs)
        if continue_session and self.carry is not None:
            start = self.carry
            obs = torch.tensor([env.reset(options={"continue": True})[0] for env in envs])
        else:
            start = self.start_state(n)
            obs = torch.tensor([env.reset(seed=s)[0] for env, s in zip(envs, seeds)])
        h, prev_action, prev_reward, history = (start["h"], start["prev_action"],
                                                start["prev_reward"], start["history"])

        active = torch.ones(n, dtype=torch.bool)      # episodes still running
        truncated = torch.zeros(n, dtype=torch.bool)
        steps = {"obs": [], "next_obs": [], "actions": [], "rewards": [], "mask": [], "valid": []}
        while active.any():
            x = self.build_input(obs, prev_action, prev_reward, history)[:, None]
            out = self.model(x, self.one_hot(obs, self.n_obs)[:, None], h)
            if not torch.isfinite(out["logits"]).all():
                raise TrainingDiverged("non-finite network outputs while acting")
            actions = torch.multinomial(F.softmax(out["logits"][:, 0], -1), 1,
                                        generator=self.generator).squeeze(-1)

            # Step only the environments whose episode is still running
            next_obs, rewards = obs.clone(), torch.zeros(n)
            ended = torch.zeros(n, dtype=torch.bool)
            for i in torch.flatten(torch.nonzero(active)).tolist():
                o, r, terminated, cut_off, _ = envs[i].step(int(actions[i]))
                next_obs[i], rewards[i] = o, r
                ended[i], truncated[i] = terminated or cut_off, cut_off and not terminated

            # The action counts on stimulus steps (on outcome steps it is ignored) of running episodes
            counts = self.counts[obs] & active

            for key, value in zip(steps, (obs, next_obs, actions, rewards, counts.float(), active.float())):
                steps[key].append(value)

            # Finished episodes keep their final state and inputs (for continuing or bootstrapping)
            h = self.model.select_state(~active, h, out["h"])
            history = self.push_history(history, obs, actions, rewards, counts)
            prev_reward = torch.where(active, rewards, prev_reward)
            prev_action = torch.where(active, torch.where(counts, actions, -1), prev_action)
            obs = next_obs
            active &= ~ended

        # Value of the state after each episode's last step (only needed for bootstrapping)
        final_value = torch.zeros(n)
        if self.bootstrap:
            x = self.build_input(obs, prev_action, prev_reward, history)[:, None]
            final_value = self.model(x, self.one_hot(obs, self.n_obs)[:, None], h)["value"][:, 0]

        self.carry = {"h": h, "prev_action": prev_action, "prev_reward": prev_reward, "history": history}
        batch = {key: torch.stack(values, dim=1) for key, values in steps.items()}
        batch.update(start=start, final_value=final_value, truncated=truncated)
        return batch

    def update(self, batch):
        """One gradient step on a batch of episodes. Returns the value of each loss term."""
        x, obs_onehot = self.sequence_inputs(batch)
        out = self.model(x, obs_onehot, batch["start"]["h"])  # carried state when continuing

        terms = actor_critic_losses(out, batch, self.gamma, self.bootstrap)
        if self.model.pred:
            action_onehot = self.one_hot(batch["actions"], self.n_actions)
            terms.update(prediction_losses(self.model.predictions(out["hidden"], action_onehot),
                                           batch))

        loss = weighted_total(terms, self.weights)
        self.optimizer.zero_grad()
        loss.backward()
        values = {name: float(value.detach()) for name, value in terms.items()}

        # Stop before a non-finite loss or gradient reaches the weights (training has diverged)
        bad = [name for name, p in self.model.named_parameters()
               if p.grad is not None and not torch.isfinite(p.grad).all()]
        if not torch.isfinite(loss) or bad:
            self.optimizer.zero_grad()
            sizes = {"max_abs_logit": float(out["logits"].detach().abs().max()),
                     "max_abs_value": float(out["value"].detach().abs().max()),
                     "max_abs_hidden": float(out["hidden"].detach().abs().max()),
                     "max_abs_weight": max(float(p.detach().abs().max()) for p in self.model.parameters())}
            raise TrainingDiverged(f"non-finite loss or gradients (loss terms {values}; "
                                   f"non-finite gradients in {bad or 'none'}; sizes {sizes})", values)

        for group in self.optimizer.param_groups:  # per part, so PFC and BG don't scale each other
            torch.nn.utils.clip_grad_norm_(group["params"], 1.0)
        self.optimizer.step()
        return values
