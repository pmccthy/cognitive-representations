"""
Environments for cognitive tasks.
Author: patrick.mccarthy@dpag.ox.ac.uk
"""

import gymnasium as gym

phases = {"STIM": 0, "OUTCOME": 1}

class TwoAFC(gym.Env):
    """Two-alternative forced choice task with an optional reversal of the stimulus-action rule.

    Each trial has two timesteps:
        1. STIM:    a stimulus (STIM1 or STIM2) is shown and the agent turns LEFT or RIGHT.
                    The reward for that choice is returned on this same step.
        2. OUTCOME: an outcome marker is shown; any action ends the trial.

    Reward information reaches the agent through a previous-reward input (added by a wrapper),
    not through the observation. Observations are integers; the wrapper converts them to one-hot.
    """

    def __init__(self, n_trials=100, rule=None, p_reversal=0.0, p_reward=1.0, p_stim1=0.5):
        """
        Args:
            n_trials (int): Number of trials per episode (one episode = one session).
            rule (int or None): Starting rule (0 or 1). None samples a random rule each episode,
                which meta-learning needs so the network can't just memorise one mapping.
            p_reversal (float): Probability that the rule reverses at the end of each trial.
            p_reward (float): Probability that a correct choice is rewarded (incorrect choices
                are never rewarded). Below 1, an omission is ambiguous: bad luck or a reversal.
            p_stim1 (float or tuple): Probability that STIM1 is shown on a trial (STIM2 otherwise).
                A (low, high) tuple samples a new value uniformly at the start of each episode.
        """
        super().__init__()

        # Labels for actions and observations (index = integer value used by the spaces)
        self.ACTIONS = ["LEFT", "RIGHT"]
        self.OBSERVATIONS = ["STIM1", "STIM2", "OUTCOME"]

        # Each rule maps a stimulus to its rewarded action. Rule 1 is the reversal of rule 0.
        self.RULES = {
            0: {"STIM1": "LEFT", "STIM2": "RIGHT"},
            1: {"STIM1": "RIGHT", "STIM2": "LEFT"},
        }

        # Spaces: integer actions and observations, indexing the label lists above
        self.action_space = gym.spaces.Discrete(len(self.ACTIONS))
        self.observation_space = gym.spaces.Discrete(len(self.OBSERVATIONS))

        # Task parameters
        self.n_trials = n_trials
        self.initial_rule = rule
        self.p_reversal = p_reversal
        self.p_reward = p_reward
        self.p_stim1_setting = p_stim1

    #---------------------- gym.Env interface --------------------------------------
    def reset(self, seed=None, options=None):
        """Start a new episode (session): sample its rule and stimulus ratio, then the first trial."""
        super().reset(seed=seed)  # seeds self.np_random
        self.trial = 0

        # Starting rule: fixed, or random per episode
        if self.initial_rule is None:
            self.rule = int(self.np_random.integers(2))
        else:
            self.rule = self.initial_rule

        # Stimulus ratio: fixed, or sampled per episode from a (low, high) range
        if isinstance(self.p_stim1_setting, tuple):
            self.p_stim1 = float(self.np_random.uniform(*self.p_stim1_setting))
        else:
            self.p_stim1 = self.p_stim1_setting

        self._new_trial()
        return self.observation, self._info()

    def step(self, action):
        """Apply one action. Returns (observation, reward, terminated, truncated, info)."""
        action = self.ACTIONS[action]  # integer -> "LEFT" / "RIGHT"
        reward = 0.0
        terminated = False

        if self.phase == "STIM":
            # STIM step: score the choice against the stimulus shown; reward is returned now
            self.choice = action
            self.rewarded = action == self.correct_action and self.np_random.random() < self.p_reward
            reward = float(self.rewarded)
            self.phase = "OUTCOME"  # so the observation returned below is the outcome marker

        else:
            # OUTCOME step: the action is ignored and the trial ends
            self.trial += 1

            # Possible uncued reversal between trials
            if self.np_random.random() < self.p_reversal:
                self.reverse()

            # End the episode after n_trials, otherwise start the next trial
            terminated = self.trial >= self.n_trials
            if not terminated:
                self._new_trial()

        return self.observation, reward, terminated, False, self._info()

    # ----------------------- task logic -------------------------------------------    

    def _new_trial(self):
        """Begin a trial: sample the stimulus and clear the previous trial's choice and outcome."""
        self.phase = "STIM"
        self.stimulus = "STIM1" if self.np_random.random() < self.p_stim1 else "STIM2"
        self.choice = None
        self.rewarded = None

    def reverse(self):
        """Swap to the other rule (can also be called from outside for scheduled reversals)."""
        self.rule = 1 - self.rule

    def set_p_stim1(self, p_stim1):
        """Set the probability that STIM1 is shown (can be called from outside, like reverse).

        Takes effect from the next trial, since the current trial's stimulus is already drawn.
        Lasts until the end of the episode: reset() restores the value given to the constructor.
        """
        if not 0.0 <= p_stim1 <= 1.0:
            raise ValueError(f"p_stim1 must be between 0 and 1, got {p_stim1}")
        self.p_stim1 = float(p_stim1)

    @property
    def correct_action(self):
        # Derived from the rule and stimulus, so it can never go out of sync after a reversal
        return self.RULES[self.rule][self.stimulus]

    @property
    def observation(self):
        # The stimulus during STIM, the outcome marker during OUTCOME (as an integer)
        label = self.stimulus if self.phase == "STIM" else "OUTCOME"
        return self.OBSERVATIONS.index(label)

    def _info(self):
        # Ground truth for analysis; never shown to the agent
        return {
            "phase": self.phase,
            "stimulus": self.stimulus,
            "rule": self.rule,
            "p_stim1": self.p_stim1,
            "correct_action": self.correct_action,
            "choice": self.choice,
            "rewarded": self.rewarded,
            "trial": self.trial,
        }

class TwoStepTask(gym.Env):
    """Two-step task (Blanco-Pozo et al. 2024), following the authors' network-model version.

    Each trial takes at least three timesteps:
        1. CHOICE:          the agent pokes LEFT or RIGHT. Each choice commonly (p_common) leads
                            to one second-step port and rarely to the other.
        2. UP-/DOWN-ACTIVE: the agent pokes the active port. The reward is drawn from that port's
                            reward probability and returned on this same step.
        3. Outcome:         REWARD-AT-UP, REWARD-AT-DOWN or NO-REWARD is shown; poking CENTER
                            starts the next trial.
    Any other action leaves the state unchanged and earns nothing, so a trial can take more than
    three steps. Episodes start in NO-REWARD (which doubles as the "poke CENTER to start" state)
    and are truncated after n_steps, so every episode in a batch has the same length.

    Two hidden variables (never in the observation):
        transition rule: which port each choice commonly leads to. Fixed in Blanco-Pozo et al.;
            reverse_transitions() swaps it.
        reward block: which second-step port is good ("UP" or "DOWN": p_good there, p_bad at the
            other) or "NEUTRAL" (0.5 at both). This is what the agent has to infer. The block can
            change at the start of a trial, after block_len trials or with probability p_reversal,
            or whenever switch_block() / set_block() is called from outside.
    """

    def __init__(self, n_steps=300, transition_rule=0, p_common=0.8, block=None,
                 p_good=0.8, p_bad=0.2, p_reversal=0.0, block_len=None, p_neutral=0.0):
        """
        Args:
            n_steps (int): Timesteps per episode (a trained agent needs ~3 per trial).
            transition_rule (int or None): 0 = LEFT->UP, RIGHT->DOWN commonly; 1 = the reverse.
                None samples one per episode.
            p_common (float): Probability of the common transition.
            block (str or None): Starting reward block, "UP", "DOWN" or "NEUTRAL". None samples
                "UP" or "DOWN" each episode, so the network can't memorise one block.
            p_good, p_bad (float): Reward probabilities at the good and bad port.
            p_reversal (float): Probability that the block switches at the start of each trial
                (block lengths are then geometric, mean 1 / p_reversal).
            block_len (tuple or None): (low, high) - switch after a number of trials drawn
                uniformly from low to high inclusive (the authors' model used 20-39). Used
                instead of p_reversal.
            p_neutral (float): Probability that a switch out of a good-port block goes to a
                NEUTRAL block rather than to the other port. Switches out of NEUTRAL go to UP or
                DOWN at random.
        """
        super().__init__()

        if block_len is not None and p_reversal > 0:
            raise ValueError("use either block_len or p_reversal, not both")

        # Labels for actions and observations (index = integer value used by the spaces)
        self.ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT", "CENTER"]
        self.OBSERVATIONS = ["CHOICE", "UP-ACTIVE", "DOWN-ACTIVE",
                             "REWARD-AT-UP", "REWARD-AT-DOWN", "NO-REWARD"]
        # Observations that end a trial (reached by poking the active port); used to count trials
        self.OUTCOMES = ["REWARD-AT-UP", "REWARD-AT-DOWN", "NO-REWARD"]

        # Transition rules: the port each choice commonly leads to
        self.TRANSITIONS = {
            0: {"LEFT": "UP", "RIGHT": "DOWN"},
            1: {"LEFT": "DOWN", "RIGHT": "UP"},
        }

        # Reward blocks: the reward probability at each port
        self.BLOCKS = {
            "UP": {"UP": p_good, "DOWN": p_bad},
            "DOWN": {"UP": p_bad, "DOWN": p_good},
            "NEUTRAL": {"UP": 0.5, "DOWN": 0.5},
        }

        # Spaces: integer actions and observations, indexing the label lists above
        self.action_space = gym.spaces.Discrete(len(self.ACTIONS))
        self.observation_space = gym.spaces.Discrete(len(self.OBSERVATIONS))

        # Task parameters
        self.n_steps = n_steps
        self.initial_transition_rule = transition_rule
        self.p_common = p_common
        self.initial_block = block
        self.p_reversal = p_reversal
        self.block_len = block_len
        self.p_neutral = p_neutral

    #---------------------- gym.Env interface --------------------------------------
    def reset(self, seed=None, options=None):
        """Start a new episode: sample its transition rule and block, then wait for CENTER."""
        super().reset(seed=seed)  # seeds self.np_random
        self.step_n = 0
        self.trial = -1  # becomes 0 when the first trial is started

        # Transition rule: fixed, or random per episode
        if self.initial_transition_rule is None:
            self.transition_rule = int(self.np_random.integers(2))
        else:
            self.transition_rule = self.initial_transition_rule

        # Reward block: fixed, or a random good port per episode
        if self.initial_block is None:
            self.set_block(str(self.np_random.choice(["UP", "DOWN"])))
        else:
            self.set_block(self.initial_block)

        self.state = "NO-REWARD"
        self._clear_trial()
        return self.observation, self._info()

    def step(self, action):
        """Apply one action. Returns (observation, reward, terminated, truncated, info)."""
        action = self.ACTIONS[action]  # integer -> "UP" / "DOWN" / "LEFT" / "RIGHT" / "CENTER"
        reward = 0.0

        if self.state == "CHOICE":
            # First step: LEFT or RIGHT moves to a second-step port, common or rare
            if action in ("LEFT", "RIGHT"):
                self.choice = action
                self.common = bool(self.np_random.random() < self.p_common)
                common_port = self.TRANSITIONS[self.transition_rule][action]
                self.second_step = common_port if self.common else OTHER_PORT[common_port]
                self.state = f"{self.second_step}-ACTIVE"

        elif self.state in ("UP-ACTIVE", "DOWN-ACTIVE"):
            # Second step: poking the active port draws the reward, returned on this step
            if action == self.second_step:
                self.rewarded = bool(self.np_random.random() < self.reward_probs[action])
                reward = float(self.rewarded)
                self.state = f"REWARD-AT-{action}" if self.rewarded else "NO-REWARD"

        else:
            # Outcome (or episode start): CENTER starts the next trial
            if action == "CENTER":
                self._start_trial()

        # Any action not handled above leaves the state unchanged and earns nothing.
        # Episodes are cut off after n_steps (truncated, not terminated: the task itself goes on)
        self.step_n += 1
        truncated = self.step_n >= self.n_steps
        return self.observation, reward, False, truncated, self._info()

    # ----------------------- task logic -------------------------------------------

    def _start_trial(self):
        """Begin a trial: possibly switch block (not before the first trial), then show CHOICE."""
        self.trial += 1
        if self.trial > 0:
            if self.block_len is not None:
                self.trials_left_in_block -= 1
                if self.trials_left_in_block <= 0:
                    self.switch_block()
            elif self.np_random.random() < self.p_reversal:
                self.switch_block()
        self.state = "CHOICE"
        self._clear_trial()

    def _clear_trial(self):
        """Forget the previous trial's events (kept until now so they appear in info)."""
        self.choice = None
        self.common = None
        self.second_step = None
        self.rewarded = None

    def set_block(self, block):
        """Start a new reward block (can be called from outside for scheduled switches).

        Takes effect immediately, so call it between trials. Also restarts the block-length
        countdown when block_len is set.
        """
        if block not in self.BLOCKS:
            raise ValueError(f"block must be one of {list(self.BLOCKS)}, got {block}")
        self.block = block
        if self.block_len is not None:
            low, high = self.block_len
            self.trials_left_in_block = int(self.np_random.integers(low, high + 1))

    def switch_block(self):
        """Move to the next block: the other port (or NEUTRAL with p_neutral), or out of NEUTRAL."""
        if self.block == "NEUTRAL":
            self.set_block(str(self.np_random.choice(["UP", "DOWN"])))
        elif self.np_random.random() < self.p_neutral:
            self.set_block("NEUTRAL")
        else:
            self.set_block(OTHER_PORT[self.block])

    def reverse_transitions(self):
        """Swap which port each choice commonly leads to (for tasks with transition reversals)."""
        self.transition_rule = 1 - self.transition_rule

    @property
    def reward_probs(self):
        return self.BLOCKS[self.block]

    @property
    def correct_action(self):
        # The choice that commonly leads to the good port (None in a NEUTRAL block). Derived
        # from the rule and block, so it can never go out of sync after a switch.
        if self.block == "NEUTRAL":
            return None
        for choice, port in self.TRANSITIONS[self.transition_rule].items():
            if port == self.block:
                return choice

    @property
    def phase(self):
        if self.state == "CHOICE":
            return "CHOICE"
        if self.state in ("UP-ACTIVE", "DOWN-ACTIVE"):
            return "SECOND-STEP"
        return "OUTCOME"

    @property
    def observation(self):
        # The task state is fully observable apart from the hidden rule and block
        return self.OBSERVATIONS.index(self.state)

    def _info(self):
        # Ground truth for analysis; never shown to the agent
        return {
            "phase": self.phase,
            "trial": self.trial,
            "transition_rule": self.transition_rule,
            "block": self.block,
            "p_reward_up": self.reward_probs["UP"],
            "p_reward_down": self.reward_probs["DOWN"],
            "correct_action": self.correct_action,
            "choice": self.choice,
            "common": self.common,
            "second_step": self.second_step,
            "rewarded": self.rewarded,
        }


OTHER_PORT = {"UP": "DOWN", "DOWN": "UP"}

class Env:
    """An environment class which models a NxN grid which"""