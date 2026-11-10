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
    """A two-step task environment which models a two-step decision-making task."""
    def __init__(self):
        super().__init__()
        # Define the action and observation spaces
        pass

class Env:
    """An environment class which models a NxN grid which"""