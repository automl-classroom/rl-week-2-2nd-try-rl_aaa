from __future__ import annotations

import gymnasium as gym
import numpy as np


# ------------- TODO: Implement the following environment -------------
class MyEnv(gym.Env):
    """
    Simple 2-state, 2-action environment with deterministic transitions.

    Actions
    -------
    Discrete(2):
    - 0: move to state 0
    - 1: move to state 1

    Observations
    ------------
    Discrete(2): the current state (0 or 1)

    Reward
    ------
    Equal to the action taken.

    Start/Reset State
    -----------------
    Always starts in state 0.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, seed: int | None = None):
        """Initializes the observation and action space for the environment."""
        self.rng = np.random.default_rng(seed)

        self.rewards = [0, 1]
        self.horizon = 10
        self.curent_steps = 0
        self.position = 0

        self.observation_space = gym.spaces.Discrete(2)
        self.action_space = gym.spaces.Discrete(2)

        self.states = np.array([0, 1])
        self.actions = np.array([0, 1])


    def reset(self, seed: int | None = None):
        """Resets the environment to the initial state."""
        self.curent_steps = 0
        self.position = 0
        return self.position, {}

    def step(self, action: int):

        action = int(action)
        if not self.action_space.contains(action):
            raise RuntimeError(f"{action} is not a valid action (needs to be 0 or 1)")

        self.curent_steps += 1
        self.position = action
        reward = float(self.rewards[self.position])
        terminated = False
        truncated = self.curent_steps >= self.horizon

        return self.position, reward, terminated, truncated, {}

    def get_reward_per_action(self) -> np.ndarray:
        """Returns the reward matrix for each action."""
        nS, nA = self.observation_space.n, self.action_space.n
        R = np.zeros((nS, nA), dtype=float)
        for s in range(nS):
            for a in range(nA):
                #nxt = max(0, min(nS - 1, s + (-1 if a == 0 else 1)))
                R[s, a] = self.rewards[a]
        return R


    def get_transition_matrix(self) -> np.ndarray:
        nS, nA = len(S), len(A)
        T = np.zeros((nS, nA, nS), dtype=float)
        for s in S:
            for a in A:
                s_next = max(0, min(nS - 1, s + (-1 if a == 0 else 1)))
                T[s, a, s_next] = 1.0 
        return T

    def render(self, mode="human"):
        """Renders the environment."""
        """
        Render the current state of the environment.

        Parameters
        ----------
        mode : str
            Render mode (only "human" is supported).
        """
        print(f"[MarsRover] pos={self.position}, steps={self.current_steps}")

       


class PartialObsWrapper(gym.Wrapper):
    """Wrapper that makes the underlying env partially observable by injecting
    observation noise: with probability `noise`, the true state is replaced by
    a random (incorrect) observation.

    Parameters
    ----------
    env : gym.Env
        The fully observable base environment.
    noise : float, default=0.1
        Probability in [0,1] of seeing a random wrong observation instead
        of the true one.
    seed : int | None, default=None
        Optional RNG seed for reproducibility.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, env: gym.Env, noise: float = 0.1, seed: int | None = None):
        pass
