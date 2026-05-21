# src/baselines/base.py

from abc import ABC, abstractmethod


class BaseAlgorithm(ABC):
    """
    Stable learning interface.
    Algorithms treat env as a black box.
    """

    def __init__(self, env, config: dict):
        self.env = env
        self.config = config

    @abstractmethod
    def select_actions(self, observations: dict) -> dict:
        """
        observations: {agent_name: obs_dict}
        returns: {agent_name: action_int}
        """
        pass

    @abstractmethod
    def train(self):
        pass

    def evaluate(self, episodes: int = 5):
        for _ in range(episodes):
            obs, _ = self.env.reset()
            done = False
            while not done:
                actions = self.select_actions(obs)
                step_out = self.env.step(actions)
                obs = step_out["obs"]
                done = step_out["terminated"] or step_out["truncated"]