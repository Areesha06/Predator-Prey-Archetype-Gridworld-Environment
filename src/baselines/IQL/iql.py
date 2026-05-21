# src/baselines/iql/iql.py

import numpy as np
from collections import defaultdict
from numpy.random import default_rng

from baselines.base import BaseAlgorithm
from baselines.registry.algorithm_registry import register


class IQL(BaseAlgorithm):
    """
    Tabular Independent Q-Learning.

    Properties:
    - One Q-table per agent
    - No access to core internals
    - Fully wrapper-compatible
    - State encoding derived only from observation payload
    """

    def __init__(self, env, config):
        super().__init__(env, config)

        # Hyperparameters
        self.alpha = config.get("alpha", 0.1)
        self.gamma = config.get("gamma", 0.99)
        self.epsilon = config.get("epsilon", 0.1)
        self.episodes = config.get("episodes", 500)
        self.epsilon_decay = config.get("epsilon_decay", 1.0)
        self.min_epsilon = config.get("min_epsilon", 0.01)

        # seeded RNG for reproducible exploration
        self.rng = default_rng(config.get("seed", None))

        # Initialize once to discover agent IDs
        initial_obs, _ = self.env.reset()
        self.agent_ids = list(initial_obs.keys())

        self.action_dim = config.get("action_dim", 5)

        # Q-tables
        self.q_tables = {
            agent_id: defaultdict(lambda: np.zeros(self.action_dim))
            for agent_id in self.agent_ids
        }

    # -------------------------------------------------
    # State Encoding (Wrapper Safe)
    # -------------------------------------------------
    def _encode_state(self, obs_dict):
        """
        Convert arbitrary observation dict into hashable state.
        Fully wrapper-compatible.
        """

        def recursive_convert(obj):
            if isinstance(obj, dict):
                return tuple(
                    sorted((k, recursive_convert(v)) for k, v in obj.items())
                )
            if isinstance(obj, (list, tuple)):
                return tuple(recursive_convert(v) for v in obj)
            if hasattr(obj, "tolist"):
                return tuple(obj.tolist())
            return obj

        return recursive_convert(obs_dict)

    # -------------------------------------------------
    # Action Selection
    # -------------------------------------------------
    def select_actions(self, observations):
        actions = {}

        for agent_id, obs in observations.items():
            state = self._encode_state(obs)

            if self.rng.random() < self.epsilon:
                action = int(self.rng.integers(self.action_dim))
            else:
                q_vals = self.q_tables[agent_id][state]
                action = int(np.argmax(q_vals))

            actions[agent_id] = action

        return actions

    # -------------------------------------------------
    # Training Loop
    # -------------------------------------------------
    def train(self):
        for ep in range(self.episodes):

            obs, _ = self.env.reset()
            done = False

            while not done:
                actions = self.select_actions(obs)

                step_out = self.env.step(actions)

                next_obs = step_out["obs"]
                rewards = step_out["reward"]
                done = step_out["terminated"] or step_out["truncated"]

                # Independent Q-updates
                for agent_id in self.agent_ids:
                    s = self._encode_state(obs[agent_id])
                    a = actions[agent_id]
                    r = rewards[agent_id]
                    s_next = self._encode_state(next_obs[agent_id])

                    q_current = self.q_tables[agent_id][s][a]
                    q_next_max = (
                        0.0 if done else np.max(self.q_tables[agent_id][s_next])
                    )

                    td_target = r + self.gamma * q_next_max
                    td_error = td_target - q_current

                    self.q_tables[agent_id][s][a] += (
                        self.alpha * td_error
                    )

                obs = next_obs

            # Epsilon decay
            self.epsilon = max(
                self.min_epsilon,
                self.epsilon * self.epsilon_decay
            )


register("iql", IQL)