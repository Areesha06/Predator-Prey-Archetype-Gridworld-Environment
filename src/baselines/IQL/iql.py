# src/baselines/iql/iql.py

import numpy as np
from collections import defaultdict

from baselines.base import BaseAlgorithm
from baselines.registry.algorithm_registry import register


class IQL(BaseAlgorithm):
    """
    Tabular Independent Q-Learning.
    One Q-table per agent.
    """

    def __init__(self, env, config):
        super().__init__(env, config)

        self.alpha = config.get("alpha", 0.1)
        self.gamma = config.get("gamma", 0.99)
        self.epsilon = config.get("epsilon", 0.1)
        self.episodes = config.get("episodes", 500)

        # Q-tables: {agent_name: {state_key: np.array(action_values)}}
        self.q_tables = {
            ag.agent_name: defaultdict(
                lambda: np.zeros(ag.action_space.n)
            )
            for ag in self.env.agents
        }

    # ----------------------------
    # State Encoding
    # ----------------------------
    def _encode_state(self, obs_dict):
        """
        Minimal tabular encoding:
        Uses only local position.
        """
        return tuple(obs_dict["local"])

    # ----------------------------
    # Action Selection
    # ----------------------------
    def select_actions(self, observations):
        actions = {}

        for agent_name, obs in observations.items():
            state = self._encode_state(obs)

            if np.random.rand() < self.epsilon:
                # explore
                action = np.random.randint(
                    self.env.action_space.n
                )
            else:
                q_vals = self.q_tables[agent_name][state]
                action = int(np.argmax(q_vals))

            actions[agent_name] = action

        return actions

    # ----------------------------
    # Training Loop
    # ----------------------------
    def train(self):
        for ep in range(self.episodes):
            obs, _ = self.env.reset()
            done = False

            while not done:
                actions = self.select_actions(obs)

                step_out = self.env.step(actions)

                next_obs = step_out["obs"]
                rewards = step_out["reward"]
                done = step_out["terminated"] or step_out["trunc"]

                # Q updates per agent
                for agent_name in obs.keys():
                    s = self._encode_state(obs[agent_name])
                    a = actions[agent_name]
                    r = rewards[agent_name]
                    s_next = self._encode_state(next_obs[agent_name])

                    q_current = self.q_tables[agent_name][s][a]
                    q_next_max = np.max(self.q_tables[agent_name][s_next])

                    td_target = r + self.gamma * q_next_max
                    td_error = td_target - q_current

                    self.q_tables[agent_name][s][a] += (
                        self.alpha * td_error
                    )

                obs = next_obs


register("iql", IQL)