# src/baselines/cql/cql.py

import numpy as np

from baselines.iql.iql import IQL
from baselines.registry.algorithm_registry import register


class CQL(IQL):
    """
    Tabular Conservative Q-Learning.
    Extends IQL with pessimistic regularization.
    """

    def __init__(self, env, config):
        super().__init__(env, config)
        self.cql_alpha = config.get("cql_alpha", 0.1)

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

                for agent_name in obs.keys():
                    s = self._encode_state(obs[agent_name])
                    a = actions[agent_name]
                    r = rewards[agent_name]
                    s_next = self._encode_state(next_obs[agent_name])

                    q_vals = self.q_tables[agent_name][s]

                    q_current = q_vals[a]
                    q_next_max = np.max(
                        self.q_tables[agent_name][s_next]
                    )

                    td_target = r + self.gamma * q_next_max
                    td_error = td_target - q_current

                    # Conservative penalty
                    logsumexp = np.log(
                        np.sum(np.exp(q_vals))
                    )
                    penalty = logsumexp - q_current

                    update = td_error - self.cql_alpha * penalty

                    self.q_tables[agent_name][s][a] += (
                        self.alpha * update
                    )

                obs = next_obs


register("cql", CQL)