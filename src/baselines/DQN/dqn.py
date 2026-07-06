# src/baselines/DQN/dqn.py
"""
Modular, parameterizable Deep Q-Network baseline (TensorFlow/Keras for
the neural net, numpy for everything else).

One independent Q-network + target network + replay buffer is created
PER AGENT -- mirrors IQL's "independent learning" pattern, with a neural
net standing in for the Q-table.
"""

import logging
import os
import pickle
import warnings

import numpy as np
from numpy.random import default_rng

from baselines.base import BaseAlgorithm
from baselines.registry.algorithm_registry import register
from baselines.DQN.q_network import QNetwork
from baselines.DQN.replay_buffer import ReplayBuffer
from collections import deque

LOGGER = logging.getLogger("dqn")


class DQN(BaseAlgorithm):
    def __init__(self, env, config: dict):
        super().__init__(env, config)

        # ---- hyperparameters from config --------------------------------

        self.gamma = float(config.get("gamma", 0.99))
        self.epsilon = float(config.get("epsilon", 1.0))
        self.epsilon_decay = float(config.get("epsilon_decay", 0.995))
        self.min_epsilon = float(config.get("min_epsilon", 0.05))
        self.episodes = int(config.get("episodes", 500))
        self.batch_size = int(config.get("batch_size", 32))
        self.buffer_size = int(config.get("buffer_size", 10_000))
        self.min_buffer_size = int(config.get("min_buffer_size", self.batch_size))
        self.target_update_freq = int(config.get("target_update_freq", 100))
        self.learning_rate = float(config.get("learning_rate", 0.001))
        self.hidden_layers = list(config.get("hidden_layers", [64, 64]))
        self.activation = config.get("activation", "relu")
        self.action_dim = int(config.get("action_dim", 5))
        self.normalize_obs = bool(config.get("normalize_obs", True))
        self.grad_clip = float(config.get("grad_clip", 10.0))
        # self._recent_capture = deque(maxlen=50)
        self._recent_outcomes = deque(maxlen=50)

        configured_input_dim = config.get("input_dim", None)
        self.rng = default_rng(config.get("seed", None))

        # ---- discover agents from the environment (same trick as IQL) ---

        initial_obs, _ = self.env.reset()
        self.agent_ids = list(initial_obs.keys())

        # ---- infer / validate the input dimension ------------------------

        inferred_input_dim = len(self._encode_state(initial_obs[self.agent_ids[0]]))
        if configured_input_dim is None:
            self.input_dim = inferred_input_dim
        else:
            self.input_dim = int(configured_input_dim)
            if self.input_dim != inferred_input_dim:
                raise ValueError(
                    "DQN config error: 'input_dim' does not match what the "
                    f"observation builder actually produces. Config says "
                    f"input_dim={self.input_dim}, but encoding the current "
                    f"observation gives a vector of length {inferred_input_dim}. "
                    "Fix 'input_dim' in your experiment YAML, or remove it "
                    "entirely so it gets inferred automatically."
                )

        # ---- validate action_dim against the environment's action space -

        plugin = getattr(env, "action_space_plugin", None)
        env_n_actions = plugin.n_actions if plugin is not None else env.action_space.n
        if self.action_dim != env_n_actions:
            raise ValueError(
                f"DQN config error: 'action_dim'={self.action_dim} does not "
                f"match the environment's action space size ({env_n_actions})."
            )

        if self.buffer_size < self.batch_size:
            warnings.warn(
                f"buffer_size ({self.buffer_size}) is smaller than batch_size "
                f"({self.batch_size}) -- training will never fill a full batch.",
                stacklevel=2,
            )

        # ---- one Q-network + target network + buffer PER AGENT -----------

        self.q_networks = {}
        self.target_networks = {}
        self.buffers = {}
        self._train_step_counts = {}

        for i, aid in enumerate(self.agent_ids):
            net_seed = None if config.get("seed") is None else config["seed"] + i #gives each agent's network a different but reproducible starting point — if both agents got the exact same seed, they'd start with identical weights,

            self.q_networks[aid] = QNetwork(
                input_dim=self.input_dim, hidden_layers=self.hidden_layers,
                output_dim=self.action_dim, activation=self.activation,
                learning_rate=self.learning_rate, grad_clip=self.grad_clip,
                seed=net_seed,
            )
            self.target_networks[aid] = QNetwork(
                input_dim=self.input_dim, hidden_layers=self.hidden_layers,
                output_dim=self.action_dim, activation=self.activation,
                learning_rate=self.learning_rate, grad_clip=self.grad_clip,
                seed=net_seed,
            )
            # target net starts as an exact copy of the live network

            self.target_networks[aid].copy_weights_from(self.q_networks[aid])

            self.buffers[aid] = ReplayBuffer(self.buffer_size, state_dim=self.input_dim, seed=net_seed)
            self._train_step_counts[aid] = 0

    # ------------------------------------------------------------------
    # def _encode_state(self, obs: dict) -> np.ndarray:
    #     """Built for the 'local_only' observation: {"local": [x, y]}."""
    #     local = np.asarray(obs["local"], dtype=np.float64)
    #     if self.normalize_obs:
    #         grid_max = max(self.env.size - 1, 1)
    #         local = local / grid_max
    #     return local


    # obs dict = fixed-length numpy vector the network can read
    def _encode_state(self, obs: dict) -> np.ndarray:
        """
        Built for the 'relative' observation builder. Each agent sees the
        OTHER agent's position relative to itself (egocentric: own position
        is always [0,0]). This is the minimum information that makes
        "chase" or "flee" an actually learnable behavior -- the previous
        'local_only' encoding hid the opponent entirely, so the network had
        no way to learn anything causally connected to the reward.
        """
        agents_seen = obs.get("agents", {})
        if not agents_seen:
            return np.zeros(2, dtype=np.float64)
        other_info = next(iter(agents_seen.values()))   # 1v1 -> exactly one entry
        rel_pos = np.asarray(other_info["rel_pos"], dtype=np.float64)
        if self.normalize_obs:
            grid_max = max(self.env.size - 1, 1)
            rel_pos = rel_pos / grid_max   # squash into roughly [-1, 1]
        return rel_pos

    # ------------------------------------------------------------------
    def select_actions(self, observations: dict) -> dict:
        actions = {}
        for aid, obs in observations.items():
            state = self._encode_state(obs)
            if self.rng.random() < self.epsilon:
                actions[aid] = int(self.rng.integers(self.action_dim))
            else:
                q_values = self.q_networks[aid].predict(state[None, :])[0]
                actions[aid] = int(np.argmax(q_values))
        return actions

    # ------------------------------------------------------------------
    # compute the *targets* ourselves (that's the DQN algorithm, not the
    # network library) but the actual weight update is one TF call
    # ------------------------------------------------------------------
    def _learn(self, agent_id: str) -> None:
        buf = self.buffers[agent_id]
        if len(buf) < max(self.batch_size, self.min_buffer_size):
            return # not enough experience collected yet

        states, actions, rewards, next_states, dones = buf.sample(self.batch_size)

        q_pred = self.q_networks[agent_id].predict(states)             # (B, A)
        q_next = self.target_networks[agent_id].predict(next_states)   # (B, A)
        q_next_max = np.max(q_next, axis=1)                            # (B,)


        # "targets" = what we WISH the network had predicted

        targets = q_pred.copy()
        for row in range(self.batch_size):
            bootstrap = 0.0 if dones[row] else self.gamma * q_next_max[row]
            targets[row, actions[row]] = rewards[row] + bootstrap

        self.q_networks[agent_id].train_step(states, targets)  # TF autodiff does the rest

        self._train_step_counts[agent_id] += 1
        if self._train_step_counts[agent_id] % self.target_update_freq == 0:
            self.target_networks[agent_id].copy_weights_from(self.q_networks[agent_id])

        # ------------------------------------------------------------------

    def train(self):
        for ep in range(self.episodes):
            obs, _ = self.env.reset()
            done = False
            ep_reward = {aid: 0.0 for aid in self.agent_ids}
            step_out = None  # will hold the LAST step's result once the loop ends

            while not done:
                actions = self.select_actions(obs)
                step_out = self.env.step(actions)
                next_obs = step_out["obs"]
                rewards = step_out["reward"]
                done = step_out["terminated"] or step_out["truncated"]

                for agent_id in self.agent_ids:
                    s = self._encode_state(obs[agent_id])
                    s_next = self._encode_state(next_obs[agent_id])
                    r = float(rewards[agent_id])
                    ep_reward[agent_id] += r
                    self.buffers[agent_id].push(s, actions[agent_id], r, s_next, done)
                    self._learn(agent_id)

                obs = next_obs
            # loop has exited here. step_out is the FINAL step of this episode.
            # Append EXACTLY ONCE per episode, not once per step:
            self._recent_outcomes.append(1 if step_out["terminated"] else 0)

            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

            if (ep + 1) % 50 == 0:
                reward_str = ", ".join(f"{k}={v:.2f}" for k, v in ep_reward.items())
                LOGGER.info("Episode %d/%d | eps=%.3f | %s", ep + 1, self.episodes, self.epsilon, reward_str)
                capture_rate = 100 * sum(self._recent_outcomes) / len(self._recent_outcomes)
                LOGGER.info("  Capture rate (last %d episodes): %.0f%%", len(self._recent_outcomes), capture_rate)

    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        payload = {
            "agent_ids": self.agent_ids,
            "weights": {aid: net.get_weights() for aid, net in self.q_networks.items()},
        }
        with open(path, "wb") as fh:
            pickle.dump(payload, fh)
        LOGGER.info("Saved DQN weights -> %s", path)

    @classmethod
    def load(cls, env, config: dict, path: str) -> "DQN":
        instance = cls(env, config)
        with open(path, "rb") as fh:
            payload = pickle.load(fh)
        for agent_id, weights in payload["weights"].items():
            if agent_id in instance.q_networks:
                instance.q_networks[agent_id].set_weights(weights)
                instance.target_networks[agent_id].set_weights(weights)
        LOGGER.info("Loaded DQN weights from %s", path)
        return instance


if __name__ != "__main__":
    register("dqn", DQN)




# ------------------------------------------------------------------
# Standalone CLI -- python -m baselines.DQN.dqn [--mode train|eval]
# (mirrors IQL/CQL's own CLI for quick manual testing)
# ------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    from multi_agent_package.core.gridworld import GridWorldEnv
    from multi_agent_package.core.agent import Agent
    from multi_agent_package.observations.local_only import LocalOnlyObservation

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")

    p = argparse.ArgumentParser("Numpy DQN — train or evaluate")
    p.add_argument("--mode", choices=["train", "eval"], default="train")
    p.add_argument("--episodes", type=int, default=1000)
    p.add_argument("--size", type=int, default=8)
    p.add_argument("--hidden-layers", type=int, nargs="+", default=[64, 64])
    p.add_argument("--learning-rate", type=float, default=0.001)
    p.add_argument("--buffer-size", type=int, default=10000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save-path", type=str, default="trained_dqn.pkl")
    p.add_argument("--load-path", type=str, default=None)
    args = p.parse_args()

    agents = [
        Agent(agent_type="predator", agent_team="predator_1", agent_name="predator_1"),
        Agent(agent_type="prey",     agent_team="prey_1",     agent_name="prey_1"),
    ]
    env = GridWorldEnv(agents=agents, size=args.size, perc_num_obstacle=0,
                       render_mode=None, seed=args.seed)
    env.observation_builder = LocalOnlyObservation().build  # simplest obs

    config = {
        "hidden_layers": args.hidden_layers,
        "learning_rate": args.learning_rate,
        "buffer_size": args.buffer_size,
        "batch_size": args.batch_size,
        "episodes": args.episodes,
        "epsilon": 1.0 if args.mode == "train" else 0.0, #during evaluation you want the agent to always pick its best-known action, not explore randomly.
        "action_dim": 5,
        "seed": args.seed,
    }

    if args.mode == "train":
        algo = DQN(env, config)
        algo.train()
        algo.save(args.save_path)
    else:
        if not args.load_path:
            raise SystemExit("--load-path is required for --mode eval")
        algo = DQN.load(env, config, args.load_path)
        algo.evaluate(episodes=args.episodes)

    env.close()