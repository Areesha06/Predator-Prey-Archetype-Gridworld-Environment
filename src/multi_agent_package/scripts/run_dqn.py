# src/multi_agent_package/scripts/run_dqn.py
"""
Train or evaluate DQN using configs/experiment_dqn.yaml.

Usage:
    cd src
    python -m multi_agent_package.scripts.run_dqn                      # train
    python -m multi_agent_package.scripts.run_dqn --mode eval \
        --load-path trained_dqn.pkl
"""

import argparse
import logging

import baselines  # noqa: F401 -- triggers auto-registration
from baselines.DQN.dqn import DQN
from multi_agent_package.scripts.run_from_config import (
    load_all_configs,
    build_environment,
)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%d-%m-%Y %H:%M:%S",
)

EXPERIMENT_FILE = "experiment_dqn.yaml"


def main():
    p = argparse.ArgumentParser("Run DQN experiment")
    p.add_argument("--mode", choices=["train", "eval"], default="train")
    p.add_argument("--config-dir", default="configs")
    p.add_argument("--save-path", default="trained_dqn.pkl")
    p.add_argument("--load-path", default=None)
    p.add_argument("--render", action="store_true")
    args = p.parse_args()

    configs = load_all_configs(args.config_dir, EXPERIMENT_FILE)

    if args.mode == "eval":
        configs["env"]["env"]["render_mode"] = "human" if args.render else None

    env = build_environment(configs) #reads YAMLs and wires observation_builder, reward_fn, and action_space_plugin onto the env
    algo_params = configs["experiment"]["experiment"]["algorithm"].get("params", {})

    if args.mode == "train":
        algo = DQN(env, algo_params)
        algo.train()
        algo.save(args.save_path)
    else:
        if not args.load_path:
            raise SystemExit("--load-path is required for --mode eval")
        algo = DQN.load(env, algo_params, args.load_path)
        algo.evaluate()

    env.close()


if __name__ == "__main__":
    main()


"""
Relative observation
(dx, dy)
        │
        ▼
_encode_state()
        │
        ▼
QNetwork.predict()
        │
        ▼
[Q(up), Q(down), Q(left), Q(right), Q(noop)]
        │
        ▼
Choose the action with the highest Q-value (unless ε-greedy explores)
        │
        ▼
Environment executes the action
        │
        ▼
Reward + next observation
        │
        ▼
Replay Buffer stores the transition
        │
        ▼
DQN computes target Q-values
        │
        ▼
QNetwork.train_step()
        │
        ▼
TensorFlow performs forward pass → computes loss → automatically computes gradients with GradientTape → updates all weights using Adam
"""