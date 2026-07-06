"""
Integration tests: config loading, environment building, and end-to-end training.
"""

import os
import sys
import pytest
from pathlib import Path

# Make src importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIGS_DIR = REPO_ROOT / "configs"


# ------------------------------------------------------------------
# Config loading
# ------------------------------------------------------------------

class TestLoadAllConfigs:
    def test_loads_five_sections(self):
        from multi_agent_package.scripts.run_from_config import load_all_configs
        configs = load_all_configs()
        assert set(configs.keys()) == {"env", "agents", "observations", "rewards", "experiment"}

    def test_env_section_has_size(self):
        from multi_agent_package.scripts.run_from_config import load_all_configs
        configs = load_all_configs()
        assert "size" in configs["env"]["env"]

    def test_agents_section_has_predators_and_preys(self):
        from multi_agent_package.scripts.run_from_config import load_all_configs
        configs = load_all_configs()
        assert "predators" in configs["agents"]["agents"]
        assert "preys" in configs["agents"]["agents"]

    def test_observations_has_type(self):
        from multi_agent_package.scripts.run_from_config import load_all_configs
        configs = load_all_configs()
        assert "type" in configs["observations"]["observations"]

    def test_rewards_has_base(self):
        from multi_agent_package.scripts.run_from_config import load_all_configs
        configs = load_all_configs()
        assert "base" in configs["rewards"]["rewards"]

    def test_experiment_has_algorithm(self):
        from multi_agent_package.scripts.run_from_config import load_all_configs
        configs = load_all_configs()
        assert "algorithm" in configs["experiment"]["experiment"]

    def test_algorithm_has_name(self):
        from multi_agent_package.scripts.run_from_config import load_all_configs
        configs = load_all_configs()
        assert "name" in configs["experiment"]["experiment"]["algorithm"]

    def test_iql_experiment_file(self):
        from multi_agent_package.scripts.run_from_config import load_all_configs
        configs = load_all_configs(experiment_file="experiment_iql.yaml")
        assert configs["experiment"]["experiment"]["algorithm"]["name"] == "iql"

    def test_cql_experiment_file(self):
        from multi_agent_package.scripts.run_from_config import load_all_configs
        configs = load_all_configs(experiment_file="experiment_cql.yaml")
        assert configs["experiment"]["experiment"]["algorithm"]["name"] == "cql"

    def test_mixed_experiment_file(self):
        from multi_agent_package.scripts.run_from_config import load_all_configs
        configs = load_all_configs(experiment_file="experiment_mixed.yaml")
        assert configs["experiment"]["experiment"]["algorithm"]["name"] == "mixed"


# ------------------------------------------------------------------
# build_agents
# ------------------------------------------------------------------

class TestBuildAgents:
    def test_correct_agent_count(self):
        from multi_agent_package.scripts.run_from_config import build_agents, load_all_configs
        configs = load_all_configs()
        agents = build_agents(configs["agents"])
        pred_count = configs["agents"]["agents"]["predators"]["count"]
        prey_count = configs["agents"]["agents"]["preys"]["count"]
        assert len(agents) == pred_count + prey_count

    def test_agent_types_correct(self):
        from multi_agent_package.scripts.run_from_config import build_agents, load_all_configs
        configs = load_all_configs()
        agents = build_agents(configs["agents"])
        pred_count = configs["agents"]["agents"]["predators"]["count"]
        types = [a.agent_type for a in agents]
        assert types[:pred_count] == ["predator"] * pred_count

    def test_agent_names_unique(self):
        from multi_agent_package.scripts.run_from_config import build_agents, load_all_configs
        configs = load_all_configs()
        agents = build_agents(configs["agents"])
        names = [a.agent_name for a in agents]
        assert len(names) == len(set(names))


# ------------------------------------------------------------------
# build_environment
# ------------------------------------------------------------------

class TestBuildEnvironment:
    def _load_and_build(self, experiment_file="experiment_iql.yaml"):
        from multi_agent_package.scripts.run_from_config import load_all_configs, build_environment
        # Override render_mode to None so no display is needed
        configs = load_all_configs(experiment_file=experiment_file)
        configs["env"]["env"]["render_mode"] = None
        return build_environment(configs)

    def test_returns_gridworld_env(self):
        from multi_agent_package.core.gridworld import GridWorldEnv
        env = self._load_and_build()
        assert isinstance(env, GridWorldEnv)

    def test_observation_builder_attached(self):
        env = self._load_and_build()
        assert env.observation_builder is not None
        assert callable(env.observation_builder)

    def test_reward_fn_attached(self):
        env = self._load_and_build()
        assert env.reward_fn is not None
        assert callable(env.reward_fn)

    def test_env_resets_successfully(self):
        env = self._load_and_build()
        obs, info = env.reset()
        assert isinstance(obs, dict)
        assert len(obs) > 0

    def test_env_steps_successfully(self):
        env = self._load_and_build()
        obs, _ = env.reset()
        actions = {aid: 4 for aid in obs}
        out = env.step(actions)
        assert "obs" in out
        assert "reward" in out


# ------------------------------------------------------------------
# End-to-end training: IQL
# ------------------------------------------------------------------

class TestEndToEndIQL:
    def test_iql_trains_without_error(self):
        from multi_agent_package.scripts.run_from_config import load_all_configs, build_environment
        import baselines
        from baselines.IQL.iql import IQL

        configs = load_all_configs(experiment_file="experiment_iql.yaml")
        configs["env"]["env"]["render_mode"] = None
        env = build_environment(configs)

        params = configs["experiment"]["experiment"]["algorithm"].get("params", {})
        params["episodes"] = 3  # keep test fast

        algo = IQL(env, params)
        algo.train()

        for table in algo.q_tables.values():
            assert len(table) > 0


# ------------------------------------------------------------------
# End-to-end training: CQL
# ------------------------------------------------------------------

class TestEndToEndCQL:
    def test_cql_trains_without_error(self):
        from multi_agent_package.scripts.run_from_config import load_all_configs, build_environment
        import baselines
        from baselines.CQL.cql import CQL

        configs = load_all_configs(experiment_file="experiment_cql.yaml")
        configs["env"]["env"]["render_mode"] = None
        env = build_environment(configs)

        params = configs["experiment"]["experiment"]["algorithm"].get("params", {})
        params["episodes"] = 3

        algo = CQL(env, params)
        algo.train()
        assert len(algo.q_table) > 0


# ------------------------------------------------------------------
# End-to-end training: Mixed
# ------------------------------------------------------------------

class TestEndToEndMixed:
    def test_mixed_trains_without_error(self):
        from multi_agent_package.scripts.run_from_config import load_all_configs, build_environment
        import baselines
        from baselines.MIXED.mix_train import MixedTrainer

        configs = load_all_configs(experiment_file="experiment_mixed.yaml")
        configs["env"]["env"]["render_mode"] = None
        env = build_environment(configs)

        params = configs["experiment"]["experiment"]["algorithm"].get("params", {})
        params["episodes"] = 3

        algo = MixedTrainer(env, params)
        algo.train()

        # At least one table must be populated
        all_tables = list(algo._iql_tables.values()) + list(algo._cql_tables.values())
        assert any(len(t) > 0 for t in all_tables)



# ------------------------------------------------------------------
# End-to-end training: DQN
# ------------------------------------------------------------------


class TestEndToEndDQN:
    def test_dqn_trains_without_error(self):
        from multi_agent_package.scripts.run_from_config import load_all_configs, build_environment
        import baselines
        from baselines.DQN.dqn import DQN

        configs = load_all_configs(experiment_file="experiment_dqn.yaml")
        configs["env"]["env"]["render_mode"] = None
        env = build_environment(configs)

        params = configs["experiment"]["experiment"]["algorithm"].get("params", {})
        params["episodes"] = 2
        params["min_replay_size"] = params.get("min_buffer_size", 8)

        algo = DQN(env, params)
        algo.train()

        for aid in algo.agent_ids:
            assert len(algo.buffers[aid]) > 0