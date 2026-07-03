# tests/test_baselines_dqn.py
"""
Tests for DQN (Deep Q-Network baseline).
"""

import os
import tempfile

import numpy as np
import pytest

from multi_agent_package.core.agent import Agent
from multi_agent_package.core.gridworld import GridWorldEnv
from multi_agent_package.observations.relative import RelativeObservation
from baselines.DQN.dqn import DQN
from baselines.DQN.q_network import QNetwork
from baselines.DQN.replay_buffer import ReplayBuffer


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def make_env(n_pred=1, n_prey=1, size=5, seed=0, perc_obstacle=0, max_steps=50):
    """
    Builds a small env wired with the 'relative' observation builder --
    the same one DQN._encode_state() is written to read from. max_steps
    is capped (default 50, not None) so that a badly-behaved untrained
    network can't make a test hang on an episode that never terminates.
    """
    agents = []
    for i in range(1, n_pred + 1):
        agents.append(Agent(agent_type="predator", agent_team=f"predator_{i}", agent_name=f"pred_{i}"))
    for i in range(1, n_prey + 1):
        agents.append(Agent(agent_type="prey", agent_team=f"prey_{i}", agent_name=f"prey_{i}"))

    env = GridWorldEnv(
        agents=agents, size=size, perc_num_obstacle=perc_obstacle,
        render_mode=None, seed=seed, max_steps=max_steps,
    )
    env.observation_builder = RelativeObservation(
        include_agents=True, include_obstacles=False, include_walls=False,
        distance_type="manhattan",
    ).build
    return env


def base_config(**overrides):
    cfg = {
        "hidden_layers": [8, 8],
        "activation": "relu",
        "learning_rate": 0.01,
        "gamma": 0.99,
        "epsilon": 0.5,
        "epsilon_decay": 1.0,
        "min_epsilon": 0.0,
        "action_dim": 5,
        "buffer_size": 200,
        "batch_size": 8,
        "min_buffer_size": 8,
        "target_update_freq": 1,
        "episodes": 3,
        "seed": 0,
        "normalize_obs": True,
    }
    cfg.update(overrides)
    return cfg


# ------------------------------------------------------------------
# QNetwork -- the building block, tested in isolation from the env
# ------------------------------------------------------------------

class TestQNetworkValidation:
    def test_rejects_non_positive_input_dim(self):
        with pytest.raises(ValueError):
            QNetwork(input_dim=0, hidden_layers=[8], output_dim=5)

    def test_rejects_non_positive_output_dim(self):
        with pytest.raises(ValueError):
            QNetwork(input_dim=2, hidden_layers=[8], output_dim=0)

    def test_rejects_empty_hidden_layers(self):
        with pytest.raises(ValueError):
            QNetwork(input_dim=2, hidden_layers=[], output_dim=5)

    def test_rejects_negative_hidden_layer_size(self):
        with pytest.raises(ValueError):
            QNetwork(input_dim=2, hidden_layers=[8, -4], output_dim=5)

    def test_predict_output_shape(self):
        net = QNetwork(input_dim=2, hidden_layers=[8, 8], output_dim=5, seed=0)
        out = net.predict(np.array([[0.1, 0.2]]))
        assert out.shape == (1, 5)

    def test_predict_rejects_wrong_input_width(self):
        net = QNetwork(input_dim=2, hidden_layers=[8], output_dim=5, seed=0)
        with pytest.raises(ValueError):
            net.predict(np.array([[0.1, 0.2, 0.3]]))  # 3 numbers, net expects 2

    def test_copy_weights_from_makes_predictions_match(self):
        net_a = QNetwork(input_dim=2, hidden_layers=[8], output_dim=4, seed=1)
        net_b = QNetwork(input_dim=2, hidden_layers=[8], output_dim=4, seed=2)
        x = np.array([[0.3, -0.7]], dtype=np.float32)
        before_a = net_a.predict(x)
        net_b.copy_weights_from(net_a)
        after_b = net_b.predict(x)
        np.testing.assert_allclose(after_b, before_a, atol=1e-5)

    def test_train_step_reduces_loss(self):
        """Sanity check that gradient descent is actually happening, and
        that the linear output layer CAN represent a negative target
        (Q-values must be allowed to go negative in this environment)."""
        net = QNetwork(input_dim=2, hidden_layers=[8], output_dim=3, learning_rate=0.05, seed=0)
        x = np.array([[0.5, -0.5]], dtype=np.float32)
        target = np.array([[1.0, -1.0, 0.0]], dtype=np.float32)
        first_loss = net.train_step(x, target)
        loss = first_loss
        for _ in range(50):
            loss = net.train_step(x, target)
        assert loss < first_loss


# ------------------------------------------------------------------
# ReplayBuffer -- this is where the deque-vs-array slowdown bug lived.
# test_oldest_overwritten_when_full is a direct regression test for
# the "ring buffer" forgetting behavior the fix depends on.
# ------------------------------------------------------------------

class TestReplayBuffer:
    def test_starts_empty(self):
        buf = ReplayBuffer(capacity=10, state_dim=2, seed=0)
        assert len(buf) == 0

    def test_push_increases_length(self):
        buf = ReplayBuffer(capacity=10, state_dim=2, seed=0)
        buf.push([0.0, 0.0], 1, 1.0, [0.1, 0.1], False)
        assert len(buf) == 1

    def test_length_caps_at_capacity(self):
        buf = ReplayBuffer(capacity=5, state_dim=2, seed=0)
        for i in range(20):
            buf.push([float(i), 0.0], 0, 0.0, [0.0, 0.0], False)
        assert len(buf) == 5

    def test_oldest_overwritten_when_full(self):
        """Regression test: pushing past capacity must overwrite the
        OLDEST entries first (ring buffer), not silently grow or drop
        the newest ones."""
        buf = ReplayBuffer(capacity=3, state_dim=1, seed=0)
        for i in range(5):
            buf.push([float(i)], 0, 0.0, [0.0], False)
        states, *_ = buf.sample(3)
        stored_values = sorted(states.flatten().tolist())
        assert stored_values == [2.0, 3.0, 4.0]  # 0 and 1 were evicted

    def test_sample_shapes(self):
        buf = ReplayBuffer(capacity=20, state_dim=2, seed=0)
        for i in range(10):
            buf.push([float(i), 0.0], i % 5, float(i), [float(i) + 1, 0.0], False)
        states, actions, rewards, next_states, dones = buf.sample(4)
        assert states.shape == (4, 2)
        assert actions.shape == (4,)
        assert rewards.shape == (4,)
        assert next_states.shape == (4, 2)
        assert dones.shape == (4,)

    def test_sample_more_than_available_raises(self):
        buf = ReplayBuffer(capacity=10, state_dim=2, seed=0)
        buf.push([0.0, 0.0], 0, 0.0, [0.0, 0.0], False)
        with pytest.raises(ValueError):
            buf.sample(5)

    def test_invalid_capacity_raises(self):
        with pytest.raises(ValueError):
            ReplayBuffer(capacity=0, state_dim=2)

    def test_invalid_state_dim_raises(self):
        with pytest.raises(ValueError):
            ReplayBuffer(capacity=10, state_dim=0)


# ------------------------------------------------------------------
# DQN initialization
# ------------------------------------------------------------------

class TestDQNInit:
    def test_agent_ids_discovered(self):
        env = make_env()
        algo = DQN(env, base_config())
        assert set(algo.agent_ids) == {"pred_1", "prey_1"}

    def test_input_dim_inferred_from_relative_obs(self):
        """With the 'relative' observation builder, each agent's state
        is the other agent's [dx, dy] -- length 2."""
        env = make_env()
        algo = DQN(env, base_config())
        assert algo.input_dim == 2

    def test_action_dim_matches_env_when_unset_in_config(self):
        env = make_env()
        algo = DQN(env, base_config())
        assert algo.action_dim == 5  # discrete_5: right/up/left/down/noop

    def test_q_networks_created_per_agent(self):
        env = make_env()
        algo = DQN(env, base_config())
        assert set(algo.q_networks.keys()) == {"pred_1", "prey_1"}

    def test_target_networks_created_per_agent(self):
        env = make_env()
        algo = DQN(env, base_config())
        assert set(algo.target_networks.keys()) == {"pred_1", "prey_1"}

    def test_buffers_created_per_agent(self):
        env = make_env()
        algo = DQN(env, base_config())
        assert set(algo.buffers.keys()) == {"pred_1", "prey_1"}

    def test_target_network_starts_synced_with_q_network(self):
        env = make_env()
        algo = DQN(env, base_config())
        for aid in algo.agent_ids:
            q_w = algo.q_networks[aid].get_weights()
            t_w = algo.target_networks[aid].get_weights()
            for qw, tw in zip(q_w, t_w):
                np.testing.assert_array_equal(qw, tw)

    def test_invalid_action_dim_raises(self):
        env = make_env()
        with pytest.raises(ValueError):
            DQN(env, base_config(action_dim=4))  # env actually has 5 actions

    def test_invalid_input_dim_raises(self):
        env = make_env()
        with pytest.raises(ValueError):
            DQN(env, base_config(input_dim=99))  # relative obs actually gives 2

    def test_buffer_smaller_than_batch_warns(self):
        env = make_env()
        with pytest.warns(UserWarning):
            DQN(env, base_config(buffer_size=4, batch_size=8))


# ------------------------------------------------------------------
# State encoding -- this is where the "blind predator" bug lived.
# These tests pin down exactly what _encode_state must produce for
# the relative observation builder.
# ------------------------------------------------------------------

class TestDQNEncodeState:
    def test_returns_length_two_vector(self):
        env = make_env()
        algo = DQN(env, base_config())
        obs, _ = env.reset()
        s = algo._encode_state(obs["pred_1"])
        assert s.shape == (2,)

    def test_missing_agents_key_returns_zeros(self):
        """Defensive fallback if an observation has no 'agents' entry."""
        env = make_env()
        algo = DQN(env, base_config())
        s = algo._encode_state({"local": {"pos": np.array([0, 0])}})
        np.testing.assert_array_equal(s, np.zeros(2))

    def test_normalization_scales_into_unit_range(self):
        env = make_env(size=5)
        algo = DQN(env, base_config(normalize_obs=True))
        obs = {"agents": {"prey_1": {"rel_pos": np.array([4, -4]), "dist": 8, "type": "prey"}}}
        s = algo._encode_state(obs)
        # size=5 -> grid_max=4 -> [4,-4] / 4 -> [1.0, -1.0]
        np.testing.assert_allclose(s, [1.0, -1.0])

    def test_normalization_off_keeps_raw_values(self):
        env = make_env(size=5)
        algo = DQN(env, base_config(normalize_obs=False))
        obs = {"agents": {"prey_1": {"rel_pos": np.array([3, -2]), "dist": 5, "type": "prey"}}}
        s = algo._encode_state(obs)
        np.testing.assert_allclose(s, [3.0, -2.0])


# ------------------------------------------------------------------
# Action selection
# ------------------------------------------------------------------

class TestDQNSelectActions:
    def test_returns_all_agents(self):
        env = make_env()
        algo = DQN(env, base_config())
        obs, _ = env.reset()
        actions = algo.select_actions(obs)
        assert set(actions.keys()) == {"pred_1", "prey_1"}

    def test_actions_in_valid_range(self):
        env = make_env()
        algo = DQN(env, base_config())
        obs, _ = env.reset()
        for _ in range(10):
            actions = algo.select_actions(obs)
            for a in actions.values():
                assert 0 <= a < 5

    def test_epsilon_one_gives_varied_random_actions(self):
        env = make_env()
        algo = DQN(env, base_config(epsilon=1.0, seed=0))
        obs, _ = env.reset()
        seen = set()
        for _ in range(50):
            actions = algo.select_actions(obs)
            seen.add(actions["pred_1"])
        assert len(seen) > 1

    def test_epsilon_zero_is_greedy(self):
        """Monkey-patch predict() to force a known 'best' action, the
        same trick test_baselines_iql.py uses on a Q-table instead."""
        env = make_env()
        algo = DQN(env, base_config(epsilon=0.0))
        obs, _ = env.reset()
        fake_q = np.array([[0.0, 0.0, 5.0, 0.0, 0.0]])  # action 2 is best
        algo.q_networks["pred_1"].predict = lambda x: fake_q
        actions = algo.select_actions(obs)
        assert actions["pred_1"] == 2


# ------------------------------------------------------------------
# Training loop
# ------------------------------------------------------------------

class TestDQNTrain:
    def test_buffers_fill_during_training(self):
        env = make_env()
        algo = DQN(env, base_config(episodes=3))
        algo.train()
        for aid in algo.agent_ids:
            assert len(algo.buffers[aid]) > 0

    def test_weights_change_after_training(self):
        env = make_env()
        algo = DQN(env, base_config(episodes=5))
        before = [w.copy() for w in algo.q_networks["pred_1"].get_weights()]
        algo.train()
        after = algo.q_networks["pred_1"].get_weights()
        changed = any(not np.allclose(b, a) for b, a in zip(before, after))
        assert changed

    def test_target_network_tracks_q_network_when_freq_one(self):
        env = make_env()
        algo = DQN(env, base_config(episodes=5, target_update_freq=1))
        algo.train()
        for aid in algo.agent_ids:
            q_w = algo.q_networks[aid].get_weights()
            t_w = algo.target_networks[aid].get_weights()
            for qw, tw in zip(q_w, t_w):
                np.testing.assert_allclose(qw, tw, atol=1e-5)

    def test_target_network_frozen_when_update_freq_is_huge(self):
        env = make_env()
        algo = DQN(env, base_config(episodes=5, target_update_freq=10_000_000))
        initial_target = [w.copy() for w in algo.target_networks["pred_1"].get_weights()]
        algo.train()
        final_target = algo.target_networks["pred_1"].get_weights()
        for iw, fw in zip(initial_target, final_target):
            np.testing.assert_array_equal(iw, fw)  # never synced -> unchanged

    def test_epsilon_decays(self):
        env = make_env()
        algo = DQN(env, base_config(epsilon=1.0, epsilon_decay=0.5, min_epsilon=0.0, episodes=5))
        algo.train()
        assert algo.epsilon < 1.0

    def test_epsilon_respects_floor(self):
        env = make_env()
        algo = DQN(env, base_config(epsilon=1.0, epsilon_decay=0.1, min_epsilon=0.2, episodes=20))
        algo.train()
        assert algo.epsilon >= 0.2


# ------------------------------------------------------------------
# Save / load
# ------------------------------------------------------------------

class TestDQNPersistence:
    def test_save_creates_file(self):
        env = make_env()
        algo = DQN(env, base_config(episodes=3))
        algo.train()
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            path = f.name
        try:
            algo.save(path)
            assert os.path.exists(path)
        finally:
            os.unlink(path)

    def test_load_restores_q_network_weights(self):
        env = make_env()
        algo = DQN(env, base_config(episodes=3))
        algo.train()
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            path = f.name
        try:
            algo.save(path)
            algo2 = DQN.load(make_env(), base_config(episodes=3), path)
            for aid in algo.agent_ids:
                w1 = algo.q_networks[aid].get_weights()
                w2 = algo2.q_networks[aid].get_weights()
                for a, b in zip(w1, w2):
                    np.testing.assert_allclose(a, b, atol=1e-6)
        finally:
            os.unlink(path)

    def test_load_also_syncs_target_network(self):
        env = make_env()
        algo = DQN(env, base_config(episodes=3))
        algo.train()
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            path = f.name
        try:
            algo.save(path)
            algo2 = DQN.load(make_env(), base_config(episodes=3), path)
            for aid in algo2.agent_ids:
                q_w = algo2.q_networks[aid].get_weights()
                t_w = algo2.target_networks[aid].get_weights()
                for a, b in zip(q_w, t_w):
                    np.testing.assert_allclose(a, b, atol=1e-6)
        finally:
            os.unlink(path)

    def test_load_instance_can_evaluate(self):
        env = make_env()
        algo = DQN(env, base_config(episodes=3))
        algo.train()
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            path = f.name
        try:
            algo.save(path)
            algo2 = DQN.load(make_env(), base_config(epsilon=0.0, episodes=3), path)
            algo2.evaluate(episodes=1)  # must not raise
        finally:
            os.unlink(path)