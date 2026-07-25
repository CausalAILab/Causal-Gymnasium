import math

import pytest

from causal_gym.envs import DTRPCH, MABPCH, MDPPCH
from causal_gym.envs.dtr import DTRSCM
from causal_gym.envs.mab import MABSCM


def _set_full_confounding_context(env, u):
    env.u = u
    env._reward_u = u


@pytest.mark.parametrize(
    ("u", "action", "expected_reward"),
    [
        (0.20, 0, 1),
        (0.20, 1, 1),
        (0.35, 0, 1),
        (0.35, 1, 0),
        (0.40, 0, 0),
        (0.30, 1, 0),
    ],
)
def test_mab_full_confounding_uses_chapter_7_structural_equation(
    u, action, expected_reward
):
    env = MABSCM(confounding_strength=1.0, arms_probs=[0.4, 0.3])
    env.reset(seed=0)
    _set_full_confounding_context(env, u)

    _, reward, _, _, _ = env.step(action)

    assert reward == expected_reward


def test_mab_chapter_7_probabilities_on_exact_grid():
    env = MABSCM(confounding_strength=1.0, arms_probs=[0.4, 0.3])
    env.reset(seed=0)
    samples = 1000
    observational = []
    do_zero = []
    do_one = []

    for index in range(samples):
        u = (index + 0.5) / samples
        _set_full_confounding_context(env, u)
        natural_action = env.action()
        _, natural_reward, _, _, _ = env.step(natural_action)
        observational.append((natural_action, natural_reward))

        _set_full_confounding_context(env, u)
        do_zero.append(env.step(0)[1])
        _set_full_confounding_context(env, u)
        do_one.append(env.step(1)[1])

    action_zero = [reward for action, reward in observational if action == 0]
    action_one = [reward for action, reward in observational if action == 1]

    assert len(action_zero) / samples == pytest.approx(0.2)
    assert len(action_one) / samples == pytest.approx(0.8)
    assert sum(reward for _, reward in observational) / samples == pytest.approx(0.3)
    assert sum(action_zero) / len(action_zero) == pytest.approx(0.0)
    assert sum(action_one) / len(action_one) == pytest.approx(0.375)
    assert sum(do_zero) / samples == pytest.approx(0.4)
    assert sum(do_one) / samples == pytest.approx(0.3)
    assert (sum(do_zero) + sum(do_one)) / (2 * samples) == pytest.approx(0.35)


def test_mab_counterfactual_reuses_the_same_exogenous_context():
    env = MABPCH(confounding_strength=1.0, arms_probs=[0.4, 0.3])
    env.reset(seed=0)
    _set_full_confounding_context(env.env, 0.35)

    _, reward, _, _, info = env.ctf_do(
        lambda observation, natural_action: 0
    )

    assert info["natural_action"] == 1
    assert info["action"] == 0
    assert reward == 1


def test_mab_unconfounded_arm_probabilities_are_preserved():
    env = MABPCH(confounding_strength=0.0, arms_probs=[1.0, 0.0])

    env.reset(seed=0)
    assert env.do(lambda observation: 0)[1] == 1
    env.reset(seed=0)
    assert env.do(lambda observation: 1)[1] == 0


@pytest.mark.parametrize("strength", [-0.1, 1.1])
def test_mab_rejects_invalid_confounding_strength(strength):
    with pytest.raises(ValueError, match="between 0 and 1"):
        MABSCM(confounding_strength=strength)


def test_mdp_counterfactual_reuses_sampled_u1(monkeypatch):
    env = MDPPCH()
    env.reset(seed=0)
    observed = {}

    monkeypatch.setattr(env.env, "sample_u", lambda: (1, 0, 0))

    def natural_action(state, u1):
        observed["natural_u1"] = u1
        return 0

    def step(action, u1, u2, u3):
        observed["step_u1"] = u1
        return 0, 0, False, False, {}

    monkeypatch.setattr(env.env, "action", natural_action)
    monkeypatch.setattr(env.env, "step", step)

    env.ctf_do(lambda observation, natural_action: 1)

    assert observed == {"natural_u1": 1, "step_u1": 1}


def test_mdp_truncates_at_the_configured_horizon():
    env = MDPPCH(max_step=1)
    env.reset(seed=0)

    _, _, terminated, truncated, _ = env.do(lambda observation: 0)

    assert terminated is False
    assert truncated is True


def _graph_edges(env):
    return {
        (edge["from_"], edge["to_"], edge["type_"])
        for edge in env.env.get_graph.edges
    }


def test_dtr_graph_matches_unconfounded_structural_equations():
    env = DTRPCH(a1=0, a2=0)

    assert _graph_edges(env) == {
        ("S1", "X1", "directed"),
        ("S1", "S2", "directed"),
        ("X1", "S2", "directed"),
        ("S2", "X2", "directed"),
        ("S1", "Y", "directed"),
        ("X1", "Y", "directed"),
        ("S2", "Y", "directed"),
        ("X2", "Y", "directed"),
        ("U", "Y", "directed"),
    }


def test_dtr_graph_adds_only_active_shared_confounder_edges():
    env = DTRPCH(a1=0.5, a2=0.5)
    edges = _graph_edges(env)

    assert ("U", "X1", "directed") in edges
    assert ("U", "X2", "directed") in edges
    assert all(edge_type in {"directed", "bidirected"} for _, _, edge_type in edges)
    assert ("U", "S1", "directed") not in edges
    assert ("U", "S2", "directed") not in edges
    assert ("S1", "X2", "directed") not in edges
    assert ("X1", "X2", "directed") not in edges


def test_dtr_outcome_uses_independent_logistic_disturbance(monkeypatch):
    env = DTRSCM()
    env.u = 0.0
    env.s1 = env.x1 = env.s2 = 0
    env.stage = 1
    monkeypatch.setattr(env, "_logistic", lambda: 1.0)

    assert env.step(0)[1] == 1

    env.stage = 1
    monkeypatch.setattr(env, "_logistic", lambda: -1.0)
    assert env.step(0)[1] == 0


@pytest.mark.parametrize(
    ("state", "expected"),
    [
        ((0, 0, 0, 1), 0.9846294445),
        ((0, 0, 0, 0), 0.7851467237),
        ((1, 0, 0, 0), 0.2148532763),
        ((1, 1, 1, 1), 0.0153705555),
        ((1, 1, 0, 0), 0.0007840943),
    ],
)
def test_dtr_outcome_matches_reference_q_table(state, expected):
    env = DTRSCM()
    grid_size = 256
    total = 0
    env._fixed_outcome_noise = 0.0
    env._logistic = lambda: env._fixed_outcome_noise

    s1, x1, s2, x2 = state
    for u_index in range(grid_size):
        env.u = (u_index + 0.5) / grid_size
        for noise_index in range(grid_size):
            logistic_cdf_value = (noise_index + 0.5) / grid_size
            env._fixed_outcome_noise = math.log(
                logistic_cdf_value / (1.0 - logistic_cdf_value)
            )
            env.s1 = s1
            env.x1 = x1
            env.s2 = s2
            env.stage = 1
            total += env.step(x2)[1]

    assert total / (grid_size * grid_size) == pytest.approx(expected, abs=1e-3)
