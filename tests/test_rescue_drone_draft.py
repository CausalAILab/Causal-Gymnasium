import numpy as np
import pytest

from causal_gym.core import Task
from causal_gym.envs.rescue_drone import RescueDronePCH, RescueDroneSCM


def test_reset_and_stage_noise_are_seed_reproducible():
    first = RescueDroneSCM(include_latent_in_info=True)
    second = RescueDroneSCM(include_latent_in_info=True)

    obs_first, _ = first.reset(seed=17)
    obs_second, _ = second.reset(seed=17)
    assert np.array_equal(obs_first, obs_second)

    out_first = first.step(0)
    out_second = second.step(0)
    assert np.array_equal(out_first[0], out_second[0])
    assert out_first[1:] == out_second[1:]


def test_latent_is_hidden_by_default_but_available_in_explicit_debug_mode():
    hidden = RescueDroneSCM()
    hidden.reset(seed=3)
    _, _, _, _, hidden_info = hidden.step(0)
    assert "exogenous" not in hidden_info

    debug = RescueDroneSCM(include_latent_in_info=True)
    debug.reset(seed=3)
    _, _, _, _, debug_info = debug.step(0)
    assert set(debug_info["exogenous"]) == {"gust_x", "gust_y", "hazard"}


def test_unit_counterfactuals_reuse_one_exogenous_realization_without_stepping():
    env = RescueDroneSCM()
    obs_before, _ = env.reset(seed=9)
    table_first = env.unit_counterfactuals()
    table_second = env.unit_counterfactuals()

    assert table_first == table_second
    assert len(table_first["alternatives"]) == env.action_space.n
    assert np.array_equal(obs_before, env.observation())


def test_pch_distinguishes_see_do_and_ctf_do():
    see_do = RescueDronePCH(task=Task(learning_regime="see_do", assumptions="dag"))
    see_do.reset(seed=21)
    _, _, _, _, see_info = see_do.see()
    assert "natural_action" in see_info

    see_do.reset(seed=21)
    _, _, _, _, do_info = see_do.do(lambda obs: 0)
    assert do_info["action"] == 0

    counterfactual = RescueDronePCH(
        task=Task(learning_regime="ctf_do", assumptions="dag")
    )
    counterfactual.reset(seed=21)
    _, _, _, _, ctf_info = counterfactual.ctf_do(
        lambda obs, intended: (intended + 1) % 5
    )
    assert ctf_info["action"] == (ctf_info["natural_action"] + 1) % 5


def test_do_requires_a_policy_callable():
    env = RescueDronePCH(task=Task(learning_regime="do", assumptions="dag"))
    env.reset(seed=1)
    with pytest.raises(TypeError, match="callable"):
        env.do(0)


def test_graph_matches_structural_arguments_and_shared_noise():
    graph = RescueDroneSCM().get_graph
    edge_types = {
        (edge["from_"], edge["to_"], edge["type_"]) for edge in graph.edges
    }
    assert ("S", "O", "directed") in edge_types
    assert ("O", "X", "directed") in edge_types
    assert ("S", "Y", "directed") in edge_types
    assert ("X", "Y", "directed") in edge_types
    assert any(
        {source, target} == {"X", "Y"} and edge_type == "bidirected"
        for source, target, edge_type in edge_types
    )
    assert any(
        {source, target} == {"X", "S'"} and edge_type == "bidirected"
        for source, target, edge_type in edge_types
    )
