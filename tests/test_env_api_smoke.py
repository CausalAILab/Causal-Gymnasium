import gymnasium as gym
import pytest

from causal_gym.envs import CartPoleWindPCH, DTRPCH, FrozenLakePCH, MABPCH, MDPPCH, RobotWalkPCH


LIGHTWEIGHT_PCH_ENVS = [
    CartPoleWindPCH,
    DTRPCH,
    FrozenLakePCH,
    MABPCH,
    MDPPCH,
    RobotWalkPCH,
]


def policy(env):
    return 0 if not hasattr(env, "action_space") else env.action_space.sample()


def do_policy(env):
    return lambda observation: policy(env)


def ctf_policy(env):
    return lambda observation, natural_action: policy(env)


@pytest.mark.parametrize("env_cls", LIGHTWEIGHT_PCH_ENVS)
def test_lightweight_pch_env_reset_and_spaces(env_cls):
    env = env_cls()
    obs, info = env.reset(seed=123)

    assert isinstance(info, dict)
    assert hasattr(env, "action_space")
    assert hasattr(env, "observation_space")
    assert env.action_space.contains(policy(env))
    assert env.observation_space.contains(obs)


@pytest.mark.parametrize("env_cls", LIGHTWEIGHT_PCH_ENVS)
def test_lightweight_pch_env_see_do_ctf_do_return_gymnasium_tuple(env_cls):
    env = env_cls()

    env.reset(seed=123)
    see_result = env.see()
    assert len(see_result) == 5
    assert isinstance(see_result[4], dict)

    env.reset(seed=123)
    do_result = env.do(do_policy(env))
    assert len(do_result) == 5
    assert isinstance(do_result[4], dict)

    env.reset(seed=123)
    ctf_result = env.ctf_do(ctf_policy(env))
    assert len(ctf_result) == 5
    assert isinstance(ctf_result[4], dict)


def test_registered_lightweight_env_entry_points_exist():
    assert gym.spec("causal_gym/MDPExample-v0").entry_point == "causal_gym.envs:MDPPCH"
    assert gym.spec("causal_gym/CartPoleWind-v0").entry_point == "causal_gym.envs:CartPoleWindPCH"
    assert gym.spec("causal_gym/FrozenLakePCH-v0").entry_point == "causal_gym.envs:FrozenLakePCH"
    assert gym.spec("causal_gym/RobotWalk-v0").entry_point == "causal_gym.envs:RobotWalkPCH"
