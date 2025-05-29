from .windy_minigrid import WindyMiniGridSCM, WindyMiniGridPCH
from .mdp_example import MDPExampleSCM, MDPExamplePCH
from .lava_minigrid import CustomCrossingEnv
from .wind_dist import WIND_DIST
from .wheredo_example import ExampleSCM_9_5, ExamplePCH_9_5
from .wheredo_exercise import ExerciseSCM, ExercisePCH
from .cartpole_wind import CartPoleWindSCM, CartPoleWindPCH
from .frozen_lake import FrozenLakeSCM, FrozenLakePCH
from .lunar_lander import LunarLanderSCM, LunarLanderPCH
from .mnist import MNISTSCM, MNISTPCH
from .highway_single_step import HighwaySingleStepSCM, HighwaySingleStepPCH
from .highway import HighwaySCM, HighwayPCH
from .race import RaceSCM, RacePCH
from .masked_atari import MaskedAtariSCM, MaskedAtariPCH
from .random_friction_ant import RandomFrictionAntMujocoSCM, RandomFrictionAntMujocoPCH

from gymnasium.envs.registration import register

# We don't add max_episode_steps, disable_env_checker, or order_enforce for SCMs
# this is to remove redundant env wrappers that are incompatible with SCM APIs

register(
    id="causal_gym/WindyGridWorld-v0",
    entry_point="causal_gym.envs:WindyGridWorldEnv",
    max_episode_steps=None,
    disable_env_checker=True,
    order_enforce=False,
)

SUPPORTED_ATARI_ENVS = [
    "Pong",
]
for env_name in SUPPORTED_ATARI_ENVS:
    register(
        id=f"causal_gym/Masked{env_name}-v0",
        entry_point="causal_gym.envs:MaskedAtariPCH",
        kwargs={"env_name": env_name},
        max_episode_steps=None,
        disable_env_checker=True,
        order_enforce=False,
    )

register(
    id="causal_gym/MDPExample-v0",
    entry_point="causal_gym.envs:MDPExamplePCH",
    max_episode_steps=None,
    disable_env_checker=True,
    order_enforce=False,
)

register(
    id="causal_gym/CartPoleWind-v0",
    entry_point="causal_gym.envs:CartPoleWindPCH",
    max_episode_steps=None,
    disable_env_checker=True,
    order_enforce=False,
)

register(
    id="causal_gym/FrozenLakePCH-v0",
    entry_point="causal_gym.envs:FrozenLakePCH",
    max_episode_steps=None,
    disable_env_checker=True,
    order_enforce=False,
)

register(
    id="causal_gym/LunarLanderPCH-v0",
    entry_point="causal_gym.envs:LunarLanderPCH",
    max_episode_steps=None,
    disable_env_checker=True,
    order_enforce=False,
)

register(
    id="causal_gym/RandomFrictionAntPCH-v0",
    entry_point="causal_gym.envs:RandomFrictionAntMujocoPCH",
    max_episode_steps=None,
    disable_env_checker=True,
    order_enforce=False,
)


# These are still in the native gymnasium environments format
register(
    id="Custom-LavaCrossing-easy-v0",
    entry_point=CustomCrossingEnv,
    kwargs={"mode": 'easy'},
)

register(
    id="Custom-LavaCrossing-hard-v0",
    entry_point=CustomCrossingEnv,
    kwargs={"mode": 'hard'},
)

register(
    id="Custom-LavaCrossing-extreme-v0",
    entry_point=CustomCrossingEnv,
    kwargs={"mode": 'extreme'},
)

register(
    id="Custom-LavaCrossing-maze-v0",
    entry_point=CustomCrossingEnv,
    kwargs={"mode": 'maze'},
)

register(
    id="Custom-LavaCrossing-maze-complex-v0",
    entry_point=CustomCrossingEnv,
    kwargs={"mode": 'maze2'},
)
