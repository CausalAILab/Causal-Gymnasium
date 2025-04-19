from .windy_gridworld import WindyGridWorldEnv
from .windy_minigrid import WindyMiniGridSCM, WindyMiniGridPCH
from .mdp_example import MDPExampleSCM, MDPExamplePCH
from .lava_minigrid import CustomCrossingEnv
from .wind_dist import WIND_DIST

from .mnist import MNISTSCM, MNISTPCH
from .highway_single_step import HighwaySingleStepSCM, HighwaySingleStepPCH

from gymnasium.envs.registration import register
register(
    id="causal_gym/WindyGridWorld-v0",
    entry_point="causal_gym.envs:WindyGridWorldEnv",
    max_episode_steps=10,
)

# register(
#     id="causal_gym/MDPExample-v0",
#     entry_point="causal_gym.envs:MDPExamplePCH",
#     max_episode_steps=50,
# )

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

