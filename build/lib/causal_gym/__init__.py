from gymnasium.envs.registration import register
from causal_gym.core import (
    SCM,
    SCMWrapper,
    ActionSCMWrapper,
    ObservationSCMWrapper,
    RewardSCMWrapper,
    PolicySCMWrapper,
)

register(
    id="causal_gym/SimplePacman-v0",
    entry_point="causal_gym.envs:SimplePacmanEnv",
    max_episode_steps=10,
)

register(
    id="causal_gym/WindyGridWorld-v0",
    entry_point="causal_gym.envs:WindyGridWorldEnv",
    max_episode_steps=10,
)

register(
    id="causal_gym/GridWorld-v0",
    entry_point="causal_gym.envs:GridWorldEnv",
    max_episode_steps=10,
)
