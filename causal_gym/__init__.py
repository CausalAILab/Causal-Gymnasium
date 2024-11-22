from gymnasium.envs.registration import register
from causal_gym.core import (
    SCM,
    PCH,
    SCMWrapper,
    ActionSCMWrapper,
    ObservationSCMWrapper,
    RewardSCMWrapper,
    PolicySCMWrapper,
)

register(
    id="causal_gym/WindyGridWorld-v0",
    entry_point="causal_gym.envs:WindyGridWorldEnv",
    max_episode_steps=10,
)
