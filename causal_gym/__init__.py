from gymnasium.envs.registration import register
from causal_gym.core import (
    SCM,
    PCH,
    PCHWrapper,
    ActionPCHWrapper,
    ObservationPCHWrapper,
    RewardPCHWrapper,
    PolicyPCHWrapper,
)

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
