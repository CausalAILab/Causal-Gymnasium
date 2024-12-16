from gymnasium.envs.registration import register
from .scm import (
    SCM,
)
from .pch import (
    PCH,
    PCHWrapper,
    ActionPCHWrapper,
    ObservationPCHWrapper,
    RewardPCHWrapper,
    PolicyPCHWrapper,
)
from .policy_scope import (
    PolicyScope
)
from .task import (
    Task
)
from .types import (
    ObsType,
    ActType,
    PolicyType,
    WrapperObsType,
    WrapperActType,
    WrapperPolicyType
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
