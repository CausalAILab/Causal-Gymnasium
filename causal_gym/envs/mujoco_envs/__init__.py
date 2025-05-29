from gym.envs.registration import registry

# re-register custom envs to replace gym's default ones
from gymnasium.envs.registration import register as gym_register, registry

def register(id: str, **kwargs):
    """
    Re-register an env id if it already exists (Gymnasium 0.27+ API).
    """
    if id in registry:          # registry is now a dict[str, EnvSpec]
        del registry[id]        # remove the old spec before re-adding
    return gym_register(id=id, **kwargs)

register(
    id='Reacher-v2',
    entry_point='envs.mujoco:ReacherEnv',
    max_episode_steps=50,
    reward_threshold=-3.75,
)

register(
    id='Pusher-v2',
    entry_point='envs.mujoco:PusherEnv',
    max_episode_steps=100,
    reward_threshold=0.0,
)

register(
    id='Thrower-v2',
    entry_point='envs.mujoco:ThrowerEnv',
    max_episode_steps=100,
    reward_threshold=0.0,
)

register(
    id='Striker-v2',
    entry_point='envs.mujoco:StrikerEnv',
    max_episode_steps=100,
    reward_threshold=0.0,
)

register(
    id='InvertedPendulum-v2',
    entry_point='envs.mujoco:InvertedPendulumEnv',
    max_episode_steps=1000,
    reward_threshold=950.0,
)

register(
    id='InvertedDoublePendulum-v2',
    entry_point='envs.mujoco:InvertedDoublePendulumEnv',
    max_episode_steps=1000,
    reward_threshold=9100.0,
)

register(
    id='HalfCheetah-v2',
    entry_point='envs.mujoco:HalfCheetahEnv',
    max_episode_steps=1000,
    reward_threshold=4800.0,
)

register(
    id='HalfCheetah-v3',
    entry_point='envs.mujoco.half_cheetah_v3:HalfCheetahEnv',
    max_episode_steps=1000,
    reward_threshold=4800.0,
)

register(
    id='Hopper-v2',
    entry_point='envs.mujoco:HopperEnv',
    max_episode_steps=1000,
    reward_threshold=3800.0,
)

register(
    id='Hopper-v3',
    entry_point='envs.mujoco.hopper_v3:HopperEnv',
    max_episode_steps=1000,
    reward_threshold=3800.0,
)

register(
    id='Swimmer-v2',
    entry_point='envs.mujoco:SwimmerEnv',
    max_episode_steps=1000,
    reward_threshold=360.0,
)

register(
    id='Swimmer-v3',
    entry_point='envs.mujoco.swimmer_v3:SwimmerEnv',
    max_episode_steps=1000,
    reward_threshold=360.0,
)

register(
    id='Walker2d-v2',
    max_episode_steps=1000,
    entry_point='envs.mujoco:Walker2dEnv',
)

register(
    id='Walker2d-v3',
    max_episode_steps=1000,
    entry_point='envs.mujoco.walker2d_v3:Walker2dEnv',
)

register(
    id='Ant-v2',
    entry_point='envs.mujoco:AntEnv',
    max_episode_steps=1000,
    reward_threshold=6000.0,
)

register(
    id='Ant-v3',
    entry_point='envs.mujoco.ant_v3:AntEnv',
    max_episode_steps=1000,
    reward_threshold=6000.0,
)

register(
    id='Humanoid-v2',
    entry_point='envs.mujoco:HumanoidEnv',
    max_episode_steps=1000,
)

register(
    id='Humanoid-v3',
    entry_point='envs.mujoco.humanoid_v3:HumanoidEnv',
    max_episode_steps=1000,
)

register(
    id='HumanoidStandup-v2',
    entry_point='envs.mujoco:HumanoidStandupEnv',
    max_episode_steps=1000,
)

# register(
#     id="AdroitHandDoor-v1",
#     entry_point='envs.mujoco:AdroitHandDoorEnv',
#     max_episode_steps=10,
# )
print("Registering AdroitDoorMasked-v1")
register(
    id="AdroitDoorMasked-v1",
    entry_point="envs.adroit_door_wrapper:AdroitDoorEnvMasked",
    max_episode_steps=250, 
)
