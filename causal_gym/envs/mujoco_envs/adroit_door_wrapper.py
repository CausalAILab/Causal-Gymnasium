"""
Masked version of Gymnasium-Robotics Adroit Door (28-DoF hand + door).
© 2025  — BSD-3-Clause, same licence as source repos.
"""
from __future__ import annotations
import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box
from gymnasium_robotics.envs.adroit_hand.adroit_door import AdroitHandDoorEnv
from gymnasium import spaces # Ensure spaces is imported

class AdroitDoorEnvMasked(AdroitHandDoorEnv):
    """
    Wraps the AdroitHandDoorEnv.
    The 'mask' argument is accepted for compatibility with existing scripts that might pass it
    during gym.make(), but it is NOT used by this wrapper to modify the action space
    or filter actions. The environment will always present its full, original action space (28 DoF)
    and expect full-dimensioned actions in its step method.
    Action sub-setting and mapping are handled externally by the agent/training script.

    Args
    ----
    mask : np.ndarray[bool] | None
        This argument is IGNORED by this version of the wrapper for action processing.
        It's kept for compatibility with gym.make calls that might provide it.
    reward_type : str
        Forwarded to the parent class ("dense" or "sparse").
    **kwargs
        Forwarded verbatim to `AdroitHandDoorEnv`.
    """
    metadata = AdroitHandDoorEnv.metadata          # preserve native render_modes/FPS

    def __init__(self, mask: np.ndarray | None = None, reward_type: str = "dense", **kwargs):
        print("Initializing AdroitDoorEnvMasked")
        # Initialize the superclass (AdroitHandDoorEnv)
        # The `mask` argument received here is deliberately NOT used to alter
        # self.action_space or observation_space.
        # AdroitHandDoorEnv itself sets up the full 28-dim action space.
        super().__init__(reward_type=reward_type, **kwargs)

        # # ——— Normalise & store mask ———
        # if mask is None:
        #     mask = np.ones(self.action_space.shape, dtype=bool)
        # mask   = np.asarray(mask, dtype=bool)
        # assert mask.shape == self.action_space.shape, \
        #     f"Mask must be shape {self.action_space.shape} but got {mask.shape}"
        # self._mask = mask
        # Ensure self.action_space is the original one from AdroitHandDoorEnv
        # This should be handled by AdroitHandDoorEnv's __init__ correctly.
        # No need to redefine self.action_space here based on the input `mask`.

        # # ——— Keep the same 28-D Box so PPO nets don't break ———
        # low, high = self.action_space.low, self.action_space.high       # [-1, 1]
        # self.action_space = Box(low=low, high=high, dtype=np.float32)   # [oai_citation:1‡GitHub](https://github.com/Farama-Foundation/Gymnasium-Robotics/blob/main/gymnasium_robotics/envs/adroit_hand/adroit_door.py)

        # The `mask` argument is not stored or used further for action manipulation within this class.
        if mask is not None:
            # You could log a warning if strict adherence to ignoring it is important.
            # print("AdroitDoorEnvMasked: 'mask' argument was provided but is ignored for action processing.")
            pass

        # Observation is already flat (39,) in upstream env, but we declare it here
        # with the desired dtype for the rest of the PPO pipeline.
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(39,), dtype=np.float32)

    def step(self, action: np.ndarray):
        """
        Expects a full 28-dimensional action vector.
        Casts observation to np.float32.
        """
        obs, reward, terminated, truncated, info = super().step(action)
        return obs.astype(np.float32), reward, terminated, truncated, info

    def reset(self, *args, **kwargs):
        """
        Resets the environment. The observation space is the original one.
        Casts observation to np.float32.
        """
        obs, info = super().reset(*args, **kwargs)
        return obs.astype(np.float32), info

    # --------------------------------------------------------------------- #
    #  Interface                       (same signature as Gymnasium envs)   #
    # --------------------------------------------------------------------- #
    # def step(self, action: np.ndarray):
    #     """
    #     Replace masked entries by zero before forwarding to Mujoco.
    #     """
    #     action = np.asarray(action, dtype=np.float32).copy()
    #     action[~self._mask] = 0.0                   # frozen DoF                 # [oai_citation:3‡Ray Docs](https://docs.ray.io/en/latest/rllib/rllib-examples.html?utm_source=chatgpt.com)
    #     return super().step(action)

    # def reset(self, *args, **kwargs):
    #     """
    #     Pass-through reset.  Return flat observation.
    #     """
    #     obs, info = super().reset(*args, **kwargs)
    #     return obs.astype(np.float32, copy=False), info                   # [oai_citation:4‡Gymnasium](https://gymnasium.farama.org/introduction/create_custom_env/)
    # def step(self, action):
    #     action = np.asarray(action, dtype=np.float32).copy()
    #     action[~self._mask] = 0.0
    #     obs, reward, terminated, truncated, info = super().step(action)
    #     obs = obs.astype(np.float32, copy=False)        # <── cast once here
    #     speed_pen  = 5e-4 * np.sum(self.data.qvel**2)
    #     act_pen    = 1e-3 * np.sum(np.square(action))
    #     reward += -speed_pen - act_pen
    #     return obs, reward, terminated, truncated, info


    # --------------------------------------------------------------------- #
    #  Interface                       (same signature as Gymnasium envs)   #
    # --------------------------------------------------------------------- #
    # def step(self, action: np.ndarray):
    #     """
    #     Replace masked entries by zero before forwarding to Mujoco.
    #     """
    #     action = np.asarray(action, dtype=np.float32).copy()
    #     action[~self._mask] = 0.0                   # frozen DoF                 # [oai_citation:3‡Ray Docs](https://docs.ray.io/en/latest/rllib/rllib-examples.html?utm_source=chatgpt.com)
    #     return super().step(action)

    # def reset(self, *args, **kwargs):
    #     """
    #     Pass-through reset.  Return flat observation.
    #     """
    #     obs, info = super().reset(*args, **kwargs)
    #     return obs.astype(np.float32, copy=False), info                   # [oai_citation:4‡Gymnasium](https://gymnasium.farama.org/introduction/create_custom_env/)
    # def step(self, action):
    #     action = np.asarray(action, dtype=np.float32).copy()
    #     action[~self._mask] = 0.0
    #     obs, reward, terminated, truncated, info = super().step(action)
    #     obs = obs.astype(np.float32, copy=False)        # <── cast once here
    #     speed_pen  = 5e-4 * np.sum(self.data.qvel**2)
    #     act_pen    = 1e-3 * np.sum(np.square(action))
    #     reward += -speed_pen - act_pen
    #     return obs, reward, terminated, truncated, info

    # def reset(self, *args, **kwargs):
    #     obs, info = super().reset(*args, **kwargs)
    #     jerk_pen = 0.0005 * np.sum((action - self.prev_action)**2)
    #     reward += -jerk_pen
    #     self.prev_action = action
    #     return obs.astype(np.float32, copy=False), info  # <── and here
    # def step(self, action):
    #     action = np.asarray(action, dtype=np.float32).copy()
    #     action[~self._mask] = 0.0
    #     obs, reward, terminated, truncated, info = super().step(action)
    #     obs = obs.astype(np.float32, copy=False)        # <── cast once here
    #     speed_pen  = 5e-4 * np.sum(self.data.qvel**2)
    #     act_pen    = 1e-3 * np.sum(np.square(action))
    #     reward += -speed_pen - act_pen
    #     return obs, reward, terminated, truncated, info

#         obs, reward, terminated, truncated, info = super().step(action)
#         obs = obs.astype(np.float32, copy=False)

#         # ---------- smooth-motion penalties ----------
#         # speed_pen = 0e-5 * np.sum(self.data.qvel ** 2)                 # velocity
#         # act_pen   = 1e-9 * np.sum(action ** 2)                         # action L2
#         # # jerk_pen  = 5e-4 * np.sum((action - self.prev_action) ** 2)    # jerk
#         # reward += -(speed_pen + act_pen)  #- (jerk_pen)

#         # self.prev_action = action
#         return obs, reward, terminated, truncated, info

#     def reset(self, *args, **kwargs):
#         obs, info = super().reset(*args, **kwargs)
#         # self.prev_action[:] = 0.0                                       # reset history
#         return obs.astype(np.float32, copy=False), info

    # def reset(self, *args, **kwargs):
    #     obs, info = super().reset(*args, **kwargs)
    #     jerk_pen = 0.0005 * np.sum((action - self.prev_action)**2)
    #     reward += -jerk_pen
    #     self.prev_action = action
    #     return obs.astype(np.float32, copy=False), info  # <── and here
    
# #### TODOS:
# # * New penalities (DONE!)
# # * Shift masking to the agent side/w padding (DONE!)
# # * Run through 32 (DONE!)
# # * Win percentage/ Rewards (DONE!)
 