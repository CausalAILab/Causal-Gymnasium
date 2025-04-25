### frozenlake.py
import gymnasium as gym
import numpy as np
from ..core import SCM, PCH
from ..core.types import PolicyType, ObsType, ActType

class FrozenLakeSCM(SCM[PolicyType, ObsType, ActType]):
    """
    Structural wrapper for FrozenLake-v1.
    Exogenous randomness comes solely from the env's start-state and slipperiness.
    """
    def __init__(self, map_name="4x4", is_slippery=True, render_mode=None):
        super().__init__()
        self.env = gym.make(
            "FrozenLake-v1",
            map_name=map_name,
            is_slippery=is_slippery,
            render_mode=render_mode,
        )
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def sample_u(self):
        # No extra exogenous beyond start seed & slipperiness
        return {}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed, options=options)
        obs, info = self.env.reset(seed=seed)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        return obs, reward, done, info

    def action(self):
        # Default behavior policy: uniform random
        return self.np_random.integers(self.action_space.n)

    def observation(self):
        # Not used; step returns obs directly
        raise NotImplementedError

    def close(self):
        return self.env.close()

class FrozenLakePCH(PCH):
    """
    PCH wrapper for FrozenLakeSCM:
      - see()   produces observational (L1) data
      - do(a)   produces interventional (L2) data
    """
    def __init__(self, **kwargs):
        self.env = FrozenLakeSCM(**kwargs)
        super().__init__(env=self.env)

    def see(self):
        # Observational: sample action from behavior policy
        a = self.env.action()
        obs, r, done, info = self.env.step(a)
        return a, obs, r, done, info

    def do(self, action):
        # Interventional: force given action
        obs, r, done, info = self.env.step(action)
        return obs, r, done, info