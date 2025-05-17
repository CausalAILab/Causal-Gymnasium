"""
lunar_lander.py
----------------
Causal‑AI Gym wrapper for Gymnasium's **LunarLander‑v2** (discrete action).  
Adds an explicit latent *wind* variable so that exogenous randomness is
exposed through `sample_u()` and visible in the causal graph.

Exports two public classes:
    * `LunarLanderSCM` – implements the SCM interface
    * `LunarLanderPCH` – thin PCH wrapper exposing `see()` / `do()`

Place this file inside `causal_gym/envs/` next to the existing
`frozen_lake.py` and `cartpole_wind.py` modules.
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from ..core import SCM, PCH
from ..core.types import ObsType, ActType, PolicyType

__all__ = ["LunarLanderSCM", "LunarLanderPCH"]

# =============================================================
#  LunarLanderSCM
# =============================================================
class LunarLanderSCM(SCM[PolicyType, ObsType, ActType]):
    """Lunar‑Lander with a latent horizontal *wind* force (u_wind)."""

    def __init__(
        self,
        *,
        max_episode_steps: int = 1000,
        wind_mean: float = 0.0,
        wind_std: float = 0.2,
        render_mode=None,
    ) -> None:
        super().__init__()
        self._env = gym.make("LunarLander-v3", render_mode=render_mode)
        self._env._max_episode_steps = max_episode_steps

        # Exogenous wind distribution (sampled once per episode)
        self.wind_mean = wind_mean
        self.wind_std = wind_std
        self.current_wind: float | None = None
        self._last_obs: ObsType | None = None # Initialize last observation storage

        # Observation / action spaces
        self.observation_space: spaces.Box = self._env.observation_space  # 8‑D
        self.action_space: spaces.Discrete = self._env.action_space        # 4 actions

        # Simple random behaviour policy (can be replaced by caller)
        self.policy = lambda obs: self.np_random.integers(0, self.action_space.n)

    # ------------------------------------------------------------------
    #  SCM API
    # ------------------------------------------------------------------
    def sample_u(self):
        """Draw the latent wind for *this* episode and store it."""
        self.current_wind = self.np_random.normal(self.wind_mean, self.wind_std)
        return {"u_wind": self.current_wind}

    # Gym‑style interface ------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed, options=options)
        obs, info = self._env.reset(seed=seed)
        self.sample_u()  # draw new wind
        self._last_obs = obs.astype(np.float32) # Store initial observation
        return self._last_obs, info

    def step(self, action: int):
        obs, reward, terminated, truncated, info = self._env.step(action)

        # Inject wind by applying a small horizontal force each physics step
        if self.current_wind is not None:
            # access underlying Box2D lander body
            self._env.unwrapped.lander.ApplyForceToCenter((self.current_wind, 0.0), True)
            # observation after force (approx) – we leave obs unchanged; stochasticity
            # is captured in the physics engine itself.

        self._last_obs = obs.astype(np.float32) # Store observation after step
        return self._last_obs, float(reward), terminated, truncated, info # Return 5 values

    # Convenience helpers -----------------------------------------------
    def action(self):
        obs = self.observation()
        if obs is None:
             # Handle case where action is called before reset (e.g., return default action)
             # This might occur if the policy needs an observation immediately upon init
             # For now, let's assume reset is always called first. If errors persist, 
             # we might need a default observation or action here.
             # Returning random action as a placeholder if obs is None
             return self.np_random.integers(0, self.action_space.n)
        return self.policy(obs)

    def observation(self):
        # Return the last stored observation
        return self._last_obs 

    def render(self):
        return self._env.render()

    def close(self):
        return self._env.close()

    # Causal graph (optional, for visualisation) ------------------------
    @property
    def get_graph(self):
        nodes = {0: "Wind(U)", 1: "State(V)", 2: "Action(X)", 3: "Reward(Y)"}
        base = [[0] * 4 for _ in range(4)]
        base[0][1] = 1  # U → V
        base[1][2] = 1  # V → X
        base[1][3] = 1  # V → Y
        base[2][3] = 1  # X → Y
        conf = [[0] * 4 for _ in range(4)]
        return nodes, base, conf


# =============================================================
#  LunarLanderPCH
# =============================================================
class LunarLanderPCH(PCH):
    """PCH wrapper exposing (see) and (do) for LunarLanderSCM."""

    def __init__(self, **kwargs):
        self.env = LunarLanderSCM(**kwargs)  # Create SCM and assign to self.env
        super().__init__(env=self.env)       # Call PCH.__init__, passing self.env as a kwarg

    # ------------------------------------------------------------------
    #  Layer‑1 observational regime – SEE
    # ------------------------------------------------------------------
    def see(self):
        a = self.env.action()
        obs, r, done, info = self.env.step(a)
        return a, obs, r, done, info

    # ------------------------------------------------------------------
    #  Layer‑2 interventional regime – DO
    # ------------------------------------------------------------------
    def do(self, a: int):
        obs, r, terminated, truncated, info = self.env.step(a)
        return obs, r, terminated, truncated, info # Return 5 values