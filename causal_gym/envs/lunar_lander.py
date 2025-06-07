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

import cv2
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from ..core import SCM, PCH, Task, Graph
from ..core.types import ObsType, ActType, PolicyType
from .constants import WIND_ICONS
from .utils import overlay_resized_image

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
        policy: PolicyType | None = None,
    ) -> None:
        super().__init__()
        self._env = gym.make("LunarLander-v3", render_mode=render_mode)
        self._env._max_episode_steps = max_episode_steps

        # Exogenous wind distribution (sampled once per episode)
        self.wind_mean = wind_mean
        self.wind_std = wind_std
        # positive for left, negative for right wind
        self.current_wind: float | None = None
        self.wind_map: tuple | None = None
        self._last_obs: ObsType | None = None # Initialize last observation storage

        # Observation / action spaces
        self.observation_space: spaces.Box = self._env.observation_space  # 8‑D
        self.action_space: spaces.Discrete = self._env.action_space        # 4 actions

        # Simple random behaviour policy (can be replaced by caller)
        if policy is not None:
            self.policy = policy
        else:
            self.policy = lambda obs, wind: self.np_random.integers(0, self.action_space.n)

    # ------------------------------------------------------------------
    #  SCM API
    # ------------------------------------------------------------------
    def sample_u(self):
        """Draw the latent wind for *this* episode and store it."""
        # the map is a continuous box from 0 to 20 on x 
        # and 0-15 roughly on y axes
        # we discretize it to a 15x20 grid for simplicity
        self.wind_map = self.np_random.normal(self.wind_mean, self.wind_std, size=(15, 20))
        return self.wind_map

    def get_current_wind(self):
        # Extract lander position from last observation
        lander_x, lander_y = self.unwrapped.lander.position.x, self.unwrapped.lander.position.y
        # Ensure the lander position is within the bounds of the wind map
        x_to_index = lambda x: np.floor(max(0, min(19.9, x))).astype(int)
        y_to_index = lambda y: np.floor(max(0, min(14.9, y))).astype(int)
        if self.wind_map is None:
            raise ValueError("Wind map has not been sampled yet. Call sample_u() first.")
        self.current_wind = self.wind_map[y_to_index(lander_y)][x_to_index(lander_x)]
        return self.current_wind

    # Gym‑style interface ------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed, options=options)
        obs, info = self._env.reset(seed=seed)
        info['wind_map'] = self.sample_u()  # draw new wind
        self._last_obs = obs.astype(np.float32) # Store initial observation
        return self._last_obs, info

    def step(self, action: int):
        # Inject wind by applying a small horizontal force each physics step
        if self.wind_map is not None:
            self.current_wind = self.get_current_wind()  # Get the current wind based on lander position
            # access underlying Box2D lander body
            self._env.unwrapped.lander.ApplyForceToCenter((self.current_wind, 0.0), True)
            # observation after force (approx) – we leave obs unchanged; stochasticity
            # is captured in the physics engine itself.

        # Get current natural action
        self._natural_action = self.action()
        
        # Step the environment after the wind application
        obs, reward, terminated, truncated, info = self._env.step(action)

        # Only sample wind for each single episode
        # self.sample_u()  # draw new wind

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
        return self.policy(obs, self.current_wind)

    def observation(self):
        # Return the last stored observation
        return self._last_obs 

    def render(self, show_wind=False, show_natural_action=False):
        obs = self._env.render()
        if show_wind:
            # right, down, left, up, circle, cross
            if self.current_wind > 0:
                wind_icon = WIND_ICONS[0] 
            elif self.current_wind < 0:
                wind_icon = WIND_ICONS[2]
            else:
                wind_icon = WIND_ICONS[4]
            # overlay the windicon upon original obs
            cv2.putText(
                obs,                       # image (in-place modification)
                text='Wind:',              # text to draw
                org=(10, 350),             # bottom-left corner of text
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,  # font
                fontScale=0.5,             # font scale (size)
                color=(0, 0, 0),     # color (B, G, R) — white in BGR
                thickness=1,               # thickness of the stroke
                lineType=cv2.LINE_AA       # anti-aliased line
            )
            obs = overlay_resized_image(obs, wind_icon, 1/20, 330, 50)
        if show_natural_action:
            action = self._natural_action
            if action == 0:
                # do nothing
                action_icon = WIND_ICONS[4]
            elif action == 1:
                # left orient engine
                action_icon = WIND_ICONS[2]
            elif action == 2:
                # main, up
                action_icon = WIND_ICONS[3]
            else:
                # right orient engine
                action_icon = WIND_ICONS[0]
            cv2.putText(
                obs,                       # image (in-place modification)
                text='Natural Action:',              # text to draw
                org=(10, 380),             # bottom-left corner of text
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,  # font
                fontScale=0.5,             # font scale (size)
                color=(0, 0, 0),     # color (B, G, R) — white in BGR
                thickness=1,               # thickness of the stroke
                lineType=cv2.LINE_AA       # anti-aliased line
            )
            obs = overlay_resized_image(obs, action_icon, 1/20, 360, 130)
        return obs

    def close(self):
        return self._env.close()

    # Causal graph -------------------------------------------------------
    @property
    def get_graph(self):
        nodes = [
            # {'name': 'U', 'label': 'Wind', 'type': 'latent'},
            {'name': 'S', 'label': 'State'},
            {'name': 'X', 'label': 'Action'},
            {'name': 'Y', 'label': 'Reward'},
            {'name': "S'", 'label': 'Next State'}
        ]

        edges = [
            # {'from_': 'U', 'to_': 'X', 'type_': 'directed'},
            # {'from_': 'U', 'to_': "S'", 'type_': 'directed'},
            {'from_': 'S', 'to_': 'X', 'type_': 'directed'},
            {'from_': 'S', 'to_': 'Y', 'type_': 'directed'},
            {'from_': 'X', 'to_': 'Y', 'type_': 'directed'},
            {'from_': 'S', 'to_': "S'", 'type_': 'directed'},
            {'from_': 'X', 'to_': "S'", 'type_': 'directed'},
            # Bidirected confounding between Action and Next State
            {'from_': 'X', 'to_': "S'", 'type_': 'bidirected'}
        ]
        graph = Graph(nodes=nodes, edges=edges)
        return graph


# =============================================================
#  LunarLanderPCH
# =============================================================
class LunarLanderPCH(PCH):
    """PCH wrapper exposing (see) and (do) for LunarLanderSCM."""

    def __init__(self, **kwargs):
        task = kwargs.pop("task", Task())
        self.env: LunarLanderSCM = LunarLanderSCM(**kwargs)  # Create SCM and assign to self.env
        super().__init__(env=self.env, task=task)       # Call PCH.__init__, passing self.env as a kwarg

    # ------------------------------------------------------------------
    #  Layer‑1 observational regime – SEE
    # ------------------------------------------------------------------
    def see(self):
        a = self.env.action()
        obs, r, terminated, truncated, info = self.env.step(a)
        return a, obs, r, terminated, truncated, info

    # ------------------------------------------------------------------
    #  Layer‑2 interventional regime – DO
    # ------------------------------------------------------------------
    def do(self, a: int):
        obs, r, terminated, truncated, info = self.env.step(a)
        return obs, r, terminated, truncated, info # Return 5 values
    
    # Counterfactual policy intervention
    def ctf_do(self, ctf_policy):
        intuition = self.env.action()
        action = ctf_policy(self.env.observation(), intuition)
        obs, r, terminated, truncated, info = self.env.step(action)
        info['natural_action'] = intuition
        return action, obs, r, terminated, truncated, info

    def render(self, show_wind=False, show_natural_action=False):
        return self.env.render(show_wind, show_natural_action)
    