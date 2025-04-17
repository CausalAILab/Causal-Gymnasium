"""
cartpole_wind.py
--------------------
Cart‑Pole environment wrapped in an SCM/PCH where *wind* is the
sole explicit exogenous driver.  Two public classes are exported:

* `CartPoleWindSCM` – an `SCM` implementation with wind‑gust
  latent variable.
* `CartPoleWindPCH` – a thin `PCH` wrapper exposing the usual
  `see()` / `do()` dual interface on top of the SCM.
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from ..core import SCM, PCH
from ..core.types import ObsType, ActType, PolicyType

# =============================================================
#  CartPoleWindSCM
# =============================================================
class CartPoleWindSCM(SCM[PolicyType, ObsType, ActType]):
    """CartPole with a latent horizontal *wind* force.

    The latent wind (u_wind) is sampled once at episode start and
    then applied at every step as a small additive acceleration to
    the cart's velocity.  Pole *tilt* is fixed to zero by default; if
    you still want random initial angles you can pass a non‑zero
    `init_theta_std`.
    """

    def __init__(
        self,
        *,
        max_episode_steps: int = 200,
        wind_mean: float = 0.0,
        wind_std: float = 0.01,
        init_theta_mean: float = 0.0,
        init_theta_std: float = 0.0,
        render_mode=None,
    ) -> None:
        super().__init__()
        self.max_episode_steps = max_episode_steps
        self._elapsed_steps = 0
        self.render_mode = render_mode

        # Exogenous distribution for wind
        self.wind_mean = wind_mean
        self.wind_std = wind_std
        self.current_wind: float | None = None  # filled in reset()

        # Optional init‑tilt parameters (commonly zeros)
        self.init_theta_mean = init_theta_mean
        self.init_theta_std = init_theta_std

        # Underlying Gym env
        self._env = gym.make("CartPole-v1", render_mode=render_mode)
        self._env._max_episode_steps = max_episode_steps

        self.action_space: spaces.Discrete = self._env.action_space
        self.observation_space: spaces.Box = self._env.observation_space

        # Trivial random behaviour policy (can be replaced)
        self.policy = lambda obs: self.np_random.integers(0, 2)

    # ------------------------------------------------------------------
    #  Causal‑Gym API
    # ------------------------------------------------------------------
    def sample_u(self):
        """Sample and return the exogenous variables for *this* episode."""
        self.current_wind = self.np_random.normal(self.wind_mean, self.wind_std)
        return {"u_wind": self.current_wind}

    # Gym‑style interface ------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self._elapsed_steps = 0

        exo = self.sample_u()
        obs, _info = self._env.reset(seed=seed)

        # overwrite initial tilt if desired
        theta0 = self.np_random.normal(self.init_theta_mean, self.init_theta_std)
        state = list(self._env.unwrapped.state)
        state[2] = theta0  # pole angle
        self._env.unwrapped.state = np.array(state, dtype=np.float32)

        return np.array(self._env.unwrapped.state, dtype=np.float32), {}

    def step(self, action):
        self._elapsed_steps += 1
        obs, reward, terminated, truncated, info = self._env.step(action)

        # Inject wind as horizontal acceleration (m/s per step)
        if self.current_wind is not None:
            state = list(self._env.unwrapped.state)
            state[1] += self.current_wind  # cart velocity (index 1)
            self._env.unwrapped.state = np.array(state, dtype=np.float32)
            obs = np.array(state, dtype=np.float32)

        if self._elapsed_steps >= self.max_episode_steps:
            truncated = True
            terminated = True

        return obs, reward, terminated, truncated, info

    # Convenience helpers -----------------------------------------------
    def action(self):
        obs = self.observation()
        return self.policy(obs) if self.policy else self.np_random.integers(0, 2)

    def observation(self):
        return np.array(self._env.unwrapped.state, dtype=np.float32)

    def render(self):
        return self._env.render()

    def close(self):
        return self._env.close()

    # Causal graph -------------------------------------------------------
    @property
    def get_graph(self):
        nodes = {0: "Wind(U)", 1: "State(V)", 2: "Action(X)", 3: "Reward(Y)"}
        base = [[0] * 4 for _ in range(4)]
        base[0][1] = 1  # Wind → State
        base[1][2] = 1  # State → Action
        base[1][3] = 1  # State → Reward
        base[2][3] = 1  # Action → Reward
        conf = [[0] * 4 for _ in range(4)]
        return nodes, base, conf


# =============================================================
#  CartPoleWindPCH – thin wrapper exposing see()/do()
# =============================================================
class CartPoleWindPCH(PCH):
    """PCH helper for CartPole with wind."""

    def __init__(self, **kwargs):
        self.env = CartPoleWindSCM(**kwargs)
        super().__init__()  # PCH ctor wires env

    # Observational step under behaviour policy
    def see(self):
        a = self.env.action()
        o, r, term, trunc, info = self.env.step(a)
        return a, o, r, term, trunc, info

    # Interventional step with forced action
    def do(self, action):
        o, r, term, trunc, info = self.env.step(action)
        return o, r, term, trunc, info