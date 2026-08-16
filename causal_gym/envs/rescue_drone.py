"""Draft Rescue Drone causal environment.

The module follows the engineering split used by ``lunar_lander.py``:

* :class:`RescueDroneSCM` owns the world mechanisms and exogenous noise.
* :class:`RescueDronePCH` exposes observational, interventional, and
  intention-conditioned counterfactual interactions.

Unlike a conventional grid-world, one stage-level exogenous variable is
sampled exactly once and reused by the natural action, transition, and reward
mechanisms.  This makes the unobserved-confounding claim executable rather
than only graphical.

This file is intentionally a review draft.  It is not registered in
``causal_gym.envs.__init__`` yet.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any, Callable

import numpy as np
from gymnasium import spaces

from ..core import Graph, PCH, SCM, Task
from ..core.types import PolicyType

__all__ = ["RescueDroneSCM", "RescueDronePCH"]


@dataclass(frozen=True)
class DroneState:
    """Full endogenous mission state S_i.

    The public observation is a normalized projection of this state.  The
    stage-level gust and hazard are exogenous and therefore are not stored in
    this object.
    """

    drone_x: int
    drone_y: int
    battery: int
    victim_x: int
    victim_y: int
    victim_health: int
    rescued: bool = False


BehaviorPolicy = Callable[[np.ndarray, dict[str, int]], int]


class RescueDroneSCM(SCM[PolicyType, np.ndarray, int]):
    """Finite-horizon rescue mission with explicit stage-level confounding.

    Structural interpretation for stage ``i``::

        U_i = (gust_i, hazard_i)              exogenous
        O_i <- f_O(S_i)                       observation
        X_i <- f_X(O_i, U_i)                  natural action
        S_{i+1} <- f_S(S_i, X_i, U_i)         transition
        Y_i <- f_Y(S_i, X_i, U_i)             reward

    ``U_i`` is sampled once per stage.  The natural behavior policy can sense
    it, while the learner-facing observation cannot.  Because the same
    ``U_i`` also changes the transition and reward, observational action-
    outcome associations may be confounded.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 4}

    ACTION_NAMES = {
        0: "hover",
        1: "up",
        2: "right",
        3: "down",
        4: "left",
    }
    ACTION_VECTORS = {
        0: (0, 0),
        1: (0, -1),
        2: (1, 0),
        3: (0, 1),
        4: (-1, 0),
    }
    GUST_VECTORS = {
        0: (0, 0),
        1: (0, -1),
        2: (1, 0),
        3: (0, 1),
        4: (-1, 0),
    }

    def __init__(
        self,
        *,
        grid_size: int = 7,
        max_episode_steps: int = 40,
        max_battery: int = 60,
        max_victim_health: int = 30,
        gust_probability: float = 0.35,
        hazard_probability: float = 0.20,
        render_mode: str | None = None,
        policy: BehaviorPolicy | None = None,
        include_latent_in_info: bool = False,
    ) -> None:
        super().__init__()
        if grid_size < 5:
            raise ValueError("grid_size must be at least 5")
        if max_episode_steps < 1 or max_battery < 1 or max_victim_health < 1:
            raise ValueError("episode, battery, and health limits must be positive")
        if not 0.0 <= gust_probability <= 1.0:
            raise ValueError("gust_probability must lie in [0, 1]")
        if not 0.0 <= hazard_probability <= 1.0:
            raise ValueError("hazard_probability must lie in [0, 1]")
        if render_mode not in (None, "rgb_array"):
            raise ValueError("render_mode must be None or 'rgb_array'")

        self.grid_size = int(grid_size)
        self.max_episode_steps = int(max_episode_steps)
        self.max_battery = int(max_battery)
        self.max_victim_health = int(max_victim_health)
        self.gust_probability = float(gust_probability)
        self.hazard_probability = float(hazard_probability)
        self.render_mode = render_mode
        self.include_latent_in_info = bool(include_latent_in_info)

        self.action_space = spaces.Discrete(len(self.ACTION_NAMES))
        self.observation_space = spaces.Box(
            low=np.zeros(6, dtype=np.float32),
            high=np.ones(6, dtype=np.float32),
            dtype=np.float32,
        )

        self.policy: BehaviorPolicy = policy or self._default_behavior_policy
        self.state: DroneState | None = None
        self.current_u: dict[str, int] | None = None
        self._pending_u: dict[str, int] | None = None
        self._last_obs: np.ndarray | None = None
        self._last_action: int | None = None
        self._last_natural_action: int | None = None
        self._elapsed_steps = 0

    # ------------------------------------------------------------------
    # Exogenous distribution P(U_i)
    # ------------------------------------------------------------------
    def sample_u(self) -> dict[str, int]:
        """Sample one i.i.d. exogenous realization for the next stage."""

        no_gust = 1.0 - self.gust_probability
        directional = self.gust_probability / 4.0
        gust_code = int(
            self.np_random.choice(
                5,
                p=[no_gust, directional, directional, directional, directional],
            )
        )
        gust_x, gust_y = self.GUST_VECTORS[gust_code]
        return {
            "gust_x": int(gust_x),
            "gust_y": int(gust_y),
            "hazard": int(self.np_random.random() < self.hazard_probability),
        }

    def _prepare_u(self) -> dict[str, int]:
        """Sample at most once, so all mechanisms share the same U_i."""

        if self._pending_u is None:
            self._pending_u = self.sample_u()
        return dict(self._pending_u)

    # ------------------------------------------------------------------
    # Gym / SCM interface
    # ------------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed, options=options)
        options = options or {}

        drone_position = tuple(
            options.get("drone_position", (self.grid_size // 2, self.grid_size - 1))
        )
        victim_position = options.get("victim_position")
        if victim_position is None:
            victim_position = (
                int(self.np_random.integers(0, self.grid_size)),
                int(self.np_random.integers(0, max(1, self.grid_size // 2))),
            )
        victim_position = tuple(victim_position)
        self._validate_position(drone_position, "drone_position")
        self._validate_position(victim_position, "victim_position")
        if drone_position == victim_position:
            raise ValueError("drone and victim must start in different cells")

        self.state = DroneState(
            drone_x=int(drone_position[0]),
            drone_y=int(drone_position[1]),
            battery=self.max_battery,
            victim_x=int(victim_position[0]),
            victim_y=int(victim_position[1]),
            victim_health=self.max_victim_health,
        )
        self.current_u = None
        self._pending_u = None
        self._last_action = None
        self._last_natural_action = None
        self._elapsed_steps = 0
        self._last_obs = self._state_to_observation(self.state)

        info: dict[str, Any] = {
            "mission": {
                "grid_size": self.grid_size,
                "max_battery": self.max_battery,
                "max_victim_health": self.max_victim_health,
            }
        }
        return self._last_obs.copy(), info

    def action(self) -> int:
        """Evaluate the natural action mechanism X_i <- f_X(O_i, U_i)."""

        if self.state is None:
            raise RuntimeError("Call reset() before action().")
        u = self._prepare_u()
        action = int(self.policy(self.observation(), u))
        self._validate_action(action)
        self._last_natural_action = action
        return action

    def observation(self) -> np.ndarray:
        """Return O_i, which deliberately excludes the latent U_i."""

        if self._last_obs is None:
            raise RuntimeError("Call reset() before observation().")
        return self._last_obs.copy()

    def step(self, action: int):
        if self.state is None:
            raise RuntimeError("Call reset() before step().")
        action = int(action)
        self._validate_action(action)
        u = self._prepare_u()

        next_state, reward, terminated, transition_info = self._transition(
            self.state, action, u
        )
        self._elapsed_steps += 1
        truncated = bool(
            not terminated and self._elapsed_steps >= self.max_episode_steps
        )

        self.state = next_state
        self.current_u = dict(u)
        self._pending_u = None
        self._last_action = action
        self._last_obs = self._state_to_observation(next_state)

        info: dict[str, Any] = {
            **transition_info,
            "elapsed_steps": self._elapsed_steps,
            "action_name": self.ACTION_NAMES[action],
        }
        if truncated:
            info["end_reason"] = "time_limit"
        if self.include_latent_in_info:
            info["exogenous"] = dict(u)

        return self._last_obs.copy(), float(reward), terminated, truncated, info

    def close(self):
        return None

    # ------------------------------------------------------------------
    # Structural functions F
    # ------------------------------------------------------------------
    def _transition(
        self, state: DroneState, action: int, u: dict[str, int]
    ) -> tuple[DroneState, float, bool, dict[str, Any]]:
        """Pure implementation of f_S and f_Y for one stage."""

        action_dx, action_dy = self.ACTION_VECTORS[action]
        gust_x, gust_y = u["gust_x"], u["gust_y"]
        next_x = int(np.clip(state.drone_x + action_dx + gust_x, 0, self.grid_size - 1))
        next_y = int(np.clip(state.drone_y + action_dy + gust_y, 0, self.grid_size - 1))

        energy_cost = 1 + int(action != 0) + abs(gust_x) + abs(gust_y)
        next_battery = max(0, state.battery - energy_cost)
        victim_was_alive = state.victim_health > 0
        rescued = bool(
            victim_was_alive
            and next_x == state.victim_x
            and next_y == state.victim_y
        )
        health_loss = 0 if rescued else 1 + int(u["hazard"])
        next_health = max(0, state.victim_health - health_loss)

        next_state = replace(
            state,
            drone_x=next_x,
            drone_y=next_y,
            battery=next_battery,
            victim_health=next_health,
            rescued=rescued,
        )
        terminated = bool(rescued or next_battery == 0 or next_health == 0)

        old_distance = abs(state.drone_x - state.victim_x) + abs(
            state.drone_y - state.victim_y
        )
        new_distance = abs(next_x - state.victim_x) + abs(next_y - state.victim_y)
        if rescued:
            reward = 1.0
            end_reason = "rescued"
        elif next_battery == 0:
            reward = -1.0
            end_reason = "battery_depleted"
        elif next_health == 0:
            reward = -1.0
            end_reason = "victim_lost"
        else:
            reward = (
                -0.02
                - 0.01 * energy_cost
                - 0.005 * health_loss
                + 0.05 * (old_distance - new_distance)
            )
            end_reason = None

        info = {
            "drone_position": (next_x, next_y),
            "battery": next_battery,
            "victim_health": next_health,
            "rescued": rescued,
            "energy_cost": energy_cost,
            "end_reason": end_reason,
        }
        return next_state, float(reward), terminated, info

    def _state_to_observation(self, state: DroneState) -> np.ndarray:
        denominator = float(self.grid_size - 1)
        return np.asarray(
            [
                state.drone_x / denominator,
                state.drone_y / denominator,
                state.battery / self.max_battery,
                state.victim_x / denominator,
                state.victim_y / denominator,
                state.victim_health / self.max_victim_health,
            ],
            dtype=np.float32,
        )

    def _default_behavior_policy(
        self, observation: np.ndarray, u: dict[str, int]
    ) -> int:
        """A simple natural pilot that senses the otherwise hidden gust."""

        assert self.state is not None
        desired_x = self.state.victim_x - self.state.drone_x - u["gust_x"]
        desired_y = self.state.victim_y - self.state.drone_y - u["gust_y"]
        if desired_x == 0 and desired_y == 0:
            return 0
        if abs(desired_x) > abs(desired_y):
            return 2 if desired_x > 0 else 4
        return 3 if desired_y > 0 else 1

    # ------------------------------------------------------------------
    # Full-SCM diagnostic helpers for theory demonstrations
    # ------------------------------------------------------------------
    def unit_counterfactuals(self) -> dict[str, Any]:
        """Evaluate every action for the same state and same exogenous unit.

        This is a simulator-level L3 diagnostic, not observational data exposed
        to a learning agent.  It does not advance the environment.
        """

        if self.state is None:
            raise RuntimeError("Call reset() before unit_counterfactuals().")
        u = self._prepare_u()
        intended_action = int(self.policy(self.observation(), u))
        self._validate_action(intended_action)
        alternatives: dict[int, dict[str, Any]] = {}
        for action in range(self.action_space.n):
            next_state, reward, terminated, info = self._transition(
                self.state, action, u
            )
            alternatives[action] = {
                "action_name": self.ACTION_NAMES[action],
                "next_state": asdict(next_state),
                "reward": reward,
                "terminated": terminated,
                "end_reason": info["end_reason"],
            }
        return {
            "state": asdict(self.state),
            "exogenous": dict(u),
            "intended_action": intended_action,
            "intended_action_name": self.ACTION_NAMES[intended_action],
            "alternatives": alternatives,
        }

    def render(self, show_wind: bool = False, show_natural_action: bool = False):
        if self.render_mode is None:
            return None
        if self.state is None:
            raise RuntimeError("Call reset() before render().")

        cell = 48
        canvas = np.full(
            (self.grid_size * cell, self.grid_size * cell, 3), 244, dtype=np.uint8
        )
        canvas[::cell, :, :] = 205
        canvas[:, ::cell, :] = 205

        def paint_cell(x: int, y: int, color: tuple[int, int, int], inset: int) -> None:
            x0, y0 = x * cell + inset, y * cell + inset
            x1, y1 = (x + 1) * cell - inset, (y + 1) * cell - inset
            canvas[y0:y1, x0:x1] = color

        paint_cell(self.state.victim_x, self.state.victim_y, (225, 82, 82), 10)
        paint_cell(self.state.drone_x, self.state.drone_y, (55, 119, 190), 7)
        if self.state.rescued:
            paint_cell(self.state.drone_x, self.state.drone_y, (72, 161, 104), 4)

        if show_wind:
            u = self.current_u or self._pending_u
            if u is not None:
                wind_color = (236, 180, 62) if (u["gust_x"] or u["gust_y"]) else (150, 150, 150)
                canvas[0:6, :] = wind_color
        if show_natural_action and self._last_natural_action is not None:
            palette = np.asarray(
                [
                    (130, 130, 130),
                    (111, 78, 153),
                    (46, 139, 87),
                    (201, 111, 47),
                    (73, 101, 171),
                ],
                dtype=np.uint8,
            )
            canvas[-6:, :] = palette[self._last_natural_action]
        return canvas

    @property
    def get_graph(self) -> Graph:
        """Return the latent projection implied by the implemented functions."""

        nodes = [
            {"name": "S", "label": "Mission state"},
            {"name": "O", "label": "Learner observation"},
            {"name": "X", "label": "Drone action"},
            {"name": "Y", "label": "Reward"},
            {"name": "S'", "label": "Next mission state"},
        ]
        edges = [
            {"from_": "S", "to_": "O", "type_": "directed"},
            {"from_": "O", "to_": "X", "type_": "directed"},
            {"from_": "S", "to_": "Y", "type_": "directed"},
            {"from_": "X", "to_": "Y", "type_": "directed"},
            {"from_": "S", "to_": "S'", "type_": "directed"},
            {"from_": "X", "to_": "S'", "type_": "directed"},
            # U_i is shared by f_X, f_Y, and f_S.  Graph stores bows by
            # reversing the bidirected edge internally when needed.
            {"from_": "X", "to_": "Y", "type_": "bidirected"},
            {"from_": "X", "to_": "S'", "type_": "bidirected"},
            {"from_": "Y", "to_": "S'", "type_": "bidirected"},
        ]
        return Graph(
            nodes=nodes,
            edges=edges,
            metadata={
                "latent": "U_i = (gust_i, hazard_i)",
                "mechanisms": [
                    "O_i <- f_O(S_i)",
                    "X_i <- f_X(O_i, U_i)",
                    "S_{i+1} <- f_S(S_i, X_i, U_i)",
                    "Y_i <- f_Y(S_i, X_i, U_i)",
                ],
            },
        )

    def _validate_position(self, position: tuple[Any, ...], name: str) -> None:
        if len(position) != 2 or any(not isinstance(v, (int, np.integer)) for v in position):
            raise ValueError(f"{name} must be a pair of integers")
        if any(v < 0 or v >= self.grid_size for v in position):
            raise ValueError(f"{name} must lie inside the grid")

    def _validate_action(self, action: int) -> None:
        if not self.action_space.contains(action):
            raise ValueError(f"invalid action {action}; expected 0-{self.action_space.n - 1}")


class RescueDronePCH(PCH):
    """PCH facade mirroring the Lunar Lander environment architecture."""

    def __init__(self, **kwargs) -> None:
        task = kwargs.pop("task", Task())
        self.env: RescueDroneSCM = RescueDroneSCM(**kwargs)
        super().__init__(env=self.env, task=task)

    def see(self, see_policy=None):
        """L1: let the natural action mechanism select X_i."""

        if see_policy is None:
            action = self.env.action()
        else:
            if not callable(see_policy):
                raise TypeError("see_policy must be callable")
            action = int(see_policy(self.env.observation()))
        obs, reward, terminated, truncated, info = self.env.step(action)
        info = dict(info)
        info["natural_action"] = action
        return obs, reward, terminated, truncated, info

    def do(self, do_policy):
        """L2: replace f_X with a learner-supplied decision rule."""

        if not callable(do_policy):
            raise TypeError("do_policy must be callable: do(lambda obs: action)")
        action = int(do_policy(self.env.observation()))
        obs, reward, terminated, truncated, info = self.env.step(action)
        info = dict(info)
        info["action"] = action
        return obs, reward, terminated, truncated, info

    def ctf_do(self, ctf_policy):
        """L3-style interception: observe intention, then replace the action."""

        if not callable(ctf_policy):
            raise TypeError(
                "ctf_policy must be callable: ctf_do(lambda obs, intended: action)"
            )
        intended_action = self.env.action()
        action = int(ctf_policy(self.env.observation(), intended_action))
        obs, reward, terminated, truncated, info = self.env.step(action)
        info = dict(info)
        info["natural_action"] = intended_action
        info["action"] = action
        return obs, reward, terminated, truncated, info

    def render(self, show_wind: bool = False, show_natural_action: bool = False):
        return self.env.render(
            show_wind=show_wind,
            show_natural_action=show_natural_action,
        )
