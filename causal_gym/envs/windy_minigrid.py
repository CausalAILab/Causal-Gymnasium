import math
import numpy as np

from typing import Any, Union
from enum import IntEnum
from causal_gym import SCM
from causal_gym.core import PolicyType, ActType, ObsType
from minigrid.minigrid_env import MiniGridEnv
from minigrid.utils.window import Window
from minigrid.core.world_object import WorldObj
from minigrid.utils.rendering import (
    downsample,
    fill_coords,
    highlight_img,
    point_in_rect,
    point_in_triangle,
    rotate_fn,
)
from minigrid.core.constants import OBJECT_TO_IDX, TILE_PIXELS


WIND_DIST = (.1, .1, .1, .1, .6)
# Map agent's direction to short string
AGENT_DIR_TO_STR = {0: ">", 1: "V", 2: "<", 3: "^"}

def dummy_behavioral_policy(*args, **kwargs):
    return 6

class WindyMiniGrid(SCM):
    """
    A windy minigrid world!
    Takes a minigrid environment as input, add winds to the transitions.
    By default, the behavioral policy (policy) is none and the behavioral agent will move by the wind.

    ## Wind Direction

    | Num | Name         | Action       |
    |-----|--------------|--------------|
    | 0   | right        | To east      |
    | 1   | down         | To south     |
    | 2   | left         | To west      |
    | 3   | up           | To north     |
    | 4   | still        | No wind      |
    
    ## Action Space

    | Num | Name         | Action       |
    |-----|--------------|--------------|
    | 0   | left         | Turn left    |
    | 1   | right        | Turn right   |
    | 2   | forward      | Move forward |
    | 3   | pickup       | Pick up obj  |
    | 4   | drop         | Drop obj     |
    | 5   | toggle       | Toggle button|
    | 6   | done         | Do nothing   |

    ## Observation Encoding

    - Each tile is encoded as a 3 dimensional tuple:
        `(OBJECT_IDX, COLOR_IDX, STATE)`
    - `OBJECT_TO_IDX` and `COLOR_TO_IDX` mapping can be found in
        [minigrid/minigrid.py](minigrid/minigrid.py)
    - `STATE` refers to the door state with 0=open, 1=closed and 2=locked

    ## Rewards

    A reward of '1' is given for success, and '0' for failure.

    ## Termination

    The episode ends if any one of the following conditions is met:

    1. The agent reaches the goal.
    2. Timeout (see `max_steps`).

    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    # Enumeration of possible actions
    class Actions(IntEnum):
        # Turn left, turn right, move forward
        # turn around should be a combination of turn left/right twice then move forward
        left = 0
        right = 1
        forward = 2
        # Pick up an object
        pickup = 3
        # Drop an object
        drop = 4
        # Toggle/activate an object
        toggle = 5
        # Done completing task
        done = 6

    def __init__(self, env: MiniGridEnv, policy:PolicyType = None, show_wind: bool=False, wind_dist: tuple = WIND_DIST):
        assert issubclass(env.unwrapped.__class__, MiniGridEnv), f"Input env must be of type 'MiniGridEnv'!"
        self._env = env
        self._wind_dist = wind_dist
        self.actions = MiniGridEnv.Actions
        # whether to display wind direction at rendering time
        self._show_wind = show_wind
        if policy is not None:
            self._policy = policy
        else:
            # stand still, yield control to the wind
            self._policy = dummy_behavioral_policy

    def __getattr__(self, name: str) -> Any:
        if name == "_np_random":
            raise AttributeError(
                "Can't access `_np_random` of a wrapper, use `self.unwrapped._np_random` or `self.np_random`."
            )
        elif name.startswith("_"):
            raise AttributeError(f"accessing private attribute '{name}' is prohibited")
        return self._env.__getattr__(name)
    
    def reset(self, *, seed: int = None, options: dict = None) -> tuple[ObsType, dict]:
        obs, info = self._env.reset(seed = seed, options=options)
        self._internal_state = self._get_internal_state()
        self.rng = np.random.default_rng(seed)
        self._wind_direction = self.rng.choice(len(self._wind_dist), p = self._wind_dist)
        info['wind'] = self._wind_direction
        return obs, info

    def _get_internal_state(self) -> dict:
        return {"agent_pos": self._env.unwrapped.agent_pos, "agent_dir": self._env.unwrapped.agent_dir, "map": self._env.unwrapped.grid}
    
    @property
    def get_graph(self,) -> tuple[dict[int, str], list[list[int]], list[list[int]]]:
        """Return the causal diagram of the environment.
        Returns:
            Nodes: a dictionary mapping from node index ([0, N-1]) to each node's semantic meaning.
            base_graph: an extended adjacent matrix representation of the directed graphical structure.  
                G[i,j] = -1 i<-j
                G[i,j] = 0 i j
                G[i,j] = 1 i->j
            conf_graph: a matrix representing the existence of confounders between nodes.
                G[i, j] = 0 no confounder
                G[i, j] = 1 i<->j
        """
        # TODO: find a way to automatically generate graph from code?
        raise NotImplementedError

    def action(self):
        """sample action from the behavioral policy
        """
        return self._policy(self._internal_state, self._wind_direction)
    
    def observation(self):
        return self._env.render()

    def _action_sequence(self, seq: tuple[int]):
        reward = 0
        terminated = False
        truncated = False
        for act in seq:
            next_state_tmp, reward_tmp, terminated_tmp, truncated_tmp, info_tmp = self._env.step(act)
            reward += reward
            terminated = terminated or terminated_tmp
            truncated = truncated or truncated_tmp
            if terminated or truncated:
                break
        return next_state_tmp, reward, terminated, truncated, info_tmp
    
    def _wind_to_actions(self) -> tuple[int]:
        # AGENT_DIR_TO_STR = {0: ">", 1: "V", 2: "<", 3: "^"}
        agent_dir = self._env.unwrapped.agent_dir
        if agent_dir == self._wind_direction:
            # following wind
            return [self.actions.forward, self.actions.forward]
        elif (agent_dir - 2) % 4 == self._wind_direction:
            # going against the wind
            return [self.actions.done]
        else:
            first_turn = self.actions.left if (agent_dir - self._wind_direction == 1) or (agent_dir - self._wind_direction == -3) else self.actions.right
            second_turn = self.actions.right if first_turn == self.actions.left else self.actions.left
            # wind blowing sideways
            return [self.actions.forward, first_turn, self.actions.forward, second_turn] 
    
    def see(self):
        # only add wind when moving forward, turning around or other move won't be affected by wind
        action = self.action()
        if action == self.actions.forward:
            wind_actions = self._wind_to_actions()
        else:
            wind_actions = [action]
        next_state, reward, terminated, truncated, info = self._action_sequence(wind_actions)
        # update wind direction
        self._wind_direction = self.rng.choice(len(self._wind_dist), p = self._wind_dist)
        info['wind'] = self._wind_direction
        return action, next_state, reward, terminated, truncated, info
    
    def do(self, action):
        # only add wind when moving forward, turning around or other move won't be affected by wind
        if action == self.actions.forward:
            wind_actions = self._wind_to_actions()
        else:
            wind_actions = [action]
        next_state, reward, terminated, truncated, info = self._action_sequence(wind_actions)
        # update wind direction
        info['wind'] = self._wind_direction
        return next_state, reward, terminated, truncated, info
    
    @classmethod
    def render_wind_tile(
        cls,
        obj: Union[WorldObj, None],
        agent_dir: Union[int, None] = None,
        highlight: bool = False,
        tile_size: int = TILE_PIXELS,
        subdivs: int = 3,
    ) -> np.ndarray:
        """
        Render the wind dir tile
        """

        # Hash map lookup key for the cache
        key: tuple[Any, ...] = (agent_dir, highlight, tile_size)
        key = obj.encode() + key if obj else key

        img = np.zeros(
            shape=(tile_size * subdivs, tile_size * subdivs, 3), dtype=np.uint8
        )

        # Draw the grid lines (top and left edges)
        fill_coords(img, point_in_rect(0, 0.031, 0, 1), (100, 100, 100))
        fill_coords(img, point_in_rect(0, 1, 0, 0.031), (100, 100, 100))

        if obj is not None:
            obj.render(img)

        # Overlay the agent on top
        if agent_dir is not None:
            tri_fn = point_in_triangle(
                (0.12, 0.19),
                (0.87, 0.50),
                (0.12, 0.81),
            )

            # Rotate the agent based on its direction
            tri_fn = rotate_fn(tri_fn, cx=0.5, cy=0.5, theta=0.5 * math.pi * agent_dir)
            fill_coords(img, tri_fn, (115, 193, 255))

        # Highlight the cell if needed
        if highlight:
            highlight_img(img)

        # Downsample the image to perform supersampling/anti-aliasing
        img = downsample(img, subdivs)

        return img
    
    def render(self):
        img = self._env.unwrapped.get_frame(self._env.unwrapped.highlight, self._env.unwrapped.tile_size, self._env.unwrapped.agent_pov)

        if self._show_wind:
            # Wind direction is rendered as a blue arrow on the upper left corner of the map 
            # where there is a wall tile
            tile_img = WindyMiniGrid.render_wind_tile(
                obj=self._env.unwrapped.grid.get(0, 0),
                agent_dir=self._wind_direction,
                highlight=0,
                tile_size=self._env.unwrapped.tile_size,
            )
            i = 0
            j = 0
            ymin = j * self._env.unwrapped.tile_size
            ymax = (j + 1) * self._env.unwrapped.tile_size
            xmin = i * self._env.unwrapped.tile_size
            xmax = (i + 1) * self._env.unwrapped.tile_size
            img[ymin:ymax, xmin:xmax, :] = tile_img

        if self._env.unwrapped.render_mode == "human":
            if self.window is None:
                self.window = Window("minigrid")
                self.window.show(block=False)
            self.window.set_caption(self.mission)
            self.window.show_img(img)
        elif self._env.unwrapped.render_mode == "rgb_array":
            return img
