from __future__ import annotations
import numpy as np

from typing import Any, SupportsFloat
from minigrid.minigrid_env import MiniGridEnv
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Goal, Lava, Wall, Ball
from minigrid.core.grid import Grid
from gymnasium.core import WrapperActType, ActType, ObsType, Env, Wrapper

class Coin(Ball):
    def __init__(self, color="yellow"):
        super().__init__()
        self.color = color

    def can_overlap(self):
        return True


class CustomCrossingEnv(MiniGridEnv):
    """
    LavaCrossing Env from MiniGrid but only use vertical obstacles 
    and always have two openings at the top and bottom!

    ## Description

    We support three different difficulty levels:
    - `easy` - there is only a vertical lava river in the middle with openings 
               on both top and bottom side.
    - `hard` - there is a lava lake in the middle.
    - `extreme` - there are two tunnels surrounded by lava, the upper side one 
               has gold coins while the lower one has nothing but easier to go
               through as an interventional agent.


    Depending on the `obstacle_type` parameter:
    - `Lava` - The agent has to reach the green goal square on the other corner
        of the room while avoiding rivers of deadly lava which terminate the
        episode in failure. Each lava stream runs across the room either
        horizontally or vertically, and has a single crossing point which can be
        safely used; Luckily, a path to the goal is guaranteed to exist. This
        environment is useful for studying safety and safe exploration.
    - otherwise - Similar to the `LavaCrossing` environment, the agent has to
        reach the green goal square on the other corner of the room, however
        lava is replaced by walls. This MDP is therefore much easier and maybe
        useful for quickly testing your algorithms.

    ## Mission Space
    Depending on the `obstacle_type` parameter:
    - `Lava` - "avoid the lava and get to the green goal square"
    - otherwise - "find the opening and get to the green goal square"

    ## Action Space

    | Num | Name         | Action       |
    |-----|--------------|--------------|
    | 0   | left         | Turn left    |
    | 1   | right        | Turn right   |
    | 2   | forward      | Move forward |
    | 3   | pickup       | Unused       |
    | 4   | drop         | Unused       |
    | 5   | toggle       | Unused       |
    | 6   | done         | Unused       |

    ## Observation Encoding

    - Each tile is encoded as a 3 dimensional tuple:
        `(OBJECT_IDX, COLOR_IDX, STATE)`
    - `OBJECT_TO_IDX` and `COLOR_TO_IDX` mapping can be found in
        [minigrid/core/constants.py](minigrid/core/constants.py)
    - `STATE` refers to the door state with 0=open, 1=closed and 2=locked

    ## Rewards

    A reward of '1 - 0.9 * (step_count / max_steps)' is given for success, and '0' for failure.

    ## Termination

    The episode ends if any one of the following conditions is met:

    1. The agent reaches the goal.
    2. The agent falls into lava.
    3. Timeout (see `max_steps`).

    ## Registered Name
    Need to set parameters, by default it's of size 9 with 1 crossing.

    S: size of the map SxS.
    N: number of valid crossings across lava or walls from the starting position
    to the goal

    - `Custom-LavaCrossing-v0`

    """

    def __init__(
        self,
        size=9,
        num_crossings=1,
        obstacle_type=Lava,
        agent_start_pos=(1, 1),
        agent_start_dir=0,
        max_steps: int | None = None,
        mode = 'easy',
        **kwargs,
    ):
        assert mode in ['easy', 'hard', 'extreme', 'maze', 'maze2'], f"mode should be only one of the ['easy', 'hard', 'extreme', 'maze', 'maze2']."
        self.num_crossings = num_crossings
        self.obstacle_type = obstacle_type
        self.goal_position = None
        self.size = size
        self.mode = mode

        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir

        if obstacle_type == Lava:
            mission_space = MissionSpace(mission_func=self._gen_mission_lava)
        else:
            mission_space = MissionSpace(mission_func=self._gen_mission)

        if max_steps is None:
            max_steps = 4 * size**2

        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            see_through_walls=False,  # Set this to True for maximum speed
            max_steps=max_steps,
            **kwargs,
        )

    @staticmethod
    def _gen_mission_lava():
        return "avoid the lava and get to the green goal square"

    @staticmethod
    def _gen_mission():
        return "find the opening and get to the green goal square"

    def _gen_grid(self, width, height):
        assert width % 2 == 1 and height % 2 == 1  # odd size

        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place a goal square in the bottom-right corner
        self.put_obj(Goal(), width - 2, height - 2)
        self.goal_position = (width - 2, height - 2)

        # Place obstacles
        if self.mode == 'easy':
            obstacle_pos = [(self.size//2, j) for j in range(2, height - 2)]
        elif self.mode == 'hard':
            obstacle_pos = [(i, j) for i in range(2, width - 2) for j in range(3, height - 3)]
        elif self.mode == 'extreme':
            obstacle_pos = [(i, j) for i in range(2, width - 2) for j in range(3, height - 3)]
            obstacle_pos += [(i, 1) for i in range(2, width - 2)]
            obstacle_pos += [(i, height - 3) for i in range(2, width - 2)]
        elif self.mode == 'maze':
            # maze
            map = np.array([0,0,0,1,0,0,0, 0,1,1,1,0,1,1, 0,0,0,0,0,0,1, 0,1,0,1,1,1,1, 0,1,0,1,0,0,0, 0,1,0,1,0,1,0, 0,0,0,0,0,0,0])
            obstacle_pos = []
            for i, w in enumerate(map):
                x, y = self._flat_to_loc(i)
                if w == 1:
                    obstacle_pos.append((y+1, x+1))
        else:
            # maze2
            map = np.array([0,1,0,0,0,0,0, 0,0,0,1,0,1,0, 0,0,0,0,0,1,0, 0,1,0,1,1,1,0, 0,1,0,0,0,0,0, 0,1,0,1,1,1,0, 0,0,0,0,0,0,0])
            obstacle_pos = []
            for i, w in enumerate(map):
                x, y = self._flat_to_loc(i)
                if w == 1:
                    obstacle_pos.append((y+1, x+1))
        
        for i, j in obstacle_pos:
            self.put_obj(self.obstacle_type(), i, j)

        # Place coins
        if self.mode == 'extreme':
            for i in range(2, width - 2):
                self.put_obj(Coin(), i, 2)
        elif self.mode == 'maze':
            self.put_obj(Coin(), self.size - 2, self.size - 3)
            self.put_obj(Coin(), self.size - 2, 1)
        elif self.mode == 'maze2':
            self.put_obj(Coin(), self.size - 2, self.size - 3)
            self.put_obj(Coin(), 3, 2)
            self.put_obj(Coin(), 3, 5)

        # Place the agent 
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()
        # self.agent_pos = np.array((1, 1))
        # self.agent_dir = 0

        self.mission = (
            "avoid the lava and get to the green goal square"
            if self.obstacle_type == Lava
            else "find the opening and get to the green goal square"
        )

    def _flat_to_loc(self, flat_loc):
        return [flat_loc // (self.size - 2), flat_loc % (self.size - 2)]

    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        self.step_count += 1

        reward = 0
        terminated = False
        truncated = False

        # Get the position in front of the agent
        fwd_pos = self.front_pos

        # Get the contents of the cell in front of the agent
        fwd_cell = self.grid.get(*fwd_pos)

        # Rotate left
        if action == self.actions.left:
            self.agent_dir -= 1
            if self.agent_dir < 0:
                self.agent_dir += 4

        # Rotate right
        elif action == self.actions.right:
            self.agent_dir = (self.agent_dir + 1) % 4

        # Move forward
        elif action == self.actions.forward:
            if fwd_cell is None or fwd_cell.can_overlap():
                self.agent_pos = tuple(fwd_pos)
                if fwd_cell is not None and fwd_cell.type == "ball" and fwd_cell.color == "yellow":
                    reward = .2
                    # clear the coin
                    self.grid.set(fwd_pos[0], fwd_pos[1], None)
            if fwd_cell is not None and fwd_cell.type == "goal":
                terminated = True
                reward = self._reward()
            if fwd_cell is not None and fwd_cell.type == "lava":
                terminated = True

        # Pick up an object
        elif action == self.actions.pickup:
            if fwd_cell and fwd_cell.can_pickup():
                if self.carrying is None:
                    self.carrying = fwd_cell
                    self.carrying.cur_pos = np.array([-1, -1])
                    self.grid.set(fwd_pos[0], fwd_pos[1], None)

        # Drop an object
        elif action == self.actions.drop:
            if not fwd_cell and self.carrying:
                self.grid.set(fwd_pos[0], fwd_pos[1], self.carrying)
                self.carrying.cur_pos = fwd_pos
                self.carrying = None

        # Toggle/activate an object
        elif action == self.actions.toggle:
            if fwd_cell:
                fwd_cell.toggle(self, fwd_pos)

        # Done action (not used by default)
        elif action == self.actions.done:
            pass

        else:
            raise ValueError(f"Unknown action: {action}")

        if self.step_count >= self.max_steps:
            truncated = True

        if self.render_mode == "human":
            self.render()

        obs = self.gen_obs()

        return obs, reward, terminated, truncated, {}
