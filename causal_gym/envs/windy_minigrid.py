import numpy as np
import minigrid as mg
import gymnasium as gym

from enum import IntEnum
from minigrid.minigrid_env import MiniGridEnv
from gymnasium import spaces
from causal_gym import SCM
from causal_gym.core import PolicyType, ActType, ObsType

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

    def __init__(self, env: MiniGridEnv, policy:PolicyType = None, wind_dist: tuple = WIND_DIST):
        super().__init__(policy, env)
        assert isinstance(env, MiniGridEnv), f"{env} is not a MiniGridEnv!"

        self._wind_dist = wind_dist

        self.actions = MiniGridEnv.Actions

        if policy is not None:
            self._policy = policy
        else:
            # stand still, yield control to the wind
            self._policy = dummy_behavioral_policy

    def __getattribute__(self, name: str) -> np.Any:
        return self._env.__getattribute__(name)
    
    def reset(self, *, seed: int = None, options: dict = None) -> tuple[ObsType, dict]:
        obs, info = self._env.reset(seed = seed, options=options)
        self._internal_state = self._get_internal_state()
        self.rng = np.random.default_rng(seed)
        # self.target_location = (1, 2)
        # self.agent_location = (0, 0)
        # self.agent_location = self.rng.choice([(i, j) for i, j in np.ndindex((self.size, self.size)) if (i, j) != self.target_location])
        self.num_steps = 0
        self._wind_direction = self.rng.choice(len(self._wind_dist), p = self._wind_dist)

        return obs, info + {'wind': self._wind_direction}

    def _get_internal_state(self) -> dict:
        return {"agent_pos": self._env.agent_pos, "agent_dir": self._env.agent_dir, "map": self._env.grid}
    
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
        self._wind_direction = self.rng.choice(len(self._wind_dist), p = self._wind_dist)
        return next_state_tmp, reward, terminated, truncated, info_tmp
    
    def _wind_to_actions(self) -> tuple[int]:
        # AGENT_DIR_TO_STR = {0: ">", 1: "V", 2: "<", 3: "^"}
        agent_dir = self._env.agent_dir
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
        return action, next_state, reward, terminated, truncated, info + {'wind': self._wind_direction}
    
    def do(self, action):
        # only add wind when moving forward, turning around or other move won't be affected by wind
        if action == self.actions.forward:
            wind_actions = self._wind_to_actions()
        else:
            wind_actions = [action]
        next_state, reward, terminated, truncated, info = self._action_sequence(wind_actions)
        # update wind direction
        self._wind_direction = self.rng.choice(len(self._wind_dist), p = self._wind_dist)
        return next_state, reward, terminated, truncated, info + {'wind': self._wind_direction}
