import numpy as np
from typing import Any, Tuple, Dict

from causal_gym import SCM, PCH
from causal_gym.core import ObsType, ActType
import gymnasium as gym
from gymnasium import spaces

from highway_env.vehicle.behavior import IDMVehicle
from highway_env.envs.common.action import DiscreteMetaAction

from PIL import Image, ImageDraw

DANGER_DISTANCE = 30.0 # m

class HighwaySingleStepSCM(SCM):
    ''' Causal environment for the single step highway driving scenario.'''

    def __init__(self, config: Dict[str, Any] = None, seed: int = None):
        super().__init__()

        self.rng = np.random.default_rng(seed)

        # configurations for highway environment
        self.config = config or {}
        self.config.update({
            'offscreen_rendering': True,
            'action': {
                'type': 'DiscreteMetaAction',
                'longitudal': True,
                'lateral': False
            }
        })

        # internal env
        self._env = gym.make('highway-v0', config=self.config, render_mode='rgb_array')
        self._env.reset(seed=seed)

        # set up behavioral policy
        self._meta_actions: DiscreteMetaAction = self._env.unwrapped.action_type
        self._actions_reverse = {v: k for k, v in self._meta_actions.actions.items()}
        # TODO come up with more complex behavioral policy in action()
        # perhaps modify IDMVehicle to use the tail light as a signal for braking

        self._u = None # weather
        self._l = None # front car tail light
        self.x = None # ego velocity
        self.z = None # front car velocity
        self.w = None # if left car is braking
        self.y = None # latent reward

        # TODO determine which option is more appropriate
        self.action_space = self._env.action_space
        # self.observation_space = self._env.observation_space
        # self.action_space = spaces.Discrete(3) # only longitudal used
        self.observation_space = spaces.Dict({
            'x': spaces.Box(low=0.0, high=np.inf, shape=(), dtype=np.float32),
            'z': spaces.Box(low=0.0, high=np.inf, shape=(), dtype=np.float32),
            'w': spaces.Discrete(2)
        })

    def reset(self, *, seed: int = None) -> Tuple[Any, dict]:
        self.rng = np.random.default_rng(seed)
        
        env_obs, env_info = self._env.reset()

        self._u = self.sample_u()
        self.x = self._env.unwrapped.vehicle.velocity[0]
        self.z = self.calc_z()
        self._l = self.calc_l(self.x, self.z)
        self.w = self.calc_w(self._u, self._l)
        self.y = None

        obs = {'x': self.x, 'z': self.z, 'w': self.w}
        info = {'u': self._u, 'l': self._l, 'y': self.y, 'env_obs': env_obs, 'env_info': env_info}
        return obs, info
    
    def action(self, x, z, w, l) -> ActType:
        # placeholder behavioral policy
        # note that w is not used in this policy
        # if tail light is on, slow down
        if l == 1:
            return self._actions_reverse['SLOWER']
        
        # otherwise, copy front car velocity
        if z is None or z > x:
            action = 'FASTER'
        elif z == x:
            action = 'IDLE'
        else:
            action = 'SLOWER'

        return self._actions_reverse[action]

    def observation(self):
        return {'x': self.x, 'z': self.z, 'w': self.w}
    
    def sample_u(self) -> int:
        '''Sample u from P(u).'''
        self._u = self.rng.choice(2, p=[0.38, 0.62])
        return self._u
    
    def calc_l(self, x, z) -> int:
        '''Calculate L from the X <-> Z relation.'''
        if z is not None:
            self._l = 1 if x - z > -0.4 else 0 # should lead to P(L=1) = 0.62
        else:
            self._l = 0 # no front car, so tail light is off
        return self._l
    
    def calc_z(self) -> float:
        '''Calculate Z from environment.'''
        ego = self._env.unwrapped.vehicle
        front_vehicle = self._env.unwrapped.road.neighbour_vehicles(ego)[0]

        if front_vehicle is None:
            self.z = 30.0
            return self.z

        if np.linalg.norm(front_vehicle.position - ego.position) > DANGER_DISTANCE:
            self.z = 30.0
            return self.z

        self.z = front_vehicle.velocity[0]
        return self.z
    
    def calc_w(self, u, l) -> int:
        '''Calculate W from SCM specification.'''
        self.w = u and l
        return self.w
        # ego = self._env.unwrapped.vehicle
        # if ego.lane_index[2] == 0:
        #     self.w = 0 # no left lane, so no left vehicle
        # else:
        #     left_lane_index = (ego.lane_index[0], ego.lane_index[1], ego.lane_index[2] - 1)
        #     left_vehicle = self._env.unwrapped.road.neighbour_vehicles(ego, left_lane_index)[0]
        #     if left_vehicle is not None:
        #         acc = left_vehicle.acceleration(ego_vehicle=left_vehicle)
        #         if acc < -3.0: # -3.0 m/s^2, likely braking (according to Google)
        #             self.w = 1
        #         else: # not braking
        #             self.w = 0
        #     else: # no left vehicle, so no braking
        #         self.w = 0
        # return self.w
    
    def step(self, action: Any, show_reward = False) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:   
        # step actual environment
        observation, reward, terminated, truncated, info = self._env.step(action)
     
        # sample u from P(u)
        if self._u is None:
            self.sample_u()

        # update variables
        self.x = self._env.unwrapped.vehicle.velocity[0]
        self.z = self.calc_z()
        self.calc_l(self.x, self.z)
        self.calc_w(self._u, self._l)

        # following SCM specification
        if self.z is not None:
            indicator = self.x - self.z > -0.4
        else:
            indicator = 0
            
        self.y = (self._u and not indicator) or (not self._u and indicator)
        rew = self.y if show_reward else None

        obs = {'x': self.x, 'z': self.z, 'w': self.w}
        return obs, rew, True, True, {'u': self._u, 'l': self._l, 'y': self.y}

    def render(self) -> ObsType:
        frame = self._env.render()

        front_vehicle = self._env.unwrapped.road.neighbour_vehicles(self._env.unwrapped.vehicle)[0]
        if front_vehicle is None:
            return frame
        
        # add front car tail light indicator
        img = Image.fromarray(frame)
        draw = ImageDraw.Draw(img)

        viewer = self._env.unwrapped.viewer
        x, y = viewer.sim_surface.pos2pix(front_vehicle.position[0], front_vehicle.position[1])

        if self._l == 1:
            r = 4.5
            draw.rectangle((x - 3*r, y - r, x - 2*r, y + r), fill=(255, 100, 0), outline=(255, 100, 0))

        return np.array(img)

    @property
    def get_graph(self) -> Tuple[Dict[int, str], list[list[int]], list[list[int]]]:
        # R-66 Fig 4a
        nodes = {
            0: 'X',
            1: 'Z',
            2: 'L',
            3: 'W',
            4: 'Y'
        }

        base_graph = [
            [0, 0, 0, 0, 1],  # X
            [1, 0, 0, 0, 1],  # Z
            [1, 0, 0, 1, 0],  # L
            [0, 0, 0, 0, 0],  # W
            [0, 0, 0, 0, 0],  # Y
        ]

        conf_graph = [
            [0, 0, 0, 0, 0],  # X
            [0, 0, 0, 0, 0],  # Z
            [0, 0, 0, 0, 0],  # L
            [0, 0, 0, 0, 1],  # W
            [0, 0, 0, 1, 0],  # Y
        ]

        return nodes, base_graph, conf_graph
    
    @property
    def observed_unobserved_vars(self) -> Tuple[list[str], list[str]]:
        return ['X', 'Z', 'W'], ['L', 'Y']

class HighwaySingleStepPCH(PCH):
    '''PCH wrapper for the HighwaySCM env'''

    def __init__(self, config: Dict[str, Any] = None, seed: int = None):
        # initialize underlying SCM
        self.env = HighwaySingleStepSCM(config=config, seed=seed)
        super().__init__()

    def see(self, behavioral_policy=None, show_reward = False) -> Tuple[Any, Any, float, bool, bool, Dict[str, Any]]:
        x = self.env.x
        z = self.env.calc_z()
        w = None # should not be used in a successful policy
        l = self.env.calc_l(self.env.x, z)

        if behavioral_policy is not None:
            action = behavioral_policy(x, z, w, l)
        else:
            action = self.env.action(x, z, w, l)

        obs, reward, terminated, truncated, info = self.env.step(action, show_reward=show_reward)
        return action, obs, reward, terminated, truncated, info

    def do(self, action: Any, show_reward = False) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        return self.env.step(action, show_reward=show_reward)

    def reset(self, *, seed: int = None) -> Tuple[Any, dict]:
        return self.env.reset(seed=seed)

    def render(self) -> Any:
        '''Forced mode rgb_array for this environment.'''
        return self.env.render()