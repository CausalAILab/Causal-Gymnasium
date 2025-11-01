import numpy as np
import gymnasium as gym
from gymnasium_robotics.envs.adroit_hand.adroit_door import AdroitHandDoorEnv

from ..core import SCM, PCH, Task, Graph

class AdroitHandDoorSCM(SCM):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 100,
    }
    
    def __init__(
        self, 
        max_episode_steps=400, 
        policy=None,
        **kwargs
    ):
        kwargs.setdefault("max_episode_steps", max_episode_steps)
        kwargs.setdefault("render_mode", "rgb_array")
        self._env = gym.make(
            'AdroitHandDoor-v1', 
            **kwargs
        )
        
        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space
        self.spec = self._env.spec

        if policy is not None:
            self.policy = policy 
        else:
            self.policy = lambda obs: self._env.action_space.sample()
        
    def reset(self, **kwargs):
        obs, info = self._env.reset(**kwargs)
        self.current_obs = obs.copy()
        return obs, info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self._env.step(action)
        self.current_obs = obs.copy()
        return obs, reward, terminated, truncated, info
    
    def render(self):
        return self._env.render()
    
    def observation(self):
        return self.current_obs
    
    def action(self):
        return self.policy(self.current_obs)    
    
    @property
    def get_graph(self):
        # nodes = {0: "Friction(U)", 1: "State(S)", 2: "Action(X)", 3: "Reward(Y)", 4: "Next_State(S')"}
        # base = [[0] * 5 for _ in range(5)]
        # base[0][2] = 1  # U → X
        # base[0][4] = 1  # U → S'
        # base[1][2] = 1  # S → X
        # base[1][3] = 1  # S → Y
        # base[2][3] = 1  # X → Y
        # base[1][4] = 1  # S → S'
        # base[2][4] = 1  # X → S'
        # conf = [[0] * 5 for _ in range(5)]
        # conf[2][4] = 1
        # conf[4][2] = 1
        # return nodes, base, conf
        raise NotImplementedError("Graph structure is not defined for AdroitHandDoorSCM.")


class AdroitHandDoorPCH(PCH):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 100,
    }
    
    def __init__(
        self,
        max_episode_steps=400,
        policy=None,
        **kwargs
    ):
        task = kwargs.pop("task", Task())
        self.env: AdroitHandDoorSCM = AdroitHandDoorSCM(
            max_episode_steps=max_episode_steps,
            policy=policy,
            **kwargs
        )

        if not isinstance(self.env.observation_space, gym.spaces.Box):
            raise ValueError("AdroitHandDoorPCH only supports Box observation spaces.")
        
        super().__init__(task=task)
        
    # Observational step under behaviour policy
    def see(self, see_policy=None):
        if see_policy is not None:
            a = see_policy(self.env.observation())
        else:
            a = self.env.action()
        o, r, term, trunc, info = self.env.step(a)
        info['natural_action'] = a
        return o, r, term, trunc, info

    # Interventional step with forced action
    def do(self, do_policy):
        action = do_policy(self.env.observation())
        o, r, term, trunc, info = self.env.step(action)
        info['action'] = action
        return o, r, term, trunc, info
    
    # Counterfactual policy intervention
    def ctf_do(self, ctf_policy):
        intuition = self.env.action()
        action = ctf_policy(self.env.observation(), intuition)
        obs, r, terminated, truncated, info = self.env.step(action)
        info['natural_action'] = intuition
        info['action'] = action
        return obs, r, terminated, truncated, info