import ale_py
import gymnasium as gym
import numpy as np
from typing import Callable

from ..core import PolicyType, ActType, ObsType, SCM, PCH, Task, Graph

def obs_mask(env_name: str) -> Callable:
    if env_name == "Pong":
        def helper(obs: np.ndarray) -> np.ndarray:
            assert obs.shape == (210, 160, 3), \
                f"Expected observation shape (210, 160, 3), got {obs.shape}"
            obs[:, :20, :] = 0
            obs[:25, :, :] = 0
            return obs
        return helper
    else:
        raise NotImplementedError(f"Obs mask for '{env_name}' is not implemented yet.")


class MaskedAtariSCM(SCM):
    def __init__(
        self, 
        env_name: str, 
        policy: PolicyType | None = None,
        max_episode_steps: int | None = None
    ):
        self._env = gym.make(
            f"{env_name}NoFrameskip-v4", 
            max_episode_steps=max_episode_steps,
        )
        self.spec = self._env.spec
        if policy is not None:
            self.policy = policy
        else:
            # by default, use a random policy
            self.policy = lambda full_obs: self.np_random.integers(0, self._env.action_space.n)
        self.mask = obs_mask(env_name)

        # Ensure the observation space is compatible with the mask
        if not isinstance(self._env.observation_space, gym.spaces.Box):
            raise ValueError(f"Masked {env_name} only supports Box observation spaces.")
        
        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space

    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple[ObsType, dict]:
        self.current_obs, info = self._env.reset(seed=seed, options=options)
        self.current_full_obs = self.current_obs.copy()
        self.current_obs = self.mask(self.current_obs)
        return self.current_obs, info
    
    def step(self, action: ActType) -> tuple[ObsType, float, bool, bool, dict]: 
        next_obs, reward, term, trunc, info = self._env.step(action)
        next_obs = self.mask(next_obs)
        self.current_obs = next_obs
        self.current_full_obs = next_obs.copy()
        return next_obs, reward, term, trunc, info
    
    def observation(self) -> ObsType:
        return self.current_obs
    
    def render(self):
        return self._env.render()
    
    def action(self) -> ActType:
        action = self.policy(self.current_full_obs)
        return action
    
    # Causal graph -------------------------------------------------------
    @property
    def get_graph(self):
        nodes = [
            # {'name': 'U', 'label': 'Confounder', 'type': 'latent'},
            {'name': 'O_m', 'label': 'Masked Portion'},
            {'name': 'O', 'label': 'Unmasked Portion'},
            {'name': 'X', 'label': 'Action'},
            {'name': 'Y', 'label': 'Reward'},
            {'name': 'S', 'label': 'State'},
            {'name': "S'", 'label': 'Next State'}
        ]

        edges = [
            # {'from_': 'U', 'to_': 'X', 'type_': 'directed'},
            # {'from_': 'U', 'to_': "S'", 'type_': 'directed'},
            {'from_': 'S', 'to_': 'O_m', 'type_': 'directed'},
            {'from_': 'S', 'to_': 'O', 'type_': 'directed'},
            {'from_': 'O_m', 'to_': 'X', 'type_': 'directed'},
            {'from_': 'O', 'to_': 'X', 'type_': 'directed'},
            {'from_': 'S', 'to_': 'Y', 'type_': 'directed'},
            {'from_': 'X', 'to_': 'Y', 'type_': 'directed'},
            {'from_': 'S', 'to_': "S'", 'type_': 'directed'},
            {'from_': 'X', 'to_': "S'", 'type_': 'directed'},
            # Bidirected confounding between Action and Next State
            {'from_': 'X', 'to_': "S'", 'type_': 'bidirected'},
            {'from_': 'Y', 'to_': "S'", 'type_': 'bidirected'},
            {'from_': 'X', 'to_': "Y", 'type_': 'bidirected'}
        ]
        graph = Graph(nodes=nodes, edges=edges)
        return graph
    

class MaskedAtariPCH(PCH):
    metadata = {"render_modes": ["rgb_array"]}
    def __init__(self, env_name: str, policy: PolicyType = None, task: Task = Task()):
        self.env: MaskedAtariSCM = MaskedAtariSCM(env_name, policy)
        
        # Ensure the observation space is compatible with the mask
        if not isinstance(self.env.observation_space, gym.spaces.Box):
            raise ValueError("MaskedAtariPCH only supports Box observation spaces.")
        
        self.observation_space = self.env.observation_space
        super().__init__(task=task)

    def see(self):
        action = self.env.action()
        next_obs, reward, term, trunc, info = self.env.step(action)
        return action, next_obs, reward, term, trunc, info
    
    def do(self, action: ActType) -> ObsType:
        next_obs, reward, term, trunc, info = self.env.step(action)
        return next_obs, reward, term, trunc, info

    # Counterfactual policy intervention
    def ctf_do(self, ctf_policy):
        intuition = self.env.action()
        action = ctf_policy(self.env.observation(), intuition)
        obs, r, terminated, truncated, info = self.env.step(action)
        info['natural_action'] = intuition
        return action, obs, r, terminated, truncated, info