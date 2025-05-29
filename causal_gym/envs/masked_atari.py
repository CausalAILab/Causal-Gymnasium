import ale_py
import gymnasium as gym
import numpy as np
from typing import Callable

from ..core import PolicyType, ActType, ObsType, SCM, PCH

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
    
    @property
    def get_graph(self):
        nodes = {
            0: "Masked Portion (O_m)", 
            1: "Unmasked Portion (O)", 
            2: "Action(X)", 
            3: "Reward(Y)", 
            4: "State(S)", 
            5: "Next State(S')"
        }
        # The base graph structure for the Atari environment
        base = [[0] * 6 for _ in range(6)]  # Updated to accommodate the new node
        base[4][0] = 1  # S → O_m
        base[4][1] = 1  # S → O
        base[0][2] = 1  # O_m → X, natural regime can see
        base[1][2] = 1  # O → X
        base[4][3] = 1  # S → Y
        base[2][3] = 1  # X → Y
        base[4][5] = 1  # O → S'
        base[2][5] = 1  # X → S'
        conf = [[0] * 6 for _ in range(6)]  # Updated to accommodate the new node
        # X and S'
        conf[2][5] = 1
        conf[5][2] = 1
        # Y and S'
        conf[3][5] = 1
        conf[5][3] = 1
        # X and Y
        conf[2][3] = 1
        conf[3][2] = 1
        return nodes, base, conf
    

class MaskedAtariPCH(PCH):
    metadata = {"render_modes": ["rgb_array"]}
    def __init__(self, env_name: str, policy: PolicyType = None):
        self.env: MaskedAtariSCM = MaskedAtariSCM(env_name, policy)
        
        # Ensure the observation space is compatible with the mask
        if not isinstance(self.env.observation_space, gym.spaces.Box):
            raise ValueError("MaskedAtariPCH only supports Box observation spaces.")
        
        self.observation_space = self.env.observation_space
        super().__init__()

    def see(self):
        action = self.env.action()
        next_obs, reward, term, trunc, info = self.env.step(action)
        return action, next_obs, reward, term, trunc, info
    
    def do(self, action: ActType) -> ObsType:
        next_obs, reward, term, trunc, info = self.env.step(action)
        return next_obs, reward, term, trunc, info