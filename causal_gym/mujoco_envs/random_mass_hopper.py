import gymnasium as gym
import numpy as np
from gymnasium.envs.mujoco.hopper_v4 import HopperEnv

MASS_CONFIGURATIONS = [
    {
        'name': 'heavy_upper',
        'thigh': 4.0,
        'leg': 3.0,   
        'foot': 1.0,   
    },
    {
        'name': 'light_upper',
        'thigh': 0.5, 
        'leg': 0.7, 
        'foot': 1.0,
    },
    {
        'name': 'mixed_upper',
        'thigh': 3.0,  
        'leg': 0.7,    
        'foot': 1.0, 
    },
    {
        'name': 'inverse_mixed',
        'thigh': 0.7, 
        'leg': 3.0,   
        'foot': 1.0,   
    }
]

class MassHopper(gym.Wrapper):    
    def __init__(self, env, observe_mass=False):
        super().__init__(env)
        
        self.observe_mass = observe_mass
        self.current_config = None
        
        self.mass_indices = {
            'thigh': 1, 
            'leg': 2,    
            'foot': 3,   
        }
        
        self.original_masses = {
            'thigh': float(self.unwrapped.model.body_mass[self.mass_indices['thigh']]),
            'leg': float(self.unwrapped.model.body_mass[self.mass_indices['leg']]),
            'foot': float(self.unwrapped.model.body_mass[self.mass_indices['foot']])
        }
        
        if self.observe_mass:
            old_obs_space = self.observation_space
            self.observation_space = gym.spaces.Box(
                low=np.concatenate([old_obs_space.low, np.array([0.0, 0.0, 0.0])]),
                high=np.concatenate([old_obs_space.high, np.array([5.0, 5.0, 5.0])]),
                dtype=old_obs_space.dtype
            )

    def _set_random_mass_config(self):
        """Set a random mass configuration from the predefined options"""
        self.current_config = MASS_CONFIGURATIONS[np.random.randint(len(MASS_CONFIGURATIONS))]
        
        for part, mass_mult in self.current_config.items():
            if part != 'name':
                idx = self.mass_indices[part]
                self.unwrapped.model.body_mass[idx] = self.original_masses[part] * mass_mult

    def reset(self, **kwargs):
        self._set_random_mass_config()
        obs, info = self.env.reset(**kwargs)
        
        if self.observe_mass:
            mass_obs = np.array([
                self.current_config['thigh'],
                self.current_config['leg'],
                self.current_config['foot']
            ])
            obs = np.concatenate([obs, mass_obs])
            
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        if self.observe_mass:
            mass_obs = np.array([
                self.current_config['thigh'],
                self.current_config['leg'],
                self.current_config['foot']
            ])
            obs = np.concatenate([obs, mass_obs])
            
        info['mass_config'] = self.current_config['name']
        return obs, reward, terminated, truncated, info 