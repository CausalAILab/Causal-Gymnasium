import gymnasium as gym
import numpy as np

ANT_GEOM_ID_NAME_MAPPING = {
    3: 'left_leg_geom',
    4: 'left_ankle_geom',
    6: 'right_leg_geom',
    7: 'right_ankle_geom',
    9: 'back_leg_geom',
    10: 'third_ankle_geom',
    12: 'rightback_leg_geom',
    13: 'fourth_ankle_geom',
}

class RandomFrictionAnt(gym.Wrapper):
    def __init__(self, env, target_geom_ids, observe_friction=False):
        super().__init__(env)
        self.target_geom_ids = target_geom_ids
        self.observe_friction = observe_friction
        self.frictions = np.concatenate([self.unwrapped.model.geom_friction[geom_id] for geom_id in target_geom_ids])
        
        if self.observe_friction:
            friction_dim = len(self.frictions)
            old_obs_space = self.observation_space
            
            self.observation_space = gym.spaces.Box(
                low=np.concatenate([old_obs_space.low, np.full(friction_dim, -np.inf)]),
                high=np.concatenate([old_obs_space.high, np.full(friction_dim, np.inf)]),
                dtype=old_obs_space.dtype
            )
        

    def reset(self, **kwargs):
        model = self.unwrapped.model
        for geom_id in self.target_geom_ids:
            sliding = np.random.uniform(0.2, 2.0)
            torsional = np.random.uniform(0.1, 0.9)
            rolling = np.random.uniform(0.1, 0.9)
            model.geom_friction[geom_id] = np.array([sliding, torsional, rolling])
            # print(f"[reset] Friction for {ANT_GEOM_ID_NAME_MAPPING[geom_id]}: {model.geom_friction[geom_id]}")
        self.frictions = np.concatenate([self.unwrapped.model.geom_friction[geom_id] for geom_id in self.target_geom_ids])
        obs, info = self.env.reset(**kwargs)
        if self.observe_friction:
            obs = np.concatenate((obs, self.frictions))
        return obs, info
    
    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        if self.observe_friction:
            obs = np.concatenate((obs, self.frictions))
        return obs, reward, done, truncated, info
