import numpy as np
import gymnasium as gym

from ..core import SCM, PCH

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

FRICTION_COMBINATIONS = [
    [0.2, 0.1, 0.1],  # Very slippery
    [2.0, 0.9, 0.9],  # High friction
    [0.2, 0.9, 0.1],  # Slippery but high torsional
    [2.0, 0.1, 0.9],  # High sliding but low torsional
    [1.0, 0.1, 0.9],  # Medium sliding, low torsional, high rolling
    [1.0, 0.9, 0.1],  # Medium sliding, high torsional, low rolling
]

class RandomFrictionAntMujocoSCM(SCM):
    def __init__(
        self, 
        target_geom_ids, 
        env_id="Ant-v5", 
        observe_friction=False, 
        max_episode_steps=1000, 
        policy=None,
        **kwargs
    ):
        kwargs.setdefault("max_episode_steps", max_episode_steps)
        kwargs.setdefault("render_mode", "rgb_array")
        self._env = gym.make(
            env_id,
            **kwargs,
        )
        self.target_geom_ids = target_geom_ids
        self.observe_friction = observe_friction
        self.frictions = np.concatenate([self._env.unwrapped.model.geom_friction[geom_id] for geom_id in target_geom_ids])
        
        if self.observe_friction:
            friction_dim = len(self.frictions)
            old_obs_space = self._env.observation_space
            
            self.observation_space = gym.spaces.Box(
                low=np.concatenate([old_obs_space.low, np.full(friction_dim, -np.inf)]),
                high=np.concatenate([old_obs_space.high, np.full(friction_dim, np.inf)]),
                dtype=old_obs_space.dtype
            )
        else:
            self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space
        self.spec = self._env.spec

        if policy is not None:
            self.policy = policy 
        else:
            self.policy = lambda obs: self._env.action_space.sample()
        
    def reset(self, **kwargs):
        model = self._env.unwrapped.model
        friction_values = FRICTION_COMBINATIONS[np.random.randint(0, len(FRICTION_COMBINATIONS))]
        
        for geom_id in self.target_geom_ids:
            model.geom_friction[geom_id] = np.array(friction_values)
            # print(f"[reset] Friction for {ANT_GEOM_ID_NAME_MAPPING[geom_id]}: {model.geom_friction[geom_id]}")
        self.frictions = np.concatenate([self._env.unwrapped.model.geom_friction[geom_id] for geom_id in self.target_geom_ids])
        obs, info = self._env.reset(**kwargs)
        if self.observe_friction:
            obs = np.concatenate((obs, self.frictions))
        self.current_obs = obs.copy()
        return obs, info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self._env.step(action)
        if self.observe_friction:
            obs = np.concatenate((obs, self.frictions))
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
        nodes = {0: "Friction(U)", 1: "State(S)", 2: "Action(X)", 3: "Reward(Y)", 4: "Next_State(S')"}
        base = [[0] * 5 for _ in range(5)]
        base[0][2] = 1  # U → X
        base[0][4] = 1  # U → S'
        base[1][2] = 1  # S → X
        base[1][3] = 1  # S → Y
        base[2][3] = 1  # X → Y
        base[1][4] = 1  # S → S'
        base[2][4] = 1  # X → S'
        conf = [[0] * 5 for _ in range(5)]
        conf[2][4] = 1
        conf[4][2] = 1
        return nodes, base, conf


class RandomFrictionAntMujocoPCH(PCH):
    metadata = {"render_modes": ["rgb_array"]}
    
    def __init__(
        self, 
        target_geom_ids, 
        env_id="Ant-v5", 
        observe_friction=False, 
        max_episode_steps=1000, 
        policy=None, 
        **kwargs
    ):
        self.env: RandomFrictionAntMujocoSCM = RandomFrictionAntMujocoSCM(
            target_geom_ids,
            env_id=env_id,
            observe_friction=observe_friction,
            max_episode_steps=max_episode_steps,
            policy=policy,
            **kwargs
        )
        
        if not isinstance(self.env.observation_space, gym.spaces.Box):
            raise ValueError("RandomFrictionMujocoPCH only supports Box observation spaces.")
        
        super().__init__()
        
    def see(self):
        action = self.env.action()
        next_obs, reward, term, trunc, info = self.env.step(action)
        return action, next_obs, reward, term, trunc, info
    
    def do(self, action):
        next_obs, reward, term, trunc, info = self.env.step(action)
        return next_obs, reward, term, trunc, info