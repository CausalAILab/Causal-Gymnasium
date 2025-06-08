import numpy as np
import gymnasium as gym

from ..core import SCM, PCH, Task, Graph

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
    
    # Causal graph -------------------------------------------------------
    @property
    def get_graph(self):
        nodes = [
            # {'name': 'U', 'label': 'Friction', 'type': 'latent'},
            {'name': 'S', 'label': 'State'},
            {'name': 'X', 'label': 'Action'},
            {'name': 'Y', 'label': 'Reward'},
            {'name': "S'", 'label': 'Next State'}
        ]

        edges = [
            # {'from_': 'U', 'to_': 'X', 'type_': 'directed'},
            # {'from_': 'U', 'to_': "S'", 'type_': 'directed'},
            {'from_': 'S', 'to_': 'X', 'type_': 'directed'},
            {'from_': 'S', 'to_': 'Y', 'type_': 'directed'},
            {'from_': 'X', 'to_': 'Y', 'type_': 'directed'},
            {'from_': 'S', 'to_': "S'", 'type_': 'directed'},
            {'from_': 'X', 'to_': "S'", 'type_': 'directed'},
            # Bidirected confounding between Action and Next State
            {'from_': 'X', 'to_': "S'", 'type_': 'bidirected'}
        ]
        graph = Graph(nodes=nodes, edges=edges)
        return graph


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
        task = kwargs.pop("task", Task())
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
        
        super().__init__(task=task)
        
    # Observational step under behaviour policy
    def see(self, see_policy=None):
        if see_policy is not None:
            a = see_policy(self.env.observation)
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