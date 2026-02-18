import ogbench
import mujoco
import numpy as np
from numpy.typing import NDArray

from typing import Dict, Optional, List, Tuple, Any, Set
from causal_gym import SCM, PCH
from causal_gym.core import ActType, Graph
from gymnasium import spaces

class PointMazeSCM(SCM):
    def __init__(self, env_id: str = 'pointmaze-medium-navigate-singletask-task1-v0', num_steps: int = 1000, expert_mode: bool = False, custom_hidden=None, success_radius: float = 5.0, seed: Optional[int] = None):
        super().__init__()

        self.rng = np.random.default_rng(seed)
        self.num_steps = num_steps
        self.expert_mode = expert_mode
        self.hidden_dims = set() if expert_mode else {'L'}
        if custom_hidden is not None:
            self.hidden_dims = custom_hidden
        self._t = 0

        self._env = ogbench.make_env_and_datasets(env_id, env_only=True, max_episode_steps=num_steps)
        _, info = self._env.reset(seed=seed)
        self._goal_xy = info['goal'][:2]
        self.success_radius = success_radius

        self.P = [] # position, 2-dimensional vector of x,y
        self.L = [] # ball linear velocity, 2-dimensional vector of x,y
        self.X = [] # action, 2-dimensional vector of linear force x,y
        self._Y = [] # sparse reward

        # wind confounding
        self._U = [] # wind field
        self.W = [] # noisy velocity sensor

        self.action_space = self._env.action_space # Box(-1.0, 1.0, (2,), float32)

        # build appropriate observation space
        act_low  = np.asarray(self.action_space.low, dtype=self.action_space.dtype)
        act_high = np.asarray(self.action_space.high, dtype=self.action_space.dtype)

        full_obs = {
            'P': spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float64),
            'L': spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float64),
            'W': spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float64),
            'X': spaces.Box(low=act_low, high=act_high, shape=self.action_space.shape, dtype=np.float64)
        }

        self.observation_space = spaces.Dict({k: v for k, v in full_obs.items() if k not in self.hidden_dims})
        '''
        Observation space details:
        Box(-inf, inf, (4,), float64)
        0-1 = position x,y
        2-3 = ball linear velocity x, y (borrowed from qvel)
        '''

    def _P(self) -> NDArray[np.float64]:
        return self._env.env.env.env.get_ob()

    def _L(self) -> NDArray[np.float64]:
        return self._env.unwrapped.data.qvel

    def _U_val(self) -> NDArray[np.float64]:
        p_gust = 0.05
        min_strength = 0.05
        max_strength = 0.15

        # start of episode, random gust or no gust
        if self._t == 0:
            if self.rng.random() < p_gust:
                angle = self.rng.uniform(0.0, 2.0 * np.pi)
                mag = self.rng.uniform(min_strength, max_strength)
                u = np.array([mag * np.cos(angle), mag * np.sin(angle)], dtype=np.float64)
            else:
                u = np.zeros(2, dtype=np.float64)

        else:
            if self._t % 5 != 0:
                return self._U[-1]
            if self.rng.random() < p_gust:
                angle = self.rng.uniform(0.0, 2.0 * np.pi)
                mag = self.rng.uniform(min_strength, max_strength)
                u = np.array([mag * np.cos(angle), mag * np.sin(angle)], dtype=np.float64)
            else:
                u = np.zeros(2, dtype=np.float64)

        return u

    def _W(self) -> NDArray[np.float64]:
        u = self._U[-1]
        l = self.L[-1]

        if self.expert_mode:
            w_l, w_u = 0.9, 0.1
            noise_std = 0.01
        else:
            w_l, w_u = 0.3, 0.7
            noise_std = 0.1

        noise = self.rng.normal(0.0, noise_std, size=2)
        return (w_l * l + w_u * u + noise).astype(np.float64)

    def observation(self, history: bool = False) -> Dict[str, Any]:
        obs = {}

        if history:
            if 'P' not in self.hidden_dims:
                obs['P'] = self.P
            if 'L' not in self.hidden_dims:
                obs['L'] = self.L
            if 'W' not in self.hidden_dims:
                obs['W'] = self.W
            if 'X' not in self.hidden_dims:
                obs['X'] = self.X
        else:
            if 'P' not in self.hidden_dims:
                obs['P'] = self.P[-1]
            if 'L' not in self.hidden_dims:
                obs['L'] = self.L[-1]
            if 'W' not in self.hidden_dims:
                obs['W'] = self.W[-1]
            if 'X' not in self.hidden_dims:
                obs['X'] = self.X[-1] if len(self.X) > 0 else np.zeros(self.action_space.shape, dtype=np.float64)

        return obs

    def reset(self, history: bool = False, seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed)
        env_obs, env_info = self._env.reset()

        self._t = 0
        self.P = [self._P()]
        self.L = [self._L()]
        self._U = [self._U_val()]
        self.W = [self._W()]
        self.X = []
        self._Y = []

        obs = self.observation(history=history)
        hiddens = {}
        if 'P' in self.hidden_dims:
            hiddens['P'] = self.P
        if 'L' in self.hidden_dims:
            hiddens['L'] = self.L
        if 'W' in self.hidden_dims:
            hiddens['W'] = self.W
        if 'X' in self.hidden_dims:
            hiddens['X'] = self.X

        info = {'Y': self._Y, 'U': self._U, 'env_obs': env_obs, 'env_info': env_info, 'hidden_obs': hiddens}
        return obs, info

    def action(self, P: List[NDArray[np.float64]], L: List[NDArray[np.float64]], W: List[NDArray[np.float64]]) -> ActType:
        # placeholder behavior policy
        return self.action_space.sample()
    
    def compute_success(self) -> bool:
        ag = np.asarray(self.P[-1], dtype=np.float64)
        dg = np.asarray(self._goal_xy, dtype=np.float64)
        diff = ag - dg
        dist = np.linalg.norm(diff)
        return dist <= self.success_radius

    def _reward(self) -> float:
        ag = np.asarray(self.P[-1], dtype=np.float64)
        dg = np.asarray(self._goal_xy, dtype=np.float64)
        diff = ag - dg
        dist = np.linalg.norm(diff)

        # wind penalty
        u_norm = float(np.linalg.norm(self._U[-1]))
        dist_norm = dist / 25.0 # approximate maze size
        lambda_u = 1.0
        wind_penalty = lambda_u * u_norm * (1.0 + dist_norm)

        return float(self.compute_success()) - wind_penalty

    def step(self, action: Any, history: bool = False, show_reward: bool = True) -> Tuple[dict, float, bool, bool, dict]:
        # actions are float32, but observed actions need to be float64
        self.X.append(np.asarray(action, dtype=np.float64))

        # apply wind and gust
        u = self._U_val()
        self._U.append(u)

        model = self._env.env.env.env.model
        data = self._env.env.env.env.data

        data.xfrc_applied[:] = 0.0 # reset last step's forces
        torso_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'torso')
        total_force = np.array([
            u[0],
            u[1],
            0.0, # no z
            0.0, 0.0, 0.0 # no torque
        ], dtype=np.float64)
        data.xfrc_applied[torso_id] = total_force

        # step environment
        env_obs, reward, _, truncated, env_info = self._env.step(action)
        terminated = self.compute_success()

        # update rest of SCM state
        self._t += 1
        self.P.append(self._P())
        self.L.append(self._L())
        self.W.append(self._W())

        reward = self._reward()
        self._Y.append(reward)

        obs = self.observation(history=history)

        hiddens = {}
        if 'P' in self.hidden_dims:
            hiddens['P'] = self.P
        if 'L' in self.hidden_dims:
            hiddens['L'] = self.L
        if 'W' in self.hidden_dims:
            hiddens['W'] = self.W
        if 'X' in self.hidden_dims:
            hiddens['X'] = self.X

        info = {'Y': self._Y, 'U': self._U, 'env_obs': env_obs, 'env_info': env_info, 'hidden_obs': hiddens}
        return obs, reward if show_reward else None, terminated, truncated, info

    def render(self):
        return self._env.render()

    def close(self):
        self._env.close()

    @property
    def get_graph(self) -> Graph:
        state_vars = ['P', 'L', 'W']
        variables = state_vars + ['X']
        H = self.num_steps
        n = H * len(variables) + len(state_vars) + 1

        nodes = {}
        i = 0
        for t in range(H):
            for v in variables:
                nodes[i] = f'{v}{t}'
                i += 1

        # terminal state
        for v in state_vars:
            nodes[i] = f'{v}{H}'
            i += 1

        nodes[i] = f'Y{H}' # ensures Y comes last in temporal ordering

        base_graph = [[0]*n for _ in range(n)]
        conf_graph = [[0]*n for _ in range(n)]

        y = n - 1

        # intra-timestep edges
        for step in range(H):
            base = step * len(variables)
            p, l, w, x = base, base + 1, base + 2, base + 3

            base_graph[l][p] = 1 # linear velocity affects position

            # state influence decision-making
            base_graph[l][x] = 1
            base_graph[p][x] = 1

            base_graph[p][y] = 1 # reward is based on position

            # wind confounding
            base_graph[l][w] = 1

            conf_graph[w][y] = 1
            conf_graph[p][y] = 1
            conf_graph[w][p] = 1

        # intra-timestep edges for terminal state
        base_term = H * len(variables)
        p, l, w, x = base_term, base_term + 1, base_term + 2, base_term + 3

        base_graph[l][p] = 1 # linear velocity affects position

        base_graph[p][y] = 1 # reward is based on position

        # wind confounding for terminal state
        base_graph[l][w] = 1

        conf_graph[w][y] = 1
        conf_graph[p][y] = 1
        conf_graph[w][p] = 1

        # inter-timstep edges
        for step in range(H):
            base = step * len(variables)
            base_next = (step + 1) * len(variables)

            p, l, w, x = base, base + 1, base + 2, base + 3
            p2, l2, w2, x2 = base_next, base_next + 1, base_next + 2, base_next + 3

            base_graph[x][l2] = 1 # linear force impacts linear velocity
            base_graph[x][p2] = 1 # action impacts position

            # state persistence
            base_graph[p][p2] = 1
            base_graph[l][l2] = 1
            base_graph[x][x2] = 1

        nodes = [{'name': n} for n in nodes.values()]
        edges = []
        for i in range(len(nodes)):
            for j in range(len(nodes)):
                if base_graph[i][j] == 1:
                    edges.append({'from_': nodes[i]['name'], 'to_': nodes[j]['name'], 'type_': 'directed'})
                if conf_graph[i][j] == 1:
                    edges.append({'from_': nodes[i]['name'], 'to_': nodes[j]['name'], 'type_': 'bidirected'})
        graph = Graph(nodes=nodes, edges=edges)
        return graph

    @property
    def observed_unobserved_vars(self) -> Tuple[list[str], list[str]]:
        all_vars = ['P', 'L', 'W', 'X']
        observed = [v for v in all_vars if v not in self.hidden_dims]
        unobserved = list(self.hidden_dims) + ['U', 'Y']
        return observed, unobserved

class PointMazeExpert:
    def __init__(self, env_id: str = 'pointmaze-medium-navigate-singletask-task1-v0', num_steps: int = 1000, expert_mode: bool = False, success_radius: float = 5.0, goal_xy: np.ndarray = np.array([20.0, 20.0]), seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed)
        self.num_steps = num_steps

        _, train, test = ogbench.make_env_and_datasets(env_id, max_episode_steps=num_steps, compact_dataset=True)
        self._expert_trajs = {k: np.concatenate([train[k], test[k]], axis=0) for k in train.keys()}

        self.num_eps = len(self._expert_trajs['observations']) // 1000 - 1
        self._t = -1 # reset brings it to 0

        self.success_radius = success_radius
        self._goal_xy = goal_xy

        self.hidden_dims = set() if expert_mode else {'L'}

        self.P = []
        self.L = []
        self.X = []
        self._Y = []

    def observation(self) -> Dict[str, Any]:
        obs = {}

        if 'P' not in self.hidden_dims:
            obs['P'] = self.P
        if 'L' not in self.hidden_dims:
            obs['L'] = self.L
        if 'X' not in self.hidden_dims:
            obs['X'] = self.X

        return obs

    def reset(self):
        self._t += 1
        if self._t >= len(self._expert_trajs['observations']):
            self._t = 0
            print('Warning: Expert trajectories exhausted. If this is a reset to begin do(), ignore this message.')

        self.P = []
        self.L = []
        self.X = []
        self._Y = []

    def _reward(self) -> float:
        ag = np.asarray(self.P[-1], dtype=np.float64)
        dg = np.asarray(self._goal_xy, dtype=np.float64)

        diff = ag - dg
        dist = np.linalg.norm(diff)

        success = (dist <= self.success_radius).astype(np.float64)
        return -1.0 + success

    def step(self):
        obs = self._expert_trajs['observations'][self._t]
        action = self._expert_trajs['actions'][self._t].astype(np.float64)

        terminated = self._expert_trajs['terminals'][self._t]
        truncated = False # never happens in an OGBench dataset

        self.P.append(obs[0:3])
        self.L.append(obs[15:18])
        self.X.append(action)
        
        reward = self._reward()
        self._Y.append(reward)

        hiddens = {}
        if 'P' in self.hidden_dims:
            hiddens['P'] = self.P
        if 'L' in self.hidden_dims:
            hiddens['L'] = self.L
        if 'X' in self.hidden_dims:
            hiddens['X'] = self.X

        info = {'Y': self._Y, 'env_obs': obs, 'env_info': {}, 'hidden_obs': hiddens, 'natural_action': action}

        self._t += 1
        return self.observation(), reward, terminated, truncated, info

class PointMazePCH(PCH):
    def __init__(self, env_id: str = 'pointmaze-medium-navigate-singletask-task1-v0', num_steps: int = 1000, expert_mode: bool = False, custom_hidden=None, success_radius: float = 5.0, seed: Optional[int] = None):
        # initialize underlying SCM
        self.env = PointMazeSCM(env_id=env_id, num_steps=num_steps, expert_mode=expert_mode, custom_hidden=custom_hidden, success_radius=success_radius, seed=seed)
        self.expert = PointMazeExpert(env_id=env_id, num_steps=num_steps, expert_mode=True, success_radius=success_radius, goal_xy=self.env._goal_xy, seed=seed)
        super().__init__()

        self.last_actor_is_expert = True

    def see(self, behavioral_policy=None, show_reward=True) -> Tuple[Any, Any, float, bool, bool, Dict[str, Any]]:
        P = self.env.P
        L = self.env.L
        W = self.env.W

        if behavioral_policy is not None:
            action = behavioral_policy(P, L, W)
        else:
            return self.expert.step()

        obs, reward, terminated, truncated, info = self.env.step(action, history=True, show_reward=show_reward)
        info['natural_action'] = action

        self.last_actor_is_expert = True
        return obs, reward, terminated, truncated, info

    # Interventional step with forced action
    def do(self, do_policy, show_reward = True):
        action = do_policy(self.env.observation(history=True))
        o, r, term, trunc, info = self.env.step(action, history=True, show_reward=show_reward)
        info['action'] = action

        self.last_actor_is_expert = False
        return o, r, term, trunc, info

    # Counterfactual policy intervention
    def ctf_do(self, ctf_policy):
        P = self.env.P
        L = self.env.L
        W = self.env.W

        intuition = self.env.action(P, L, W)
        action = ctf_policy(self.env.observation(), intuition)
        obs, r, terminated, truncated, info = self.env.step(action)
        info['natural_action'] = intuition
        info['action'] = action

        self.last_actor_is_expert = False
        return obs, r, terminated, truncated, info

    def reset(self, *, seed: int = None) -> Tuple[Any, dict]:
        if self.last_actor_is_expert:
            self.expert.reset()

        return self.env.reset(history=True, seed=seed)

    def render(self) -> Any:
        return self.env.render()

    def close(self) -> None:
        self.env.close()

    @property
    def get_graph(self) -> Graph:
        return self.env.get_graph