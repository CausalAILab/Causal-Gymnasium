import ogbench
import mujoco
import numpy as np
from numpy.typing import NDArray

from typing import Dict, Optional, List, Tuple, Any, Set
from causal_gym import SCM, PCH
from causal_gym.core import ActType, Graph
from gymnasium import spaces

class PointMazeSCM(SCM):
    def __init__(self, env_id: str = 'pointmaze-medium-navigate-singletask-task4-v0', num_steps: int = 1000, expert_mode: bool = False, custom_hidden=None, success_radius: float = 2.0, seed: Optional[int] = None):
        super().__init__()

        self.rng = np.random.default_rng(seed)
        self.num_steps = num_steps
        self.expert_mode = expert_mode
        self.hidden_dims = set() if expert_mode else {'H'}
        if custom_hidden is not None:
            self.hidden_dims = custom_hidden
        self._t = 0

        self._env = ogbench.make_env_and_datasets(env_id, env_only=True, max_episode_steps=num_steps)
        _, info = self._env.reset(seed=seed)
        self._goal_xy = info['goal'][:2]
        self.success_radius = success_radius

        self.H = []  # horizontal position (x)
        self.V = []  # vertical position (y)
        self.X = []  # action, 2-dimensional vector of linear force x,y
        self._Y = [] # sparse reward

        self._U = [] # fog intensity (unobserved confounder)
        self.D = []  # H surrogate, noisy

        self.action_space = self._env.action_space # Box(-1.0, 1.0, (2,), float32)

        # build appropriate observation space
        act_low  = np.asarray(self.action_space.low, dtype=self.action_space.dtype)
        act_high = np.asarray(self.action_space.high, dtype=self.action_space.dtype)

        full_obs = {
            'H': spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float64),
            'V': spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float64),
            'D': spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float64),
            'X': spaces.Box(low=act_low, high=act_high, shape=self.action_space.shape, dtype=np.float64)
        }

        self.observation_space = spaces.Dict({k: v for k, v in full_obs.items() if k not in self.hidden_dims})

    def _H_val(self) -> NDArray[np.float64]:
        return np.array([self._env.unwrapped.data.qpos[0]], dtype=np.float64)

    def _V_val(self) -> NDArray[np.float64]:
        return np.array([self._env.unwrapped.data.qpos[1]], dtype=np.float64)

    def _position(self) -> NDArray[np.float64]:
        return np.array([self.H[-1][0], self.V[-1][0]], dtype=np.float64)

    def _U_val(self) -> np.float64:
        """Fog intensity: 0 = clear, (0, 0.5] = foggy."""
        if self.rng.random() < 0.3:
            return np.float64(self.rng.uniform(0.1, 0.5))
        return np.float64(0.0)

    def _D_val(self) -> NDArray[np.float64]:
        h_val = float(self.H[-1])
        u = float(self._U[-1])

        if self.expert_mode:
            noise_std = 0.1 + u  # fog increases sensor noise
            noise = self.rng.normal(0.0, noise_std)
            return np.array([h_val + noise], dtype=np.float64)
        else:
            # Only reveal direction sign of H
            if h_val < 0.0:
                return np.array([1.0], dtype=np.float64)
            else:
                return np.array([-1.0], dtype=np.float64)

    def observation(self, history: bool = False) -> Dict[str, Any]:
        obs = {}

        if history:
            if 'H' not in self.hidden_dims:
                obs['H'] = self.H
            if 'V' not in self.hidden_dims:
                obs['V'] = self.V
            if 'D' not in self.hidden_dims:
                obs['D'] = self.D
            if 'X' not in self.hidden_dims:
                obs['X'] = self.X
        else:
            if 'H' not in self.hidden_dims:
                obs['H'] = self.H[-1]
            if 'V' not in self.hidden_dims:
                obs['V'] = self.V[-1]
            if 'D' not in self.hidden_dims:
                obs['D'] = self.D[-1]
            if 'X' not in self.hidden_dims:
                obs['X'] = self.X[-1] if len(self.X) > 0 else np.zeros(self.action_space.shape, dtype=np.float64)

        return obs

    def reset(self, history: bool = False, seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed)
        env_obs, env_info = self._env.reset()

        self._t = 0
        self.H = [self._H_val()]
        self.V = [self._V_val()]
        self._U = [self._U_val()]
        self.D = [self._D_val()]
        self.X = []
        self._Y = []

        obs = self.observation(history=history)
        hiddens = {}
        if 'H' in self.hidden_dims:
            hiddens['H'] = self.H
        if 'V' in self.hidden_dims:
            hiddens['V'] = self.V
        if 'D' in self.hidden_dims:
            hiddens['D'] = self.D
        if 'X' in self.hidden_dims:
            hiddens['X'] = self.X

        info = {'Y': self._Y, 'U': self._U, 'env_obs': env_obs, 'env_info': env_info, 'hidden_obs': hiddens}
        return obs, info

    def action(self, H: List[NDArray[np.float64]], V: List[NDArray[np.float64]], D: List[NDArray[np.float64]]) -> ActType:
        # placeholder behavior policy
        return self.action_space.sample()

    def compute_success(self) -> bool:
        pos = self._position()
        dg = np.asarray(self._goal_xy, dtype=np.float64)
        dist = np.linalg.norm(pos - dg)
        return dist <= self.success_radius

    def _reward(self) -> float:
        u = float(self._U[-1])
        return float(self.compute_success()) - 0.1 * u

    def step(self, action: Any, history: bool = False, show_reward: bool = True) -> Tuple[dict, float, bool, bool, dict]:
        # actions are float32, but observed actions need to be float64
        self.X.append(np.asarray(action, dtype=np.float64))

        # sample fog for this step
        u = self._U_val()
        self._U.append(u)

        # fog applies a small random force (U -> H, V)
        self._env.unwrapped.data.xfrc_applied[:] = 0.0
        if u > 0:
            model = self._env.unwrapped.model
            data = self._env.unwrapped.data
            torso_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'torso')
            fx = float(u) * self.rng.normal(0.0, 0.2)
            fy = float(u) * self.rng.normal(0.0, 0.2)
            data.xfrc_applied[torso_id] = [fx, fy, 0, 0, 0, 0]

        # step environment
        env_obs, reward, _, truncated, env_info = self._env.step(action)

        # update SCM state
        self._t += 1
        self.H.append(self._H_val())
        self.V.append(self._V_val())
        self.D.append(self._D_val())

        terminated = self.compute_success()
        reward = self._reward()
        self._Y.append(reward)

        obs = self.observation(history=history)

        hiddens = {}
        if 'H' in self.hidden_dims:
            hiddens['H'] = self.H
        if 'V' in self.hidden_dims:
            hiddens['V'] = self.V
        if 'D' in self.hidden_dims:
            hiddens['D'] = self.D
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
        state_vars = ['H', 'V', 'D']
        variables = state_vars + ['X']
        T = self.num_steps
        n = T * len(variables) + len(state_vars) + 1

        nodes = {}
        i = 0
        for t in range(T):
            for v in variables:
                nodes[i] = f'{v}{t}'
                i += 1

        # terminal state
        for v in state_vars:
            nodes[i] = f'{v}{T}'
            i += 1

        nodes[i] = f'Y{T}' # ensures Y comes last in temporal ordering

        base_graph = [[0]*n for _ in range(n)]
        conf_graph = [[0]*n for _ in range(n)]

        y = n - 1

        # intra-timestep edges
        for step in range(T):
            base = step * len(variables)
            h, v, d, x = base, base + 1, base + 2, base + 3

            # d is surrogate of h
            base_graph[h][d] = 1

            # state influences decision-making
            base_graph[h][x] = 1
            base_graph[v][x] = 1

            # position determines reward
            base_graph[h][y] = 1
            base_graph[v][y] = 1

            # fog confounding: U -> {H, V, D, Y}
            conf_graph[h][v] = 1
            conf_graph[h][d] = 1
            conf_graph[h][y] = 1
            conf_graph[v][d] = 1
            conf_graph[v][y] = 1
            conf_graph[d][y] = 1

        # intra-timestep edges for terminal state
        base_term = T * len(variables)
        h, v, d = base_term, base_term + 1, base_term + 2

        base_graph[h][d] = 1

        base_graph[h][y] = 1
        base_graph[v][y] = 1

        conf_graph[h][v] = 1
        conf_graph[h][d] = 1
        conf_graph[h][y] = 1
        conf_graph[v][d] = 1
        conf_graph[v][y] = 1
        conf_graph[d][y] = 1

        # inter-timestep edges
        for step in range(T):
            base = step * len(variables)
            base_next = (step + 1) * len(variables)

            h, v, d, x = base, base + 1, base + 2, base + 3
            h2, v2, d2 = base_next, base_next + 1, base_next + 2

            base_graph[x][h2] = 1 # action affects next horizontal position
            base_graph[x][v2] = 1 # action affects next vertical position

            # state persistence
            base_graph[h][h2] = 1
            base_graph[v][v2] = 1

            # action persistence
            if step < T - 1:
                x2 = base_next + 3
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
        all_vars = ['H', 'V', 'D', 'X']
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

        self.hidden_dims = set() if expert_mode else {'H'}

        self.H = []
        self.V = []
        self.D = []
        self.X = []
        self._Y = []

    def observation(self) -> Dict[str, Any]:
        obs = {}

        if 'H' not in self.hidden_dims:
            obs['H'] = self.H
        if 'V' not in self.hidden_dims:
            obs['V'] = self.V
        if 'D' not in self.hidden_dims:
            obs['D'] = self.D
        if 'X' not in self.hidden_dims:
            obs['X'] = self.X

        return obs

    def reset(self):
        self._t += 1
        if self._t >= len(self._expert_trajs['observations']):
            self._t = 0
            print('Warning: Expert trajectories exhausted. If this is a reset to begin do(), ignore this message.')

        self.H = []
        self.V = []
        self.D = []
        self.X = []
        self._Y = []

    def _reward(self) -> float:
        pos = np.array([self.H[-1][0], self.V[-1][0]], dtype=np.float64)
        dg = np.asarray(self._goal_xy, dtype=np.float64)

        diff = pos - dg
        dist = np.linalg.norm(diff)

        success = (dist <= self.success_radius).astype(np.float64)
        return -1.0 + success

    def step(self):
        obs = self._expert_trajs['observations'][self._t]
        action = self._expert_trajs['actions'][self._t].astype(np.float64)

        terminated = self._expert_trajs['terminals'][self._t]
        truncated = False # never happens in an OGBench dataset

        h = np.array([obs[0]], dtype=np.float64)
        v = np.array([obs[1]], dtype=np.float64)

        self.H.append(h)
        self.V.append(v)
        self.X.append(action)

        # noisy surrogate of H
        noise = self.rng.normal(0.0, 0.1)
        d = np.array([h[0] + noise], dtype=np.float64)
        self.D.append(d)

        reward = self._reward()
        self._Y.append(reward)

        hiddens = {}
        if 'H' in self.hidden_dims:
            hiddens['H'] = self.H
        if 'V' in self.hidden_dims:
            hiddens['V'] = self.V
        if 'D' in self.hidden_dims:
            hiddens['D'] = self.D
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
        H = self.env.H
        V = self.env.V
        D = self.env.D

        if behavioral_policy is not None:
            action = behavioral_policy(H, V, D)
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
        H = self.env.H
        V = self.env.V
        D = self.env.D

        intuition = self.env.action(H, V, D)
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

    @property
    def observed_unobserved_vars(self) -> Tuple[list[str], list[str]]:
        return self.env.observed_unobserved_vars
