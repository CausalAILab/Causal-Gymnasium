import ogbench
import mujoco
import numpy as np
from numpy.typing import NDArray

from typing import Dict, Optional, List, Tuple, Any, Set
from causal_gym import SCM, PCH
from causal_gym.core import ActType, Graph
from gymnasium import spaces

class AntMazeSCM(SCM):
    def __init__(self, env_id: str = 'antmaze-medium-navigate-singletask-task1-v0', num_steps: int = 1000, hidden_dims: Optional[Set[str]] = None, success_radius: float = 5.0, seed: Optional[int] = None):
        super().__init__()

        self.rng = np.random.default_rng(seed)
        self.num_steps = num_steps
        self.hidden_dims = hidden_dims if hidden_dims is not None else set()
        self._t = 0

        self._env = ogbench.make_env_and_datasets(env_id, env_only=True, max_episode_steps=num_steps)
        _, info = self._env.reset(seed=seed)
        self._goal_xy = info['goal'][:2]
        self.success_radius = success_radius

        self.P = [] # position, 3-dimensional vector of x,y,z
        self.O = [] # torso orientation, 4-dimensional quaternion x,y,z,w
        self.A = [] # joint angles, 8-dimensional vector
        self.L = [] # torso linear velocity, 3-dimensional vector of x,y,z
        self.T = [] # torso angular velocity, 3-dimensional vector of x,y,z
        self.J = [] # joint angular velocities, 8-dimensional vector
        self.X = [] # action, 8-dimensional vector of torques
        self._Y = [] # sparse reward

        # wind confounding
        self._U = [] # macro wind field (latent to all)
        self._F = [] # targeted gust (latent to imitator)
        self.W = [] # noisy wind alarm

        self.action_space = self._env.action_space # Box(-1.0, 1.0, (8,), float32)

        # build appropriate observation space
        act_low  = np.asarray(self.action_space.low, dtype=self.action_space.dtype)
        act_high = np.asarray(self.action_space.high, dtype=self.action_space.dtype)

        full_obs = {
            'P': spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float64),
            'O': spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float64),
            'A': spaces.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float64),
            'L': spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float64),
            'T': spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float64),
            'J': spaces.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float64),
            'F': spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float64),
            'W': spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float64),
            'X': spaces.Box(low=act_low, high=act_high, shape=self.action_space.shape, dtype=np.float64)
        }

        self.observation_space = spaces.Dict({k: v for k, v in full_obs.items() if k not in self.hidden_dims})
        '''
        Observation space details:
        Box(-inf, inf, (29,), float64)
        0-2 = position x,y,z
        3-6 = torso orientation, x, y, z, w
        7-14 = joint angles
        15-17 = torso linear velocity x, y, z
        18-20 = torso angular velocity x, y, z
        21-28 = joint angular velocities
        '''

    def _P(self) -> NDArray[np.float64]:
        return self._env.env.env.env.get_ob()[0:3]

    def _O(self) -> NDArray[np.float64]:
        return self._env.env.env.env.get_ob()[3:7]

    def _A(self) -> NDArray[np.float64]:
        return self._env.env.env.env.get_ob()[7:15]

    def _L(self) -> NDArray[np.float64]:
        return self._env.env.env.env.get_ob()[15:18]

    def _T(self) -> NDArray[np.float64]:
        return self._env.env.env.env.get_ob()[18:21]

    def _J(self) -> NDArray[np.float64]:
        return self._env.env.env.env.get_ob()[21:29]

    def _U_val(self) -> NDArray[np.float64]:
        p = 0.8
        max_strength = 0.5

        # start of episode, random wind
        if self._t == 0:
            mag = self.rng.uniform(0.0, max_strength)
            angle = self.rng.uniform(0.0, 2.0 * np.pi)
            u = np.array([mag * np.cos(angle), mag * np.sin(angle)], dtype=np.float64)

        # Markov chain
        else:
            if self.rng.random() < p:
                u = self._U[-1]
            else:
                mag = self.rng.uniform(0.0, max_strength)
                angle = self.rng.uniform(0.0, 2.0 * np.pi)
                u = np.array([mag * np.cos(angle), mag * np.sin(angle)], dtype=np.float64)

        return u

    def _F_val(self) -> NDArray[np.float64]:
        p = 0.8
        p_gust = 0.5
        min_strength = 0.3
        max_strength = 1.0

        # start of episode, random gust or no gust
        if self._t == 0:
            if self.rng.random() < p_gust:
                f = np.zeros(2, dtype=np.float64)
            else:
                mag = self.rng.uniform(min_strength, max_strength)
                angle = self.rng.uniform(0.0, 2.0 * np.pi)
                f = np.array([mag * np.cos(angle), mag * np.sin(angle)], dtype=np.float64)

        # Markov chain
        else:
            if self.rng.random() < p:
                f = self._F[-1]
            else:
                if self.rng.random() < p_gust:
                    f = np.zeros(2, dtype=np.float64)
                else:
                    mag = self.rng.uniform(min_strength, max_strength)
                    angle = self.rng.uniform(0.0, 2.0 * np.pi)
                    f = np.array([mag * np.cos(angle), mag * np.sin(angle)], dtype=np.float64)

        return f

    def _W(self) -> NDArray[np.float64]:
        # favor U to increase bias
        u_norm = float(np.linalg.norm(self._U[-1]))
        f_norm = float(np.linalg.norm(self._F[-1]))

        u_signal = u_norm
        f_signal = np.tanh(f_norm)

        alpha_u = 2.0
        alpha_f = 0.8

        base_noise = 0.05
        extra_noise = 0.25 * (f_norm / (f_norm + 1e-6))
        noise_std = base_noise + extra_noise
        noise = float(self.rng.normal(0.0, noise_std))

        w = np.tanh(alpha_u * u_signal + alpha_f * f_signal + noise)
        return np.array([w], dtype=np.float64)

    def observation(self, history: bool = False) -> Dict[str, Any]:
        obs = {}

        if history:
            if 'P' not in self.hidden_dims:
                obs['P'] = self.P
            if 'O' not in self.hidden_dims:
                obs['O'] = self.O
            if 'A' not in self.hidden_dims:
                obs['A'] = self.A
            if 'L' not in self.hidden_dims:
                obs['L'] = self.L
            if 'T' not in self.hidden_dims:
                obs['T'] = self.T
            if 'J' not in self.hidden_dims:
                obs['J'] = self.J
            if 'F' not in self.hidden_dims:
                obs['F'] = self._F
            if 'W' not in self.hidden_dims:
                obs['W'] = self.W
            if 'X' not in self.hidden_dims:
                obs['X'] = self.X
        else:
            if 'P' not in self.hidden_dims:
                obs['P'] = self.P[-1]
            if 'O' not in self.hidden_dims:
                obs['O'] = self.O[-1]
            if 'A' not in self.hidden_dims:
                obs['A'] = self.A[-1]
            if 'L' not in self.hidden_dims:
                obs['L'] = self.L[-1]
            if 'T' not in self.hidden_dims:
                obs['T'] = self.T[-1]
            if 'J' not in self.hidden_dims:
                obs['J'] = self.J[-1]
            if 'F' not in self.hidden_dims:
                obs['F'] = self._F[-1]
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
        self.O = [self._O()]
        self.A = [self._A()]
        self.L = [self._L()]
        self.T = [self._T()]
        self.J = [self._J()]
        self._U = [self._U_val()]
        self._F = [self._F_val()]
        self.W = [self._W()]
        self.X = []
        self._Y = []

        obs = self.observation(history=history)
        hiddens = {}
        if 'P' in self.hidden_dims:
            hiddens['P'] = self.P
        if 'O' in self.hidden_dims:
            hiddens['O'] = self.O
        if 'A' in self.hidden_dims:
            hiddens['A'] = self.A
        if 'L' in self.hidden_dims:
            hiddens['L'] = self.L
        if 'T' in self.hidden_dims:
            hiddens['T'] = self.T
        if 'J' in self.hidden_dims:
            hiddens['J'] = self.J
        if 'F' in self.hidden_dims:
            hiddens['F'] = self._F
        if 'W' in self.hidden_dims:
            hiddens['W'] = self.W
        if 'X' in self.hidden_dims:
            hiddens['X'] = self.X

        info = {'Y': self._Y, 'U': self._U, 'env_obs': env_obs, 'env_info': env_info, 'hidden_obs': hiddens}
        return obs, info

    def action(self, P: List[NDArray[np.float64]], O: List[NDArray[np.float64]], A: List[NDArray[np.float64]], L: List[NDArray[np.float64]], T: List[NDArray[np.float64]], J: List[NDArray[np.float64]], F: List[NDArray[np.float64]], W: List[NDArray[np.float64]]) -> ActType:
        # placeholder behavior policy
        return self.action_space.sample()
    
    def compute_success(self) -> bool:
        ag = np.asarray(self.P[-1][:2], dtype=np.float64)
        dg = np.asarray(self._goal_xy, dtype=np.float64)
        diff = ag - dg
        dist = np.linalg.norm(diff)
        return dist <= self.success_radius

    def _reward(self) -> float:
        ag = np.asarray(self.P[-1][:2], dtype=np.float64)
        dg = np.asarray(self._goal_xy, dtype=np.float64)
        diff = ag - dg
        dist = np.linalg.norm(diff)

        # wind penalty
        u_norm = float(np.linalg.norm(self._U[-1]))
        dist_norm = dist / 30.0 # approximate maze size
        lambda_u = 0.5
        wind_penalty = lambda_u * u_norm * (1.0 + dist_norm)

        return -1.0 - wind_penalty + float(self.compute_success())

    def step(self, action: Any, history: bool = False, show_reward: bool = True) -> Tuple[dict, float, bool, bool, dict]:
        # actions are float32, but observed actions need to be float64
        self.X.append(np.asarray(action, dtype=np.float64))

        # apply wind and gust
        u = self._U_val()
        f = self._F_val()
        self._U.append(u)
        self._F.append(f)

        model = self._env.env.env.env.model
        data = self._env.env.env.env.data

        data.xfrc_applied[:] = 0.0 # reset last step's forces
        torso_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'torso')
        total_force = np.array([
            u[0] + f[0],
            u[1] + f[1],
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
        self.O.append(self._O())
        self.A.append(self._A())
        self.L.append(self._L())
        self.T.append(self._T())
        self.J.append(self._J())
        self.W.append(self._W())

        reward = self._reward()
        self._Y.append(reward)

        obs = self.observation(history=history)

        hiddens = {}
        if 'P' in self.hidden_dims:
            hiddens['P'] = self.P
        if 'O' in self.hidden_dims:
            hiddens['O'] = self.O
        if 'A' in self.hidden_dims:
            hiddens['A'] = self.A
        if 'L' in self.hidden_dims:
            hiddens['L'] = self.L
        if 'T' in self.hidden_dims:
            hiddens['T'] = self.T
        if 'J' in self.hidden_dims:
            hiddens['J'] = self.J
        if 'F' in self.hidden_dims:
            hiddens['F'] = self._F
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
        state_vars = ['P', 'O', 'A', 'L', 'T', 'J', 'F', 'W']
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
            p, o, a, l, t, j, f, w, x = base, base + 1, base + 2, base + 3, base + 4, base + 5, base + 6, base + 7, base + 8

            base_graph[j][a] = 1 # joint angular velocities affect joint angles
            base_graph[j][t] = 1 # joint angular velocities affect torso angular velocity
            base_graph[j][l] = 1 # joint angular velocities affect torso linear velocity

            base_graph[t][o] = 1 # torso angular velocity affects torso orientation

            base_graph[a][o] = 1 # joint angles affect torso orientation
            base_graph[a][l] = 1 # joint angles affect torso linear velocity

            base_graph[o][l] = 1 # torso orientation affects torso linear velocity

            base_graph[l][p] = 1 # torso linear velocity affects position

            # state influence decision-making
            base_graph[l][x] = 1
            base_graph[p][x] = 1
            base_graph[t][x] = 1
            base_graph[o][x] = 1
            base_graph[j][x] = 1
            base_graph[a][x] = 1

            base_graph[p][y] = 1 # reward is based on position

            # wind confounding
            base_graph[f][w] = 1
            base_graph[f][l] = 1
            base_graph[f][x] = 1

            conf_graph[w][y] = 1
            conf_graph[l][y] = 1
            conf_graph[w][l] = 1

        # intra-timestep edges for terminal state
        base_term = H * len(variables)
        p, o, a, l, t, j = base_term, base_term + 1, base_term + 2, base_term + 3, base_term + 4, base_term + 5

        base_graph[j][a] = 1 # joint angular velocities affect joint angles
        base_graph[j][t] = 1 # joint angular velocities affect torso angular velocity
        base_graph[j][l] = 1 # joint angular velocities affect torso linear velocity

        base_graph[t][o] = 1 # torso angular velocity affects torso orientation

        base_graph[a][o] = 1 # joint angles affect torso orientation
        base_graph[a][l] = 1 # joint angles affect torso linear velocity

        base_graph[o][l] = 1 # torso orientation affects torso linear velocity

        base_graph[l][p] = 1 # torso linear velocity affects position

        base_graph[p][y] = 1 # reward is based on position

        # inter-timstep edges
        for step in range(H):
            base = step * len(variables)
            base_next = (step + 1) * len(variables)

            p, o, a, l, t, j, x = base, base + 1, base + 2, base + 3, base + 4, base + 5, base + 6
            p2, o2, a2, l2, t2, j2, x2 = base_next, base_next + 1, base_next + 2, base_next + 3, base_next + 4, base_next + 5, base_next + 6

            base_graph[x][j2] = 1 # torque impacts joint angular velocity

            # state persistence
            base_graph[p][p2] = 1
            base_graph[o][o2] = 1
            base_graph[a][a2] = 1
            base_graph[l][l2] = 1
            base_graph[t][t2] = 1
            base_graph[j][j2] = 1

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
        all_vars = ['P', 'O', 'A', 'L', 'T', 'J', 'F', 'W', 'X']
        observed = [v for v in all_vars if v not in self.hidden_dims]
        unobserved = list(self.hidden_dims) + ['Y', 'U']
        return observed, unobserved

class AntMazeExpert:
    def __init__(self, env_id: str = 'antmaze-medium-navigate-singletask-task1-v0', num_steps: int = 1000, hidden_dims: Optional[Set[str]] = None, success_radius: float = 5.0, goal_xy: np.ndarray = np.array([20.0, 20.0]), seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed)
        self.num_steps = num_steps

        _, train, test = ogbench.make_env_and_datasets(env_id, max_episode_steps=num_steps, compact_dataset=True)
        self._expert_trajs = {k: np.concatenate([train[k], test[k]], axis=0) for k in train.keys()}

        self.num_eps = len(self._expert_trajs['observations']) // 1000 - 1
        self._t = -1 # reset brings it to 0

        self.success_radius = success_radius
        self._goal_xy = goal_xy

        self.hidden_dims = hidden_dims if hidden_dims is not None else set()

        self.P = []
        self.O = []
        self.A = []
        self.L = []
        self.T = []
        self.J = []
        self.X = []
        self._Y = []

    def observation(self) -> Dict[str, Any]:
        obs = {}

        if 'P' not in self.hidden_dims:
            obs['P'] = self.P
        if 'O' not in self.hidden_dims:
            obs['O'] = self.O
        if 'A' not in self.hidden_dims:
            obs['A'] = self.A
        if 'L' not in self.hidden_dims:
            obs['L'] = self.L
        if 'T' not in self.hidden_dims:
            obs['T'] = self.T
        if 'J' not in self.hidden_dims:
            obs['J'] = self.J
        if 'X' not in self.hidden_dims:
            obs['X'] = self.X

        return obs

    def reset(self):
        self._t += 1
        if self._t >= len(self._expert_trajs['observations']):
            self._t = 0
            print('Warning: Expert trajectories exhausted. If this is a reset to begin do(), ignore this message.')

        self.P = []
        self.O = []
        self.A = []
        self.L = []
        self.T = []
        self.J = []
        self.X = []
        self._Y = []

    def _reward(self) -> float:
        ag = np.asarray(self.P[-1][:2], dtype=np.float64)
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
        self.O.append(obs[3:7])
        self.A.append(obs[7:15])
        self.L.append(obs[15:18])
        self.T.append(obs[18:21])
        self.J.append(obs[21:29])
        self.X.append(action)
        
        reward = self._reward()
        self._Y.append(reward)

        hiddens = {}
        if 'P' in self.hidden_dims:
            hiddens['P'] = self.P
        if 'O' in self.hidden_dims:
            hiddens['O'] = self.O
        if 'A' in self.hidden_dims:
            hiddens['A'] = self.A
        if 'L' in self.hidden_dims:
            hiddens['L'] = self.L
        if 'T' in self.hidden_dims:
            hiddens['T'] = self.T
        if 'J' in self.hidden_dims:
            hiddens['J'] = self.J
        if 'X' in self.hidden_dims:
            hiddens['X'] = self.X

        info = {'Y': self._Y, 'env_obs': obs, 'env_info': {}, 'hidden_obs': hiddens, 'natural_action': action}

        self._t += 1
        return self.observation(), reward, terminated, truncated, info

class AntMazePCH(PCH):
    def __init__(self, env_id: str = 'antmaze-medium-navigate-singletask-task1-v0', num_steps: int = 1000, hidden_dims: Optional[Set[str]] = None, success_radius: float = 5.0, seed: Optional[int] = None):
        # initialize underlying SCM
        self.env = AntMazeSCM(env_id=env_id, num_steps=num_steps, hidden_dims=hidden_dims, success_radius=success_radius, seed=seed)
        self.expert = AntMazeExpert(env_id=env_id, num_steps=num_steps, hidden_dims=hidden_dims, success_radius=success_radius, goal_xy=self.env._goal_xy, seed=seed)
        super().__init__()

        self.last_actor_is_expert = True

    def see(self, behavioral_policy=None, show_reward=True) -> Tuple[Any, Any, float, bool, bool, Dict[str, Any]]:
        P = self.env.P
        O = self.env.O
        A = self.env.A
        L = self.env.L
        T = self.env.T
        J = self.env.J
        _F = self.env._F
        W = self.env.W

        if behavioral_policy is not None:
            action = behavioral_policy(P, O, A, L, T, J, _F, W)
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
        O = self.env.O
        A = self.env.A
        L = self.env.L
        T = self.env.T
        J = self.env.J
        _F = self.env._F
        W = self.env.W

        intuition = self.env.action(P, O, A, L, T, J, _F, W)
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