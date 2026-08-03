import ogbench
import mujoco
import numpy as np
from numpy.typing import NDArray

from typing import Dict, Optional, List, Tuple, Any, Set
from causal_gym import SCM, PCH
from causal_gym.core import ActType, Graph
from gymnasium import spaces


class HumanoidMazeSCM(SCM):
    """
    Confounded HumanoidMaze environment with latent seismic tremor.

    Observation space (69D base + 2D W):
        P  (0-1):   2D  - XY maze position
        A  (2-22):  21D - Joint angles
        H  (23):    1D  - Head height
        E  (24-35): 12D - Extremities (4 limbs x 3D egocentric)
        V  (36-38): 3D  - Torso vertical (up-vector)
        C  (39-41): 3D  - COM (center of mass) velocity (HIDDEN from imitator)
        J  (42-68): 27D - Joint velocities (root 6D + hinges 21D)
        W  (new):   2D  - Noisy perceived ground vibration (observed by imitator)
        X  (action):21D - Joint torques

    Confounding:
        - C (COM velocity): Expert-only, important for navigation/course correction
        - W (perceived vibration): Noisy proxy mixing C's XY velocity with tremor direction
        - U (seismic tremor): Latent, affects dynamics (shakes floor/pushes torso), W, and reward Y
    """

    def __init__(
        self,
        env_id: str = 'humanoidmaze-medium-navigate-singletask-task1-v0',
        num_steps: int = 2000,
        expert_mode: bool = False,
        custom_hidden: Optional[Set[str]] = None,
        success_radius: float = 10.0,
        seed: Optional[int] = None,
        confound_strength: float = 1.0
    ):
        super().__init__()

        assert 0.0 <= confound_strength <= 1.0, 'confound_strength must be in [0, 1]'

        self.rng = np.random.default_rng(seed)
        self.num_steps = num_steps
        self.expert_mode = expert_mode
        self.confound_strength = confound_strength # 0 = no train/test shift, 1 = full inversion (original "-w_raw")
        self.hidden_dims = set() if expert_mode else {'C'}
        if custom_hidden is not None:
            self.hidden_dims = custom_hidden
        self._t = 0

        self._env = ogbench.make_env_and_datasets(env_id, env_only=True, max_episode_steps=num_steps)
        _, info = self._env.reset(seed=seed)
        self._goal_xy = info['goal'][:2]
        self.success_radius = success_radius

        # State history
        self.P = []  # position, 2D (maze XY)
        self.A = []  # joint angles, 21D
        self.H = []  # head height, 1D
        self.E = []  # extremities, 12D (4 limbs x 3D)
        self.V = []  # torso vertical, 3D (up-vector)
        self.C = []  # COM velocity, 3D (HIDDEN from imitator)
        self.J = []  # joint velocities, 27D (root 6D + hinges 21D)
        self.X = []  # action, 21D
        self._Y = [] # sparse reward

        # Seismic tremor confounding
        self._U = []  # seismic tremor (latent to all)
        self.W = []   # noisy perceived ground vibration

        self.action_space = self._env.action_space  # Box(-1.0, 1.0, (21,), float32)

        # Build observation space
        act_low = np.asarray(self.action_space.low, dtype=self.action_space.dtype)
        act_high = np.asarray(self.action_space.high, dtype=self.action_space.dtype)

        full_obs = {
            'P': spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float64),
            'A': spaces.Box(low=-np.inf, high=np.inf, shape=(21,), dtype=np.float64),
            'H': spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float64),
            'E': spaces.Box(low=-np.inf, high=np.inf, shape=(12,), dtype=np.float64),
            'V': spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float64),
            'C': spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float64),
            'J': spaces.Box(low=-np.inf, high=np.inf, shape=(27,), dtype=np.float64),
            'W': spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float64),
            'X': spaces.Box(low=act_low, high=act_high, shape=self.action_space.shape, dtype=np.float64)
        }

        self.observation_space = spaces.Dict({k: v for k, v in full_obs.items() if k not in self.hidden_dims})

    def _get_base_env(self):
        """Unwrap to get base MazeEnv."""
        current = self._env
        while hasattr(current, 'env'):
            current = current.env
        return current

    def _P(self) -> NDArray[np.float64]:
        """XY maze position (indices 0-1)."""
        return self._get_base_env().get_ob()[0:2].astype(np.float64)

    def _A(self) -> NDArray[np.float64]:
        """Joint angles (indices 2-22, 21D)."""
        return self._get_base_env().get_ob()[2:23].astype(np.float64)

    def _H(self) -> NDArray[np.float64]:
        """Head height (index 23, 1D)."""
        return np.array([self._get_base_env().get_ob()[23]], dtype=np.float64)

    def _E(self) -> NDArray[np.float64]:
        """Extremities (indices 24-35, 12D)."""
        return self._get_base_env().get_ob()[24:36].astype(np.float64)

    def _V(self) -> NDArray[np.float64]:
        """Torso vertical up-vector (indices 36-38, 3D)."""
        return self._get_base_env().get_ob()[36:39].astype(np.float64)

    def _C(self) -> NDArray[np.float64]:
        """COM velocity (indices 39-41, 3D)."""
        return self._get_base_env().get_ob()[39:42].astype(np.float64)

    def _J(self) -> NDArray[np.float64]:
        """Joint velocities (indices 42-68, 27D)."""
        return self._get_base_env().get_ob()[42:69].astype(np.float64)

    def _U_val(self) -> NDArray[np.float64]:
        """Compute seismic tremor value (piecewise-constant impulse process)."""
        p_gust = 0.05
        min_strength = 2.0
        max_strength = 5.0

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
        """
        Compute noisy perceived ground vibration, contaminated by seismic tremor.

        C = COM velocity is a 3D vector. C[:2] gives ground-plane velocity.
        W mixes this velocity signal with tremor direction, creating a biased proxy.
        """
        u = self._U[-1]
        u_norm = float(np.linalg.norm(u))
        if u_norm > 1e-8:
            u_hat = u / u_norm
        else:
            u_hat = np.zeros_like(u)

        # Get XY ground-plane velocity from COM velocity
        c = self.C[-1]
        vel_xy = c[:2]

        alpha_c = 0.9   # Dominant velocity signal
        alpha_u = 0.1   # Small tremor contamination
        noise_std = 0.05

        base = alpha_c * vel_xy + alpha_u * u_hat
        noise = self.rng.normal(0.0, noise_std, size=2)
        w_raw = base + noise

        if not self.expert_mode:
            # rotate w_raw by theta = confound_strength * pi: 0 -> identity (no train/test
            # shift), 1 -> full 180 deg inversion, i.e. the original "-w_raw" backward sensor
            theta = self.confound_strength * np.pi
            cos_t, sin_t = np.cos(theta), np.sin(theta)
            w_raw = np.array([
                cos_t * w_raw[0] - sin_t * w_raw[1],
                sin_t * w_raw[0] + cos_t * w_raw[1],
            ], dtype=np.float64)

        w = np.tanh(w_raw)
        return w.astype(np.float64)

    def observation(self, history: bool = False) -> Dict[str, Any]:
        """Get current observation dict, excluding hidden dimensions."""
        obs = {}

        if history:
            if 'P' not in self.hidden_dims:
                obs['P'] = self.P
            if 'A' not in self.hidden_dims:
                obs['A'] = self.A
            if 'H' not in self.hidden_dims:
                obs['H'] = self.H
            if 'E' not in self.hidden_dims:
                obs['E'] = self.E
            if 'V' not in self.hidden_dims:
                obs['V'] = self.V
            if 'C' not in self.hidden_dims:
                obs['C'] = self.C
            if 'J' not in self.hidden_dims:
                obs['J'] = self.J
            if 'W' not in self.hidden_dims:
                obs['W'] = self.W
            if 'X' not in self.hidden_dims:
                obs['X'] = self.X
        else:
            if 'P' not in self.hidden_dims:
                obs['P'] = self.P[-1]
            if 'A' not in self.hidden_dims:
                obs['A'] = self.A[-1]
            if 'H' not in self.hidden_dims:
                obs['H'] = self.H[-1]
            if 'E' not in self.hidden_dims:
                obs['E'] = self.E[-1]
            if 'V' not in self.hidden_dims:
                obs['V'] = self.V[-1]
            if 'C' not in self.hidden_dims:
                obs['C'] = self.C[-1]
            if 'J' not in self.hidden_dims:
                obs['J'] = self.J[-1]
            if 'W' not in self.hidden_dims:
                obs['W'] = self.W[-1]
            if 'X' not in self.hidden_dims:
                obs['X'] = self.X[-1] if len(self.X) > 0 else np.zeros(self.action_space.shape, dtype=np.float64)

        return obs

    def reset(self, history: bool = False, seed: Optional[int] = None):
        """Reset the environment."""
        self.rng = np.random.default_rng(seed)
        env_obs, env_info = self._env.reset()

        self._t = 0
        self.P = [self._P()]
        self.A = [self._A()]
        self.H = [self._H()]
        self.E = [self._E()]
        self.V = [self._V()]
        self.C = [self._C()]
        self.J = [self._J()]
        self._U = [self._U_val()]
        self.W = [self._W()]
        self.X = []
        self._Y = []

        obs = self.observation(history=history)
        hiddens = {}
        for var in ['P', 'A', 'H', 'E', 'V', 'C', 'J', 'W', 'X']:
            if var in self.hidden_dims:
                hiddens[var] = getattr(self, var)

        info = {'Y': self._Y, 'U': self._U, 'env_obs': env_obs, 'env_info': env_info, 'hidden_obs': hiddens}
        return obs, info

    def action(self, P, A, H, E, V, C, J, W) -> ActType:
        """Placeholder behavior policy."""
        return self.action_space.sample()

    def compute_success(self) -> bool:
        """Check if agent reached the goal."""
        ag = np.asarray(self.P[-1][:2], dtype=np.float64)
        dg = np.asarray(self._goal_xy, dtype=np.float64)
        dist = np.linalg.norm(ag - dg)
        return dist <= self.success_radius

    def _reward(self) -> float:
        """Compute reward with tremor penalty."""
        ag = np.asarray(self.P[-1][:2], dtype=np.float64)
        dg = np.asarray(self._goal_xy, dtype=np.float64)
        dist = np.linalg.norm(ag - dg)

        # Tremor penalty
        u_norm = float(np.linalg.norm(self._U[-1]))
        dist_norm = dist / 25.0  # Approximate maze size
        lambda_u = 1.0
        tremor_penalty = lambda_u * u_norm * (1.0 + dist_norm)

        return float(self.compute_success()) - tremor_penalty

    def step(self, action: Any, history: bool = False, show_reward: bool = True) -> Tuple[dict, float, bool, bool, dict]:
        """Take a step in the environment."""
        self.X.append(np.asarray(action, dtype=np.float64))

        # Apply seismic tremor
        u = self._U_val()
        self._U.append(u)

        base_env = self._get_base_env()
        model = base_env.model
        data = base_env.data

        data.xfrc_applied[:] = 0.0
        torso_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'torso')
        total_force = np.array([
            u[0], u[1], 0.0,  # XY force, no Z
            0.0, 0.0, 0.0     # No torque
        ], dtype=np.float64)
        data.xfrc_applied[torso_id] = total_force

        # Step environment
        env_obs, reward, _, truncated, env_info = self._env.step(action)
        terminated = self.compute_success()

        # Update state
        self._t += 1
        self.P.append(self._P())
        self.A.append(self._A())
        self.H.append(self._H())
        self.E.append(self._E())
        self.V.append(self._V())
        self.C.append(self._C())
        self.J.append(self._J())
        self.W.append(self._W())

        reward = self._reward()
        self._Y.append(reward)

        obs = self.observation(history=history)

        hiddens = {}
        for var in ['P', 'A', 'H', 'E', 'V', 'C', 'J', 'W', 'X']:
            if var in self.hidden_dims:
                hiddens[var] = getattr(self, var)

        info = {'Y': self._Y, 'U': self._U, 'env_obs': env_obs, 'env_info': env_info, 'hidden_obs': hiddens}
        return obs, reward if show_reward else None, terminated, truncated, info

    def render(self):
        return self._env.render()

    def close(self):
        self._env.close()

    @property
    def get_graph(self) -> Graph:
        """Build causal graph for the environment."""
        state_vars = ['P', 'A', 'H', 'E', 'V', 'C', 'J', 'W']
        variables = state_vars + ['X']
        H = self.num_steps
        n = H * len(variables) + len(state_vars) + 1

        nodes = {}
        i = 0
        for t in range(H):
            for v in variables:
                nodes[i] = f'{v}{t}'
                i += 1

        # Terminal state
        for v in state_vars:
            nodes[i] = f'{v}{H}'
            i += 1

        nodes[i] = f'Y{H}'

        base_graph = [[0] * n for _ in range(n)]
        conf_graph = [[0] * n for _ in range(n)]

        y = n - 1

        # Intra-timestep edges
        for step in range(H):
            base = step * len(variables)
            p, a, h, e, v, c, j, w, x = (base + i for i in range(9))

            # Physical dependencies
            base_graph[j][a] = 1  # Joint velocities affect joint angles
            base_graph[j][c] = 1  # Joint velocities affect COM velocity
            base_graph[a][v] = 1  # Joint angles affect torso vertical
            base_graph[a][e] = 1  # Joint angles affect extremities
            base_graph[a][h] = 1  # Joint angles affect head height
            base_graph[c][p] = 1  # COM velocity affects position
            base_graph[v][c] = 1  # Torso vertical affects COM velocity
            base_graph[h][p] = 1  # TODO verify consistency
            base_graph[e][p] = 1  # TODO verify consistency

            # State influences decision-making
            base_graph[p][x] = 1
            base_graph[a][x] = 1
            base_graph[h][x] = 1
            base_graph[e][x] = 1
            base_graph[v][x] = 1
            base_graph[c][x] = 1
            base_graph[j][x] = 1

            # Reward depends on position
            base_graph[p][y] = 1

            # Tremor confounding: C -> W
            base_graph[c][w] = 1

            # Bidirected edges (confounding via U — seismic tremor)
            conf_graph[w][y] = 1
            conf_graph[c][y] = 1
            conf_graph[w][c] = 1

        # Terminal state edges
        base_term = H * len(variables)
        p, a, h, e, v, c, j, w = (base_term + i for i in range(8))

        base_graph[j][a] = 1
        base_graph[j][c] = 1
        base_graph[a][v] = 1
        base_graph[a][e] = 1
        base_graph[a][h] = 1
        base_graph[c][p] = 1
        base_graph[v][c] = 1
        base_graph[p][y] = 1
        base_graph[c][w] = 1

        conf_graph[w][y] = 1
        conf_graph[c][y] = 1
        conf_graph[w][c] = 1

        # Inter-timestep edges
        for step in range(H):
            base = step * len(variables)
            base_next = (step + 1) * len(variables)

            p, a, h, e, v, c, j, w, x = (base + i for i in range(9))
            p2, a2, h2, e2, v2, c2, j2, w2, x2 = (base_next + i for i in range(9))

            base_graph[x][j2] = 1  # Action affects joint velocities

            # State persistence
            base_graph[p][p2] = 1
            base_graph[a][a2] = 1
            base_graph[h][h2] = 1
            base_graph[e][e2] = 1
            base_graph[v][v2] = 1
            base_graph[c][c2] = 1
            base_graph[j][j2] = 1
            base_graph[x][x2] = 1

        nodes_list = [{'name': n} for n in nodes.values()]
        edges = []
        for i in range(len(nodes_list)):
            for j in range(len(nodes_list)):
                if base_graph[i][j] == 1:
                    edges.append({'from_': nodes_list[i]['name'], 'to_': nodes_list[j]['name'], 'type_': 'directed'})
                if conf_graph[i][j] == 1:
                    edges.append({'from_': nodes_list[i]['name'], 'to_': nodes_list[j]['name'], 'type_': 'bidirected'})

        return Graph(nodes=nodes_list, edges=edges)

    @property
    def observed_unobserved_vars(self) -> Tuple[List[str], List[str]]:
        all_vars = ['P', 'A', 'H', 'E', 'V', 'C', 'J', 'W', 'X']
        observed = [v for v in all_vars if v not in self.hidden_dims]
        unobserved = list(self.hidden_dims) + ['U', 'Y']
        return observed, unobserved


class HumanoidMazeExpert:
    """Expert trajectory loader for HumanoidMaze."""

    def __init__(
        self,
        env_id: str = 'humanoidmaze-medium-navigate-singletask-task1-v0',
        num_steps: int = 2000,
        expert_mode: bool = False,
        success_radius: float = 10.0,
        goal_xy: np.ndarray = np.array([20.0, 20.0]),
        seed: Optional[int] = None
    ):
        self.rng = np.random.default_rng(seed)
        self.num_steps = num_steps

        _, train, test = ogbench.make_env_and_datasets(env_id, max_episode_steps=num_steps, compact_dataset=True)
        self._expert_trajs = {k: np.concatenate([train[k], test[k]], axis=0) for k in train.keys()}

        self.num_eps = len(self._expert_trajs['observations']) // num_steps - 1
        self._t = -1

        self.success_radius = success_radius
        self._goal_xy = goal_xy

        self.hidden_dims = set() if expert_mode else {'C'}

        self.P = []
        self.A = []
        self.H = []
        self.E = []
        self.V = []
        self.C = []
        self.J = []
        self.X = []
        self._Y = []

    def observation(self) -> Dict[str, Any]:
        obs = {}
        for var in ['P', 'A', 'H', 'E', 'V', 'C', 'J', 'X']:
            if var not in self.hidden_dims:
                obs[var] = getattr(self, var)
        return obs

    def reset(self):
        self._t += 1
        if self._t >= len(self._expert_trajs['observations']):
            self._t = 0
            print('Warning: Expert trajectories exhausted. If this is a reset to begin do(), ignore this message.')

        self.P = []
        self.A = []
        self.H = []
        self.E = []
        self.V = []
        self.C = []
        self.J = []
        self.X = []
        self._Y = []

    def _reward(self) -> float:
        ag = np.asarray(self.P[-1][:2], dtype=np.float64)
        dg = np.asarray(self._goal_xy, dtype=np.float64)
        dist = np.linalg.norm(ag - dg)
        success = float(dist <= self.success_radius)
        return -1.0 + success

    def step(self):
        obs = self._expert_trajs['observations'][self._t]
        action = self._expert_trajs['actions'][self._t].astype(np.float64)

        terminated = self._expert_trajs['terminals'][self._t]
        truncated = False

        # Parse observation (69D)
        self.P.append(obs[0:2].astype(np.float64))
        self.A.append(obs[2:23].astype(np.float64))
        self.H.append(np.array([obs[23]], dtype=np.float64))
        self.E.append(obs[24:36].astype(np.float64))
        self.V.append(obs[36:39].astype(np.float64))
        self.C.append(obs[39:42].astype(np.float64))
        self.J.append(obs[42:69].astype(np.float64))
        self.X.append(action)

        reward = self._reward()
        self._Y.append(reward)

        hiddens = {}
        for var in ['P', 'A', 'H', 'E', 'V', 'C', 'J', 'X']:
            if var in self.hidden_dims:
                hiddens[var] = getattr(self, var)

        info = {'Y': self._Y, 'env_obs': obs, 'env_info': {}, 'hidden_obs': hiddens, 'natural_action': action}

        self._t += 1
        return self.observation(), reward, terminated, truncated, info


class HumanoidMazePCH(PCH):
    """Pearl Causal Hierarchy wrapper for HumanoidMaze."""

    def __init__(
        self,
        env_id: str = 'humanoidmaze-medium-navigate-singletask-task1-v0',
        num_steps: int = 2000,
        expert_mode: bool = False,
        custom_hidden: Optional[Set[str]] = None,
        success_radius: float = 10.0,
        seed: Optional[int] = None,
        confound_strength: float = 1.0
    ):
        self.env = HumanoidMazeSCM(
            env_id=env_id,
            num_steps=num_steps,
            expert_mode=expert_mode,
            custom_hidden=custom_hidden,
            success_radius=success_radius,
            seed=seed,
            confound_strength=confound_strength
        )
        self.expert = HumanoidMazeExpert(
            env_id=env_id,
            num_steps=num_steps,
            expert_mode=True,
            success_radius=success_radius,
            goal_xy=self.env._goal_xy,
            seed=seed
        )
        super().__init__()

        self.last_actor_is_expert = True

    def see(self, behavioral_policy=None, show_reward=True) -> Tuple[Any, Any, float, bool, bool, Dict[str, Any]]:
        """Observational step."""
        P = self.env.P
        A = self.env.A
        H = self.env.H
        E = self.env.E
        V = self.env.V
        C = self.env.C
        J = self.env.J
        W = self.env.W

        if behavioral_policy is not None:
            action = behavioral_policy(P, A, H, E, V, C, J, W)
        else:
            return self.expert.step()

        obs, reward, terminated, truncated, info = self.env.step(action, history=True, show_reward=show_reward)
        info['natural_action'] = action

        self.last_actor_is_expert = True
        return obs, reward, terminated, truncated, info

    def do(self, do_policy, show_reward=True):
        """Interventional step with forced action."""
        action = do_policy(self.env.observation(history=True))
        o, r, term, trunc, info = self.env.step(action, history=True, show_reward=show_reward)
        info['action'] = action

        self.last_actor_is_expert = False
        return o, r, term, trunc, info

    def ctf_do(self, ctf_policy):
        """Counterfactual policy intervention."""
        P = self.env.P
        A = self.env.A
        H = self.env.H
        E = self.env.E
        V = self.env.V
        C = self.env.C
        J = self.env.J
        W = self.env.W

        intuition = self.env.action(P, A, H, E, V, C, J, W)
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