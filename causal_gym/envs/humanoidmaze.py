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
    Causal SCM wrapper for OGBench HumanoidMaze environment.

    Observation space (69D) verified from ogbench/locomaze/humanoid.py get_ob():
        0-1:   P  - XY maze position (qpos[:2])
        2-22:  A  - Joint angles, 21 hinge joints (qpos[7:])
        23:    H  - Head height, Z-coord of head body (xpos[2,2])
        24-35: E  - Extremities in torso frame (12D): left_hand, left_foot, right_hand, right_foot
        36-38: G  - Gravity in torso frame: world Z-axis in torso body coords (xmat[1,[6,7,8]])
        39-41: C  - COM velocity: torso subtree linear velocity (sensordata[0:3])
        42-44: L  - Root linear velocity (qvel[0:3])
        45-47: T  - Root angular velocity (qvel[3:6])
        48-68: J  - Joint velocities, 21D (qvel[6:27])

    Action space: 21D joint torques in [-1, 1]
    """

    def __init__(
        self,
        env_id: str = 'humanoidmaze-medium-navigate-singletask-task1-v0',
        num_steps: int = 1000,
        expert_mode: bool = False,
        custom_hidden: Optional[Set[str]] = None,
        success_radius: float = 5.0,
        seed: Optional[int] = None
    ):
        super().__init__()

        self.rng = np.random.default_rng(seed)
        self.num_steps = num_steps
        self.expert_mode = expert_mode
        self.hidden_dims: Set[str] = set() if expert_mode else {'G'}
        if custom_hidden is not None:
            self.hidden_dims = custom_hidden
        self._t = 0

        self._env = ogbench.make_env_and_datasets(env_id, env_only=True, max_episode_steps=num_steps)
        _, info = self._env.reset(seed=seed)
        self._goal_xy = info['goal'][:2]
        self.success_radius = success_radius

        # ===== Causal State Variables =====
        # Position & Configuration
        self.P: List[NDArray[np.float64]] = []  # XY maze position (2D) - obs[0:2]
        self.A: List[NDArray[np.float64]] = []  # Joint angles (21D) - obs[2:23]
        self.H: List[NDArray[np.float64]] = []  # Head height (1D) - obs[23]
        self.E: List[NDArray[np.float64]] = []  # Extremities in torso frame (12D) - obs[24:36]
                                                 # Order: left_hand(3), left_foot(3), right_hand(3), right_foot(3)

        # Orientation
        self.G: List[NDArray[np.float64]] = []  # Gravity in torso frame (3D) - obs[36:39]
                                                 # World Z-axis in torso body coordinates
                                                 # [0,0,1] when perfectly upright

        # Velocities
        self.C: List[NDArray[np.float64]] = []  # COM velocity (3D) - obs[39:42]
                                                 # Torso subtree center-of-mass linear velocity
        self.L: List[NDArray[np.float64]] = []  # Root linear velocity (3D) - obs[42:45]
        self.T: List[NDArray[np.float64]] = []  # Root angular velocity (3D) - obs[45:48]
        self.J: List[NDArray[np.float64]] = []  # Joint velocities (21D) - obs[48:69]

        # Action
        self.X: List[NDArray[np.float64]] = []  # Joint torques (21D)

        # Reward
        self._Y: List[float] = []  # Sparse reward (goal-reaching)

        # ===== Wind Confounding =====
        self._U: List[NDArray[np.float64]] = []  # Wind field - latent confounder (2D)
        self.W: List[NDArray[np.float64]] = []   # Noisy heading indicator (2D)

        # ===== Spaces =====
        self.action_space = self._env.action_space  # Box(-1.0, 1.0, (21,), float32)

        act_low = np.asarray(self.action_space.low, dtype=np.float64)
        act_high = np.asarray(self.action_space.high, dtype=np.float64)

        full_obs = {
            # Position & Configuration
            'P': spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float64),
            'A': spaces.Box(low=-np.inf, high=np.inf, shape=(21,), dtype=np.float64),
            'H': spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float64),
            'E': spaces.Box(low=-np.inf, high=np.inf, shape=(12,), dtype=np.float64),
            # Orientation
            'G': spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float64),
            # Velocities
            'C': spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float64),
            'L': spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float64),
            'T': spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float64),
            'J': spaces.Box(low=-np.inf, high=np.inf, shape=(21,), dtype=np.float64),
            # Confounding signal
            'W': spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float64),
            # Action
            'X': spaces.Box(low=act_low, high=act_high, shape=(21,), dtype=np.float64),
        }

        self.observation_space = spaces.Dict({k: v for k, v in full_obs.items() if k not in self.hidden_dims})

    # ===== State Extraction Helpers =====
    # These extract current state from the underlying environment's observation

    def _get_base_env(self):
        """Get the innermost MazeEnv with get_ob() method."""
        current = self._env
        while hasattr(current, 'env'):
            current = current.env
        return current

    def _get_obs(self) -> NDArray[np.float64]:
        """Get the full 69D observation from the base environment."""
        return self._get_base_env().get_ob()

    def _P(self) -> NDArray[np.float64]:
        """XY maze position (2D) - obs[0:2]"""
        return self._get_obs()[0:2]

    def _A(self) -> NDArray[np.float64]:
        """Joint angles (21D) - obs[2:23]"""
        return self._get_obs()[2:23]

    def _H(self) -> NDArray[np.float64]:
        """Head height (1D) - obs[23:24]"""
        return self._get_obs()[23:24]

    def _E(self) -> NDArray[np.float64]:
        """Extremities in torso frame (12D) - obs[24:36]"""
        return self._get_obs()[24:36]

    def _G(self) -> NDArray[np.float64]:
        """Gravity direction in torso frame (3D) - obs[36:39]"""
        return self._get_obs()[36:39]

    def _C(self) -> NDArray[np.float64]:
        """COM velocity (3D) - obs[39:42]"""
        return self._get_obs()[39:42]

    def _L(self) -> NDArray[np.float64]:
        """Root linear velocity (3D) - obs[42:45]"""
        return self._get_obs()[42:45]

    def _T(self) -> NDArray[np.float64]:
        """Root angular velocity (3D) - obs[45:48]"""
        return self._get_obs()[45:48]

    def _J(self) -> NDArray[np.float64]:
        """Joint velocities (21D) - obs[48:69]"""
        return self._get_obs()[48:69]

    def _U_val(self) -> NDArray[np.float64]:
        p_gust = 0.05
        min_strength = 2.0
        max_strength = 5.0

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
        u_norm = float(np.linalg.norm(u))
        if u_norm > 1e-8:
            u_hat = u / u_norm
        else:
            u_hat = np.zeros_like(u)

        # Get heading from torso rotation matrix
        # xmat[1] is torso's rotation matrix (body index 1)
        # First column (indices 0,3,6) is torso's local X-axis in world coords
        base_env = self._get_base_env()
        torso_xmat = base_env.data.xmat[1].reshape(3, 3)
        torso_forward = torso_xmat[:, 0]  # Body X-axis in world frame
        yaw = float(np.arctan2(torso_forward[1], torso_forward[0]))
        heading = np.array(
            [np.cos(yaw), np.sin(yaw)],
            dtype=np.float64
        )

        alpha_o = 0.9 # dominant
        alpha_u = 0.1 # small U contamination
        noise_std = 0.05

        # if not self.expert_mode:
        #     # rely on wind first and yaw second now
        #     alpha_o = 0.1
        #     alpha_u = 0.9
        #     noise_std = 0.5

        base = alpha_o * heading + alpha_u * u_hat

        noise = self.rng.normal(0.0, noise_std, size=2)

        w_raw = base + noise

        if not self.expert_mode:
            w_raw = -w_raw # "backward sensor"

        # bound
        w = np.tanh(w_raw)
        return w.astype(np.float64)

    def observation(self, history: bool = False) -> Dict[str, Any]:
        """Return current observation dict, optionally with full history."""
        obs = {}
        # All state variables in order
        state_vars = ['P', 'A', 'H', 'E', 'G', 'C', 'L', 'T', 'J', 'W', 'X']
        state_data = {
            'P': self.P, 'A': self.A, 'H': self.H, 'E': self.E, 'G': self.G,
            'C': self.C, 'L': self.L, 'T': self.T, 'J': self.J, 'W': self.W, 'X': self.X
        }

        for var in state_vars:
            if var not in self.hidden_dims:
                data = state_data[var]
                if history:
                    obs[var] = data
                else:
                    if var == 'X' and len(data) == 0:
                        obs[var] = np.zeros((21,), dtype=np.float64)
                    else:
                        obs[var] = data[-1] if len(data) > 0 else None

        return obs

    def reset(self, history: bool = False, seed: Optional[int] = None):
        """Reset the environment and initialize all state variables."""
        self.rng = np.random.default_rng(seed)
        env_obs, env_info = self._env.reset()

        self._t = 0

        # Initialize state variables from observation
        self.P = [self._P()]
        self.A = [self._A()]
        self.H = [self._H()]
        self.E = [self._E()]
        self.G = [self._G()]
        self.C = [self._C()]
        self.L = [self._L()]
        self.T = [self._T()]
        self.J = [self._J()]

        # Wind confounding
        self._U = [self._U_val()]
        self.W = [self._W()]

        # Action and reward (empty at reset)
        self.X = []
        self._Y = []

        obs = self.observation(history=history)

        # Collect hidden variables for info dict
        hiddens = {}
        state_vars = ['P', 'A', 'H', 'E', 'G', 'C', 'L', 'T', 'J', 'W', 'X']
        state_data = {
            'P': self.P, 'A': self.A, 'H': self.H, 'E': self.E, 'G': self.G,
            'C': self.C, 'L': self.L, 'T': self.T, 'J': self.J, 'W': self.W, 'X': self.X
        }
        for var in state_vars:
            if var in self.hidden_dims:
                hiddens[var] = state_data[var]

        info = {'Y': self._Y, 'U': self._U, 'env_obs': env_obs, 'env_info': env_info, 'hidden_obs': hiddens}
        return obs, info

    def action(
        self,
        P: List[NDArray[np.float64]],
        A: List[NDArray[np.float64]],
        H: List[NDArray[np.float64]],
        E: List[NDArray[np.float64]],
        G: List[NDArray[np.float64]],
        C: List[NDArray[np.float64]],
        L: List[NDArray[np.float64]],
        T: List[NDArray[np.float64]],
        J: List[NDArray[np.float64]],
        W: List[NDArray[np.float64]]
    ) -> ActType:
        """Placeholder behavior policy - returns random action."""
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
        dist_norm = dist / 25.0 # approximate maze size
        lambda_u = 1.0
        wind_penalty = lambda_u * u_norm * (1.0 + dist_norm)

        return float(self.compute_success()) - wind_penalty

    def step(self, action: Any, history: bool = False, show_reward: bool = True) -> Tuple[dict, float, bool, bool, dict]:
        """Execute one environment step with wind confounding."""
        # Actions are float32, but observed actions need to be float64
        self.X.append(np.asarray(action, dtype=np.float64))

        # Apply wind and gust
        u = self._U_val()
        self._U.append(u)

        base_env = self._get_base_env()
        model = base_env.model
        data = base_env.data

        data.xfrc_applied[:] = 0.0  # Reset last step's forces
        torso_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'torso')
        total_force = np.array([
            u[0],
            u[1],
            0.0,  # no z
            0.0, 0.0, 0.0  # no torque
        ], dtype=np.float64)
        data.xfrc_applied[torso_id] = total_force

        # Step environment
        env_obs, reward, _, truncated, env_info = self._env.step(action)
        terminated = self.compute_success()

        # Update all SCM state variables
        self._t += 1
        self.P.append(self._P())
        self.A.append(self._A())
        self.H.append(self._H())
        self.E.append(self._E())
        self.G.append(self._G())
        self.C.append(self._C())
        self.L.append(self._L())
        self.T.append(self._T())
        self.J.append(self._J())
        self.W.append(self._W())

        reward = self._reward()
        self._Y.append(reward)

        obs = self.observation(history=history)

        # Collect hidden variables for info dict
        hiddens = {}
        state_vars = ['P', 'A', 'H', 'E', 'G', 'C', 'L', 'T', 'J', 'W', 'X']
        state_data = {
            'P': self.P, 'A': self.A, 'H': self.H, 'E': self.E, 'G': self.G,
            'C': self.C, 'L': self.L, 'T': self.T, 'J': self.J, 'W': self.W, 'X': self.X
        }
        for var in state_vars:
            if var in self.hidden_dims:
                hiddens[var] = state_data[var]

        info = {'Y': self._Y, 'U': self._U, 'env_obs': env_obs, 'env_info': env_info, 'hidden_obs': hiddens}
        return obs, reward if show_reward else None, terminated, truncated, info

    def render(self):
        return self._env.render()

    def close(self):
        self._env.close()

    @property
    def get_graph(self) -> Graph:
        """
        Construct the causal graph for HumanoidMaze.

        Variables per timestep:
            P  - XY maze position (2D)
            A  - Joint angles (21D)
            H  - Head height (1D)
            E  - Extremities in torso frame (12D)
            G  - Gravity in torso frame (3D) - HIDDEN from imitator
            C  - COM velocity (3D)
            L  - Root linear velocity (3D)
            T  - Root angular velocity (3D)
            J  - Joint velocities (21D)
            W  - Noisy heading signal (2D) - confounded by wind U
            X  - Action/torques (21D)

        Confounding structure:
            U (wind, latent) causes:
              - Physical push → affects L, C, G (tilt)
              - Sensor corruption → affects W
            This creates backdoor paths that confound imitation learning.
        """
        state_vars = ['P', 'A', 'H', 'E', 'G', 'C', 'L', 'T', 'J', 'W']
        variables = state_vars + ['X']
        num_state = len(state_vars)
        num_vars = len(variables)
        H = self.num_steps

        # Total nodes: H timesteps * (state + action) + terminal state + reward
        n = H * num_vars + num_state + 1

        # Build node index mapping
        nodes = {}
        i = 0
        for t in range(H):
            for v in variables:
                nodes[i] = f'{v}{t}'
                i += 1

        # Terminal state (no action)
        for v in state_vars:
            nodes[i] = f'{v}{H}'
            i += 1

        nodes[i] = f'Y{H}'  # Reward comes last

        # Initialize adjacency matrices
        base_graph = [[0] * n for _ in range(n)]
        conf_graph = [[0] * n for _ in range(n)]

        y_idx = n - 1  # Reward node index

        def get_idx(step: int, var: str) -> int:
            """Get node index for variable at timestep."""
            if step == H:
                # Terminal state
                return H * num_vars + state_vars.index(var)
            else:
                return step * num_vars + variables.index(var)

        # ===== Intra-timestep edges =====
        for step in range(H + 1):  # Include terminal state
            # Variable indices for this timestep
            p = get_idx(step, 'P')
            a = get_idx(step, 'A')
            h = get_idx(step, 'H')
            e = get_idx(step, 'E')
            g = get_idx(step, 'G')
            c = get_idx(step, 'C')
            l = get_idx(step, 'L')
            t = get_idx(step, 'T')
            j = get_idx(step, 'J')
            w = get_idx(step, 'W')

            # --- Physics: Velocities affect configuration ---
            base_graph[j][a] = 1  # Joint velocities → joint angles
            base_graph[t][g] = 1  # Angular velocity → orientation/tilt
            base_graph[l][p] = 1  # Linear velocity → position

            # --- Configuration interactions ---
            base_graph[a][h] = 1  # Joint angles → head height (pose)
            base_graph[a][e] = 1  # Joint angles → extremity positions
            base_graph[g][e] = 1  # Orientation → extremities (torso frame)
            base_graph[a][g] = 1  # Joint config → balance/tilt

            # --- Velocity interactions ---
            base_graph[j][t] = 1  # Joint motion → torso angular vel
            base_graph[j][l] = 1  # Joint motion → torso linear vel
            base_graph[j][c] = 1  # Joint motion → COM velocity
            base_graph[t][l] = 1  # Angular vel → linear vel coupling
            base_graph[t][c] = 1  # Angular vel → COM vel
            base_graph[g][l] = 1  # Orientation affects velocity (gravity)
            base_graph[g][c] = 1  # Orientation affects COM vel
            base_graph[l][c] = 1  # Root vel → COM vel

            # --- Wind confounding ---
            base_graph[g][w] = 1  # Orientation determines heading in W

            # Bidirected edges (U causes both):
            conf_graph[w][l] = 1  # W ↔ L (wind affects both)
            conf_graph[w][c] = 1  # W ↔ C (wind affects both)
            conf_graph[l][c] = 1  # L ↔ C (wind affects both)
            conf_graph[g][l] = 1  # G ↔ L (wind tilts and pushes)
            conf_graph[g][c] = 1  # G ↔ C (wind tilts and affects COM)

            # --- Reward edges ---
            base_graph[p][y_idx] = 1  # Position → reward (goal distance)

            # Confounded reward paths:
            conf_graph[w][y_idx] = 1  # W ↔ Y (wind confounds)
            conf_graph[l][y_idx] = 1  # L ↔ Y (wind confounds)
            conf_graph[c][y_idx] = 1  # C ↔ Y (wind confounds)

            # --- State → Action (policy, non-terminal only) ---
            if step < H:
                x = get_idx(step, 'X')

                # All observed state informs action
                base_graph[p][x] = 1
                base_graph[a][x] = 1
                base_graph[h][x] = 1
                base_graph[e][x] = 1
                base_graph[g][x] = 1  # Expert sees G
                base_graph[c][x] = 1
                base_graph[l][x] = 1
                base_graph[t][x] = 1
                base_graph[j][x] = 1
                base_graph[w][x] = 1

        # ===== Inter-timestep edges =====
        for step in range(H):
            # Current timestep indices
            p = get_idx(step, 'P')
            a = get_idx(step, 'A')
            h = get_idx(step, 'H')
            e = get_idx(step, 'E')
            g = get_idx(step, 'G')
            c = get_idx(step, 'C')
            l = get_idx(step, 'L')
            t = get_idx(step, 'T')
            j = get_idx(step, 'J')
            w = get_idx(step, 'W')
            x = get_idx(step, 'X')

            # Next timestep indices
            p2 = get_idx(step + 1, 'P')
            a2 = get_idx(step + 1, 'A')
            h2 = get_idx(step + 1, 'H')
            e2 = get_idx(step + 1, 'E')
            g2 = get_idx(step + 1, 'G')
            c2 = get_idx(step + 1, 'C')
            l2 = get_idx(step + 1, 'L')
            t2 = get_idx(step + 1, 'T')
            j2 = get_idx(step + 1, 'J')
            w2 = get_idx(step + 1, 'W')

            # Action affects next state
            base_graph[x][j2] = 1  # Torques → next joint velocities

            # State persistence (momentum, inertia)
            base_graph[p][p2] = 1
            base_graph[a][a2] = 1
            base_graph[h][h2] = 1
            base_graph[e][e2] = 1
            base_graph[g][g2] = 1
            base_graph[c][c2] = 1
            base_graph[l][l2] = 1
            base_graph[t][t2] = 1
            base_graph[j][j2] = 1
            base_graph[w][w2] = 1

            # Action persistence (policy smoothness)
            if step + 1 < H:
                x2 = get_idx(step + 1, 'X')
                base_graph[x][x2] = 1

        # ===== Build Graph object =====
        node_list = [{'name': name} for name in nodes.values()]
        edges = []

        for i in range(n):
            for j in range(n):
                if base_graph[i][j] == 1:
                    edges.append({
                        'from_': node_list[i]['name'],
                        'to_': node_list[j]['name'],
                        'type_': 'directed'
                    })
                if conf_graph[i][j] == 1:
                    edges.append({
                        'from_': node_list[i]['name'],
                        'to_': node_list[j]['name'],
                        'type_': 'bidirected'
                    })

        return Graph(nodes=node_list, edges=edges)

    @property
    def observed_unobserved_vars(self) -> Tuple[list[str], list[str]]:
        all_vars = ['P', 'A', 'H', 'E', 'G', 'C', 'L', 'T', 'J', 'W', 'X']
        observed = [v for v in all_vars if v not in self.hidden_dims]
        unobserved = list(self.hidden_dims) + ['U', 'Y']
        return observed, unobserved

class AntMazeExpert:
    def __init__(self, env_id: str = 'antmaze-medium-navigate-singletask-task1-v0', num_steps: int = 1000, expert_mode: bool = False, success_radius: float = 5.0, goal_xy: np.ndarray = np.array([20.0, 20.0]), seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed)
        self.num_steps = num_steps

        _, train, test = ogbench.make_env_and_datasets(env_id, max_episode_steps=num_steps, compact_dataset=True)
        self._expert_trajs = {k: np.concatenate([train[k], test[k]], axis=0) for k in train.keys()}

        self.num_eps = len(self._expert_trajs['observations']) // 1000 - 1
        self._t = -1 # reset brings it to 0

        self.success_radius = success_radius
        self._goal_xy = goal_xy

        self.hidden_dims = set() if expert_mode else {'O'}

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
    def __init__(self, env_id: str = 'antmaze-medium-navigate-singletask-task1-v0', num_steps: int = 1000, expert_mode: bool = False, custom_hidden=None, success_radius: float = 5.0, seed: Optional[int] = None):
        # initialize underlying SCM
        self.env = AntMazeSCM(env_id=env_id, num_steps=num_steps, expert_mode=expert_mode, custom_hidden=custom_hidden, success_radius=success_radius, seed=seed)
        self.expert = AntMazeExpert(env_id=env_id, num_steps=num_steps, expert_mode=True, success_radius=success_radius, goal_xy=self.env._goal_xy, seed=seed)
        super().__init__()

        self.last_actor_is_expert = True

    def see(self, behavioral_policy=None, show_reward=True) -> Tuple[Any, Any, float, bool, bool, Dict[str, Any]]:
        P = self.env.P
        O = self.env.O
        A = self.env.A
        L = self.env.L
        T = self.env.T
        J = self.env.J
        W = self.env.W

        if behavioral_policy is not None:
            action = behavioral_policy(P, O, A, L, T, J, W)
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
        W = self.env.W

        intuition = self.env.action(P, O, A, L, T, J, W)
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