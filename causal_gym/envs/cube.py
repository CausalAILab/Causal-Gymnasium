import ogbench
import mujoco
import numpy as np
from numpy.typing import NDArray

from typing import Dict, Optional, List, Tuple, Any, Set
from causal_gym import SCM, PCH
from causal_gym.core import ActType, Graph
from gymnasium import spaces


# Constants matching CubeEnv.compute_observation() scaling
_XYZ_CENTER = np.array([0.425, 0.0, 0.0])
_XYZ_SCALER = 10.0
_GRIPPER_SCALER = 3.0
_NUM_CUBES = 4

# Per-cube variable names (position, orientation)
_POS_VARS = ('A', 'S', 'D', 'F')  # cube 0-3 positions (3D each)
_ORI_VARS = ('L', 'M', 'N', 'O')  # cube 0-3 orientations (6D each)


def _cube_var_names():
    """Return ordered list of per-cube variable names: A, L, S, M, D, N, F, O."""
    names = []
    for p, o in zip(_POS_VARS, _ORI_VARS):
        names.extend([p, o])
    return names


def _all_var_names():
    """All causal variable names in observation order."""
    return ['Q', 'V', 'E', 'H', 'G', 'C'] + _cube_var_names() + ['W', 'X']


def _state_var_names():
    """All state variable names (no action X)."""
    return ['Q', 'V', 'E', 'H', 'G', 'C'] + _cube_var_names() + ['W']


class CubeSCM(SCM):
    """
    Confounded cube-quadruple manipulation environment with latent arm disturbance.

    Observation space (55D base + 2D W):
        Q    (0-5):   6D  - Arm joint positions (6 UR5e joints)
        V    (6-11):  6D  - Arm joint velocities (HIDDEN from imitator)
        E    (12-14): 3D  - End-effector position (scaled)
        H    (15-16): 2D  - End-effector yaw [cos(yaw), sin(yaw)]
        G    (17):    1D  - Gripper opening (scaled)
        C    (18):    1D  - Gripper contact force
        A    (19-21): 3D  - Cube 0 position (scaled)
        L    (22-27): 6D  - Cube 0 orientation (quaternion 4D + cos/sin yaw 2D)
        S    (28-30): 3D  - Cube 1 position (scaled)
        M    (31-36): 6D  - Cube 1 orientation
        D    (37-39): 3D  - Cube 2 position (scaled)
        N    (40-45): 6D  - Cube 2 orientation
        F    (46-48): 3D  - Cube 3 position (scaled)
        O    (49-54): 6D  - Cube 3 orientation
        W    (new):   2D  - Noisy effector velocity proxy (observed by imitator)
        X    (action):5D  - Relative EE control (dx, dy, dz, dyaw, dgripper)

    Confounding:
        - V (joint velocities): Expert-only, important for smooth reactive control
        - W (perceived velocity): Noisy proxy mixing effector XY velocity with disturbance direction
        - U (arm disturbance): Latent, applies force to wrist, affects dynamics, W, and reward Y
    """

    def __init__(
        self,
        env_id: str = 'cube-quadruple-play-singletask-task2-v0',
        num_steps: int = 1000,
        expert_mode: bool = False,
        custom_hidden: Optional[Set[str]] = None,
        seed: Optional[int] = None,
    ):
        super().__init__()

        self.rng = np.random.default_rng(seed)
        self.num_steps = num_steps
        self.expert_mode = expert_mode
        self.hidden_dims = set() if expert_mode else {'V'}
        if custom_hidden is not None:
            self.hidden_dims = custom_hidden
        self._t = 0
        self._num_cubes = _NUM_CUBES

        self._env = ogbench.make_env_and_datasets(env_id, env_only=True, max_episode_steps=num_steps)
        _, info = self._env.reset(seed=seed)
        self._goal = info['goal']

        # State history — proprioception
        self.Q = []    # arm joint positions, 6D
        self.V = []    # arm joint velocities, 6D (HIDDEN from imitator)
        self.E = []    # end-effector position, 3D (scaled)
        self.H = []    # end-effector yaw, 2D [cos, sin]
        self.G = []    # gripper opening, 1D (scaled)
        self.C = []    # gripper contact, 1D

        # State history — per-cube positions (3D scaled)
        self.A = []    # cube 0 position
        self.S = []    # cube 1 position
        self.D = []    # cube 2 position
        self.F = []    # cube 3 position

        # State history — per-cube orientations (6D)
        self.L = []    # cube 0 orientation
        self.M = []    # cube 1 orientation
        self.N = []    # cube 2 orientation
        self.O = []    # cube 3 orientation

        self.X = []    # action, 5D
        self._Y = []   # reward

        # Arm disturbance confounding
        self._U = []   # latent disturbance force (latent to all)
        self.W = []    # noisy effector velocity proxy

        self.action_space = self._env.action_space  # Box(-1, 1, (5,), float32)

        # Build observation space
        act_low = np.asarray(self.action_space.low, dtype=self.action_space.dtype)
        act_high = np.asarray(self.action_space.high, dtype=self.action_space.dtype)

        full_obs = {
            'Q': spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float64),
            'V': spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float64),
            'E': spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float64),
            'H': spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float64),
            'G': spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float64),
            'C': spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float64),
        }
        for p in _POS_VARS:
            full_obs[p] = spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float64)
        for o in _ORI_VARS:
            full_obs[o] = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float64)
        full_obs['W'] = spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float64)
        full_obs['X'] = spaces.Box(low=act_low, high=act_high, shape=self.action_space.shape, dtype=np.float64)

        self.observation_space = spaces.Dict({k: v for k, v in full_obs.items() if k not in self.hidden_dims})

    def _get_base_env(self):
        """Unwrap to get the base CubeEnv."""
        current = self._env
        while hasattr(current, 'env'):
            current = current.env
        return current

    def _read_ob_info(self) -> dict:
        """Read the structured observation info from the base env."""
        base = self._get_base_env()
        return base.compute_ob_info()

    def _read_all_state(self):
        """Read all state variables from the base env at once."""
        ob_info = self._read_ob_info()

        q = ob_info['proprio/joint_pos'].astype(np.float64)
        v = ob_info['proprio/joint_vel'].astype(np.float64)
        e = ((ob_info['proprio/effector_pos'] - _XYZ_CENTER) * _XYZ_SCALER).astype(np.float64)
        yaw = ob_info['proprio/effector_yaw']
        h = np.array([np.cos(yaw[0]), np.sin(yaw[0])], dtype=np.float64)
        g = (ob_info['proprio/gripper_opening'] * _GRIPPER_SCALER).astype(np.float64)
        c = ob_info['proprio/gripper_contact'].astype(np.float64)

        bs = []
        os_ = []
        for i in range(self._num_cubes):
            b = ((ob_info[f'privileged/block_{i}_pos'] - _XYZ_CENTER) * _XYZ_SCALER).astype(np.float64)
            quat = ob_info[f'privileged/block_{i}_quat'].astype(np.float64)
            block_yaw = ob_info[f'privileged/block_{i}_yaw']
            o = np.concatenate([quat, [np.cos(block_yaw[0]), np.sin(block_yaw[0])]]).astype(np.float64)
            bs.append(b)
            os_.append(o)

        return q, v, e, h, g, c, bs, os_

    def _get_var(self, name):
        """Get a variable's history list by name."""
        return getattr(self, name)

    def _U_val(self) -> NDArray[np.float64]:
        """Compute arm disturbance value (piecewise-constant impulse process)."""
        p_impulse = 0.05
        min_strength = 1.0
        max_strength = 3.0

        if self._t == 0:
            if self.rng.random() < p_impulse:
                angle = self.rng.uniform(0.0, 2.0 * np.pi)
                mag = self.rng.uniform(min_strength, max_strength)
                u = np.array([mag * np.cos(angle), mag * np.sin(angle)], dtype=np.float64)
            else:
                u = np.zeros(2, dtype=np.float64)
        else:
            if self._t % 5 != 0:
                return self._U[-1]
            if self.rng.random() < p_impulse:
                angle = self.rng.uniform(0.0, 2.0 * np.pi)
                mag = self.rng.uniform(min_strength, max_strength)
                u = np.array([mag * np.cos(angle), mag * np.sin(angle)], dtype=np.float64)
            else:
                u = np.zeros(2, dtype=np.float64)

        return u

    def _W(self) -> NDArray[np.float64]:
        """
        Compute noisy effector velocity proxy, contaminated by arm disturbance.

        Uses finite-difference of effector position to approximate XY velocity.
        Mixes this velocity signal with disturbance direction, creating a biased proxy.
        """
        u = self._U[-1]
        u_norm = float(np.linalg.norm(u))
        if u_norm > 1e-8:
            u_hat = u / u_norm
        else:
            u_hat = np.zeros_like(u)

        # Compute effector XY velocity via finite difference of E (scaled positions)
        if len(self.E) >= 2:
            eff_vel_xy = (self.E[-1][:2] - self.E[-2][:2])
        else:
            eff_vel_xy = np.zeros(2, dtype=np.float64)

        alpha_v = 0.9   # Dominant velocity signal
        alpha_u = 0.1   # Small disturbance contamination
        noise_std = 0.05

        base = alpha_v * eff_vel_xy + alpha_u * u_hat
        noise = self.rng.normal(0.0, noise_std, size=2)
        w_raw = base + noise

        if not self.expert_mode:
            w_raw = -w_raw  # Inverted sensor for imitator

        w = np.tanh(w_raw)
        return w.astype(np.float64)

    def observation(self, history: bool = False) -> Dict[str, Any]:
        obs = {}
        all_vars = _all_var_names()

        if history:
            for var in all_vars:
                if var not in self.hidden_dims:
                    obs[var] = self._get_var(var)
        else:
            for var in all_vars:
                if var not in self.hidden_dims:
                    val = self._get_var(var)
                    if var == 'X':
                        obs[var] = val[-1] if len(val) > 0 else np.zeros(self.action_space.shape, dtype=np.float64)
                    else:
                        obs[var] = val[-1]

        return obs

    def reset(self, history: bool = False, seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed)
        env_obs, env_info = self._env.reset()
        self._goal = env_info['goal']

        self._t = 0
        q, v, e, h, g, c, bs, os_ = self._read_all_state()
        self.Q = [q]
        self.V = [v]
        self.E = [e]
        self.H = [h]
        self.G = [g]
        self.C = [c]
        for i, (p, o) in enumerate(zip(_POS_VARS, _ORI_VARS)):
            setattr(self, p, [bs[i]])
            setattr(self, o, [os_[i]])
        self._U = [self._U_val()]
        self.W = [self._W()]
        self.X = []
        self._Y = []

        obs = self.observation(history=history)
        hiddens = {}
        for var in _all_var_names():
            if var in self.hidden_dims:
                hiddens[var] = self._get_var(var)

        info = {'Y': self._Y, 'U': self._U, 'env_obs': env_obs, 'env_info': env_info, 'hidden_obs': hiddens}
        return obs, info

    def action(self, **kwargs) -> ActType:
        """Placeholder behavior policy."""
        return self.action_space.sample()

    def compute_success(self) -> bool:
        """Check if all cubes are at goal positions (within 0.04 threshold each)."""
        base = self._get_base_env()
        for i in range(self._num_cubes):
            obj_pos = base._data.joint(f'object_joint_{i}').qpos[:3]
            tar_pos = base._data.mocap_pos[base._cube_target_mocap_ids[i]]
            if float(np.linalg.norm(obj_pos - tar_pos)) > 0.04:
                return False
        return True

    def _compute_successes(self) -> List[bool]:
        """Check per-cube success."""
        base = self._get_base_env()
        successes = []
        for i in range(self._num_cubes):
            obj_pos = base._data.joint(f'object_joint_{i}').qpos[:3]
            tar_pos = base._data.mocap_pos[base._cube_target_mocap_ids[i]]
            successes.append(float(np.linalg.norm(obj_pos - tar_pos)) <= 0.04)
        return successes

    def _reward(self) -> float:
        """Compute reward with disturbance penalty.

        Base reward: sum(successes) - num_cubes (matches CubeEnv.compute_reward).
        Disturbance penalty: proportional to ||U|| and mean distance to goals.
        """
        base = self._get_base_env()
        successes = self._compute_successes()
        base_reward = float(sum(successes)) - float(self._num_cubes)

        # Mean distance across all cubes for penalty scaling
        total_dist = 0.0
        for i in range(self._num_cubes):
            obj_pos = base._data.joint(f'object_joint_{i}').qpos[:3]
            tar_pos = base._data.mocap_pos[base._cube_target_mocap_ids[i]]
            total_dist += float(np.linalg.norm(obj_pos - tar_pos))
        mean_dist = total_dist / self._num_cubes

        # Disturbance penalty
        u_norm = float(np.linalg.norm(self._U[-1]))
        workspace_size = 0.3
        dist_norm = mean_dist / workspace_size
        lambda_u = 1.0
        disturbance_penalty = lambda_u * u_norm * (1.0 + dist_norm)

        return base_reward - disturbance_penalty

    def step(self, action: Any, history: bool = False, show_reward: bool = True) -> Tuple[dict, float, bool, bool, dict]:
        self.X.append(np.asarray(action, dtype=np.float64))

        # Compute and apply arm disturbance
        u = self._U_val()
        self._U.append(u)

        base = self._get_base_env()
        model = base._model
        data = base._data

        data.xfrc_applied[:] = 0.0  # reset last step's forces
        wrist_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'ur5e/wrist_1_link')
        total_force = np.array([
            u[0], u[1], 0.0,  # XY force, no Z
            0.0, 0.0, 0.0     # no torque
        ], dtype=np.float64)
        data.xfrc_applied[wrist_id] = total_force

        # Step environment
        env_obs, reward, _, truncated, env_info = self._env.step(action)
        terminated = self.compute_success()

        # Update state
        self._t += 1
        q, v, e, h, g, c, bs, os_ = self._read_all_state()
        self.Q.append(q)
        self.V.append(v)
        self.E.append(e)
        self.H.append(h)
        self.G.append(g)
        self.C.append(c)
        for i, (p, o) in enumerate(zip(_POS_VARS, _ORI_VARS)):
            getattr(self, p).append(bs[i])
            getattr(self, o).append(os_[i])
        self.W.append(self._W())

        reward = self._reward()
        self._Y.append(reward)

        obs = self.observation(history=history)

        hiddens = {}
        for var in _all_var_names():
            if var in self.hidden_dims:
                hiddens[var] = self._get_var(var)

        info = {'Y': self._Y, 'U': self._U, 'env_obs': env_obs, 'env_info': env_info, 'hidden_obs': hiddens}
        return obs, reward if show_reward else None, terminated, truncated, info

    def render(self):
        return self._env.render()

    def close(self):
        self._env.close()

    @property
    def get_graph(self) -> Graph:
        """Build the causal graph for the cube-quadruple environment.

        Structure (per timestep t, for t in 0..H-1):
            - All state vars except W -> X_t  (state influences action)
            - X_t -> all state vars at t+1   (action affects next state)
            - State persistence: each var_t -> var_{t+1}
            - V_t -> W_t                     (hidden joint velocities determine proxy)
            - pos_i_t -> Y                   (reward depends on cube positions)

        Confounding (bidirected, from latent U):
            - W <-> Y   (proxy contaminated by U, reward penalized by U)
            - V <-> Y   (hidden var affected by U force, reward penalized by U)
            - W <-> V   (proxy mixes V signal with U contamination)
        """
        state_vars = _state_var_names()
        variables = state_vars + ['X']
        T = self.num_steps
        n = T * len(variables) + len(state_vars) + 1  # +terminal state vars +Y

        # Build node index mapping
        nodes = {}
        idx = 0
        for t in range(T):
            for v in variables:
                nodes[idx] = f'{v}{t}'
                idx += 1

        # Terminal state (no action at terminal)
        for v in state_vars:
            nodes[idx] = f'{v}{T}'
            idx += 1

        nodes[idx] = f'Y{T}'

        base_graph = [[0] * n for _ in range(n)]
        conf_graph = [[0] * n for _ in range(n)]

        y = n - 1  # Y node index

        nv = len(variables)  # vars per timestep (state + X)

        # Helper to get node index for variable v at time t
        def vi(var_name, t):
            if t == T:
                # Terminal state: only state vars, no X
                base_term = T * nv
                return base_term + state_vars.index(var_name)
            return t * nv + variables.index(var_name)

        # --- Intra-timestep edges (for each step including terminal) ---
        for t in range(T + 1):
            # V -> W (hidden joint velocities determine velocity proxy)
            base_graph[vi('V', t)][vi('W', t)] = 1

            # Cube position vars -> Y (reward depends on cube positions)
            for p in _POS_VARS:
                base_graph[vi(p, t)][y] = 1

            # Confounding via latent U
            conf_graph[vi('W', t)][y] = 1      # W <-> Y
            conf_graph[vi('V', t)][y] = 1      # V <-> Y
            conf_graph[vi('W', t)][vi('V', t)] = 1  # W <-> V

            # All state vars except W -> X (state influences action)
            if t < T:
                for sv in state_vars:
                    if sv == 'W':
                        continue  # W has no outgoing edges
                    base_graph[vi(sv, t)][vi('X', t)] = 1

        # --- Inter-timestep edges ---
        for t in range(T):
            # X -> all state vars at t+1 except W (W is derived from V and U only)
            for sv in state_vars:
                if sv == 'W':
                    continue
                base_graph[vi('X', t)][vi(sv, t + 1)] = 1

            # State persistence: each var_t -> var_{t+1} except W (no outgoing edges)
            for sv in state_vars:
                if sv == 'W':
                    continue
                base_graph[vi(sv, t)][vi(sv, t + 1)] = 1

            # Action persistence
            if t + 1 < T:
                base_graph[vi('X', t)][vi('X', t + 1)] = 1

        # Convert to Graph
        nodes_list = [{'name': nm} for nm in nodes.values()]
        edges = []
        for i in range(n):
            for j in range(n):
                if base_graph[i][j] == 1:
                    edges.append({'from_': nodes_list[i]['name'], 'to_': nodes_list[j]['name'], 'type_': 'directed'})
                if conf_graph[i][j] == 1:
                    edges.append({'from_': nodes_list[i]['name'], 'to_': nodes_list[j]['name'], 'type_': 'bidirected'})

        return Graph(nodes=nodes_list, edges=edges)

    @property
    def observed_unobserved_vars(self) -> Tuple[List[str], List[str]]:
        all_vars = _all_var_names()
        observed = [v for v in all_vars if v not in self.hidden_dims]
        unobserved = list(self.hidden_dims) + ['U', 'Y']
        return observed, unobserved


class CubeExpert:
    """Expert trajectory loader for the cube-quadruple environment."""

    def __init__(
        self,
        env_id: str = 'cube-quadruple-play-singletask-task2-v0',
        num_steps: int = 1000,
        expert_mode: bool = False,
        goal: Optional[np.ndarray] = None,
        seed: Optional[int] = None,
    ):
        self.rng = np.random.default_rng(seed)
        self.num_steps = num_steps
        self._num_cubes = _NUM_CUBES

        _, train, test = ogbench.make_env_and_datasets(env_id, max_episode_steps=num_steps, compact_dataset=True)
        self._expert_trajs = {k: np.concatenate([train[k], test[k]], axis=0) for k in train.keys()}

        self.num_eps = len(self._expert_trajs['observations']) // num_steps - 1
        self._t = -1

        self._goal = goal
        self.hidden_dims = set() if expert_mode else {'V'}

        self.Q = []
        self.V = []
        self.E = []
        self.H = []
        self.G = []
        self.C = []
        self.A = []
        self.S = []
        self.D = []
        self.F = []
        self.L = []
        self.M = []
        self.N = []
        self.O = []
        self.X = []
        self._Y = []

    def _get_var(self, name):
        """Get a variable's history list by name."""
        return getattr(self, name)

    def observation(self) -> Dict[str, Any]:
        obs = {}
        for var in _all_var_names():
            if var == 'W':
                continue  # Expert trajectories don't have W
            if var not in self.hidden_dims:
                obs[var] = self._get_var(var)
        return obs

    def reset(self):
        self._t += 1
        if self._t >= len(self._expert_trajs['observations']):
            self._t = 0
            print('Warning: Expert trajectories exhausted. If this is a reset to begin do(), ignore this message.')

        self.Q = []
        self.V = []
        self.E = []
        self.H = []
        self.G = []
        self.C = []
        self.A = []
        self.S = []
        self.D = []
        self.F = []
        self.L = []
        self.M = []
        self.N = []
        self.O = []
        self.X = []
        self._Y = []

    def _reward(self) -> float:
        """Placeholder reward for expert trajectories."""
        return 0.0

    def step(self):
        obs = self._expert_trajs['observations'][self._t]
        action = self._expert_trajs['actions'][self._t].astype(np.float64)

        terminated = self._expert_trajs['terminals'][self._t]
        truncated = False

        # Parse the 55D observation vector into causal variables
        # Proprioception (19D)
        self.Q.append(obs[0:6].astype(np.float64))       # arm joint positions
        self.V.append(obs[6:12].astype(np.float64))       # arm joint velocities
        self.E.append(obs[12:15].astype(np.float64))      # effector position (scaled)
        self.H.append(obs[15:17].astype(np.float64))      # effector yaw [cos, sin]
        self.G.append(obs[17:18].astype(np.float64))      # gripper opening (scaled)
        self.C.append(obs[18:19].astype(np.float64))      # gripper contact

        # Per-cube state (9D each): pos (3D) + ori (4D quat + 2D cos/sin yaw)
        for i, (p, o) in enumerate(zip(_POS_VARS, _ORI_VARS)):
            base_idx = 19 + 9 * i
            getattr(self, p).append(obs[base_idx:base_idx + 3].astype(np.float64))
            getattr(self, o).append(obs[base_idx + 3:base_idx + 9].astype(np.float64))

        self.X.append(action)

        reward = self._reward()
        self._Y.append(reward)

        hiddens = {}
        for var in _all_var_names():
            if var == 'W':
                continue
            if var in self.hidden_dims:
                hiddens[var] = self._get_var(var)

        info = {'Y': self._Y, 'env_obs': obs, 'env_info': {}, 'hidden_obs': hiddens, 'natural_action': action}

        self._t += 1
        return self.observation(), reward, terminated, truncated, info


class CubePCH(PCH):
    """Pearl Causal Hierarchy wrapper for the cube-quadruple environment."""

    def __init__(
        self,
        env_id: str = 'cube-quadruple-play-singletask-task2-v0',
        num_steps: int = 1000,
        expert_mode: bool = False,
        custom_hidden: Optional[Set[str]] = None,
        seed: Optional[int] = None,
    ):
        self.env = CubeSCM(
            env_id=env_id,
            num_steps=num_steps,
            expert_mode=expert_mode,
            custom_hidden=custom_hidden,
            seed=seed,
        )
        self.expert = CubeExpert(
            env_id=env_id,
            num_steps=num_steps,
            expert_mode=True,
            goal=self.env._goal,
            seed=seed,
        )
        super().__init__()

        self.last_actor_is_expert = True

    def _collect_env_vars(self) -> dict:
        """Collect all current variable histories from the SCM env."""
        d = {
            'Q': self.env.Q, 'V': self.env.V,
            'E': self.env.E, 'H': self.env.H,
            'G': self.env.G, 'C': self.env.C,
            'W': self.env.W,
        }
        for p, o in zip(_POS_VARS, _ORI_VARS):
            d[p] = getattr(self.env, p)
            d[o] = getattr(self.env, o)
        return d

    def see(self, behavioral_policy=None, show_reward=True) -> Tuple[Any, Any, float, bool, bool, Dict[str, Any]]:
        """Observational step."""
        if behavioral_policy is not None:
            env_vars = self._collect_env_vars()
            action = behavioral_policy(**env_vars)
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
        env_vars = self._collect_env_vars()
        intuition = self.env.action(**env_vars)
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
