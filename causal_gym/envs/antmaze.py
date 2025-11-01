import ogbench
import numpy as np
from numpy.typing import NDArray

from typing import Dict, Optional, List, Tuple, Any
from causal_gym import SCM, PCH
from causal_gym.core import ActType

class AntMazeSCM(SCM):
    def __init__(self, env_id: str = 'antmaze-large-navigate-v0', num_steps: int = 1000, seed: Optional[int] = None):
        super().__init__()

        self.rng = np.random.default_rng(seed)
        self.num_steps = num_steps
        self._t = 0

        self._env = ogbench.make_env_and_datasets(env_id, env_only=True, max_episode_steps = num_steps)
        self._env.reset(seed=seed)

        self.P = [] # position, 3-dimensional vector of x,y,z in terms of goal
        self.O = [] # torso orientation, 4-dimensional quaternion x,y,z,w
        self.A = [] # joint angles, 8-dimensional vector
        self.L = [] # torso linear velocity, 3-dimensional vector of x,y,z
        self.T = [] # torso angular velocity, 3-dimensional vector of x,y,z
        self.J = [] # joint angular velocities, 8-dimensional vector
        self.X = [] # action, 8-dimensional vector of torques
        self._Y = [] # sparse reward

        self.action_space = self._env.action_space # Box(-1.0, 1.0, (8,), float32)
        self.observation_space = self._env.observation_space
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

    def observation(self) -> Dict[str, List[List[float]]]:
        return {'P': self.P,
                'O': self.O,
                'A': self.A,
                'L': self.L,
                'T': self.T,
                'J': self.J,
                'X': self.X}

    def reset(self, seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed)
        env_obs, env_info = self._env.reset()

        self._t = 0
        self.P = [self._P()]
        self.O = [self._O()]
        self.A = [self._A()]
        self.L = [self._L()]
        self.T = [self._T()]
        self.J = [self._J()]
        self.X = []
        self._Y = []

        obs = self.observation()
        info = {'Y': self._Y, 'env_obs': env_obs, 'env_info': env_info}
        return obs, info

    def action(self, P: List[NDArray[np.float64]], O: List[NDArray[np.float64]], A: List[NDArray[np.float64]], L: List[NDArray[np.float64]], T: List[NDArray[np.float64]], J: List[NDArray[np.float64]]) -> ActType:
        # placeholder behavior policy
        return self.action_space.sample()

    def step(self, action: Any, show_reward: bool = False) -> Tuple[dict, float, bool, bool, dict]:
        self.X.append(action)

        env_obs, reward, terminated, truncated, env_info = self._env.step(action)
        self._Y.append(reward)

        self._t += 1
        self.P.append(self._P())
        self.O.append(self._O())
        self.A.append(self._A())
        self.L.append(self._L())
        self.T.append(self._T())
        self.J.append(self._J())

        obs = self.observation()
        info = {'Y': self._Y, 'env_obs': env_obs, 'env_info': env_info}
        return obs, reward if show_reward else None, terminated, truncated, info

    def render(self):
        return self._env.render()

    def close(self):
        self._env.close()

    @property
    def get_graph(self) -> Tuple[Dict[int, str], list[list[int]], list[list[int]]]:
        variables = ['P', 'O', 'A', 'L', 'T', 'J', 'X']
        n = (self.num_steps) * len(variables) + 1

        nodes = {}
        i = 0
        for t in range(self.num_steps):
            for v in variables:
                nodes[i] = f'{v}{t}'
                i += 1

        nodes[i] = f'Y{self.num_steps}' # ensures Y comes last in temporal ordering

        base_graph = [[0]*n for _ in range(n)]
        conf_graph = [[0]*n for _ in range(n)]

        # intra-timestep edges
        for t in range(self.num_steps):
            base = t * len(variables)
            p, o, a, l, t, j, x = base, base + 1, base + 2, base + 3, base + 4, base + 5, base + 6
            y = n - 1

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

            # TODO conf_graph

        # inter-timstep edges
        for t in range(self.num_steps - 1):
            base = t * len(variables)
            base_next = (t + 1) * len(variables)

            p, o, a, l, t, j, x = base, base + 1, base + 2, base + 3, base + 4, base + 5, base + 6
            p2, o2, a2, l2, t2, j2, x2 = base_next, base_next + 1, base_next + 2, base_next + 3, base_next + 4, base_next + 5, base_next + 6

            base_graph[x][j2] = 1 # torque impacts joint angles

            # state persistence
            base_graph[p][p2] = 1
            base_graph[o][o2] = 1
            base_graph[a][a2] = 1
            base_graph[l][l2] = 1
            base_graph[t][t2] = 1
            base_graph[j][j2] = 1

        return nodes, base_graph, conf_graph

    @property
    def observed_unobserved_vars(self) -> Tuple[list[str], list[str]]:
        return ['P', 'O', 'A', 'L', 'T', 'J', 'X'], ['Y']

class AntMazePCH(PCH):
    def __init__(self, env_id: str = 'antmaze-large-navigate-v0', num_steps: int = 1000, seed: Optional[int] = None):
        # initialize underlying SCM
        self.env = AntMazeSCM(num_steps=num_steps, seed=seed)
        super().__init__()

    def see(self, behavioral_policy=None, show_reward = False) -> Tuple[Any, Any, float, bool, bool, Dict[str, Any]]:
        P = self.env.P
        O = self.env.O
        A = self.env.A
        L = self.env.L
        T = self.env.T
        J = self.env.J

        if behavioral_policy is not None:
            action = behavioral_policy(P, O, A, L, T, J)
        else:
            action = self.env.action(P, O, A, L, T, J)

        obs, reward, terminated, truncated, info = self.env.step(action, show_reward=show_reward)
        return action, obs, reward, terminated, truncated, info

    def do(self, action: Any, show_reward = False) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        return self.env.step(action, show_reward=show_reward)

    def reset(self, *, seed: int = None) -> Tuple[Any, dict]:
        return self.env.reset(seed=seed)

    def render(self) -> Any:
        return self.env.render()

    def close(self) -> None:
        self.env.close()

    @property
    def get_graph(self) -> Tuple[Dict[int, str], list[list[int]], list[list[int]]]:
        return self.env.get_graph