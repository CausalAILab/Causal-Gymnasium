import numpy as np
from typing import Any, Tuple, Dict, List
import pygame

from causal_gym import SCM, PCH
from causal_gym.core import ObsType, ActType
import gymnasium as gym
from gymnasium import spaces

from highway_env.envs.common.action import DiscreteMetaAction

DANGER_DISTANCE = 20.0 # m
MERGE_DANGER_DISTANCE = 10.0 # m

class HighwaySCM(SCM):
    ''' Causal environment for the single step highway driving scenario.'''

    def __init__(self, num_steps: int, config: Dict[str, Any] = None, seed: int = None, render_mode = 'human'):
        super().__init__()

        self.rng = np.random.default_rng(seed)
        self.render_mode = render_mode

        self.num_steps = num_steps
        self.t = 0 # current timestep

        # configurations for highway environment
        self.config = config or {}

        # internal env
        self._env = gym.make('highway-v0', config=self.config, render_mode=render_mode)
        self._env.reset(seed=seed)

        # show GUI for tail light indicator and fog
        if self.render_mode == 'human':
            self._env.render()
            viewer = self._env.unwrapped.viewer

            orig_display = viewer.display

            def display_with_overlay():
                orig_display()

                screen = pygame.display.get_surface()

                ego = viewer.env.vehicle
                front = viewer.env.road.neighbour_vehicles(ego)[0]
                if front is not None and getattr(self, '_I', []) and self._I[-1] == 1:
                    x, y = viewer.sim_surface.pos2pix(front.position[0], front.position[1])
                    r = 4.5
                    rect = pygame.Rect(int(x - 3*r), int(y - r), int(r), int(2*r))
                    pygame.draw.rect(screen, (255, 100, 0), rect)

                if getattr(self, '_U', []) and self._U[-1] == 1:
                    w, h = screen.get_size()
                    alpha = self.rng.normal(loc=80, scale=30, size=(h, w))
                    alpha = np.clip(alpha, 0, 255).astype(np.uint8)

                    fog = np.empty((h, w, 4), dtype=np.uint8)
                    fog[..., :3] = 200
                    fog[...,  3] = alpha

                    buf = fog.tobytes()
                    fog_surf = pygame.image.frombuffer(buf, (w, h), 'RGBA').convert_alpha()
                    screen.blit(fog_surf, (0, 0))

                if getattr(self, 'W', []) and self.W[-1] == 1:
                    x, y = viewer.sim_surface.pos2pix(ego.position[0], ego.position[1])
                    r = 4.5
                    rect = pygame.Rect(int(x + 2*r), int(y - r), int(r), int(2*r))
                    pygame.draw.rect(screen, (255, 200, 0), rect)

                # keep Uwy hidden for now

                pygame.display.flip()

            viewer.display = display_with_overlay

        # set up behavioral policy
        self._meta_actions: DiscreteMetaAction = self._env.unwrapped.action_type
        self._actions_reverse = {v: k for k, v in self._meta_actions.actions.items()}
        self.num_lanes = self._env.unwrapped.config['lanes_count']

        # covariates
        self.D = [] # distance to front car
        self.L = [] # current lane index
        self.A = [] # left lane open
        self.B = [] # right lane open
        self.W = [] # dashboard warning

        # latents
        self._I = [] # front car's tail light indicator

        # confounder
        self._U = [] # fog

        # action
        self.X = [] # lane left, idle, lane right, faster, slower

        # reward
        self._Y = [] # rewards driving fast without crashing

        self.action_space = self._env.action_space # spaces.Discrete(5)
        self.observation_space = spaces.Dict({
            'X': spaces.Sequence(spaces.Discrete(self.action_space.n)),
            'D': spaces.Sequence(spaces.Box(0.0, np.inf, shape=(), dtype=np.float32)),
            'L': spaces.Sequence(spaces.Discrete(self.num_lanes)),
            'A': spaces.Sequence(spaces.Discrete(2)),
            'B': spaces.Sequence(spaces.Discrete(2)),
            'W': spaces.Sequence(spaces.Discrete(2))
        })

    def calc_D(self) -> float:
        # influence from previous D and X is intrinsic

        ego = self._env.unwrapped.vehicle
        front_vehicle = self._env.unwrapped.road.neighbour_vehicles(self._env.unwrapped.vehicle)[0]

        if front_vehicle is None:
            return np.inf

        return front_vehicle.position[0] - ego.position[0] - front_vehicle.LENGTH / 2 - ego.LENGTH / 2

    def calc_L(self) -> int:
        # influence from previous L and X is intrinsic

        true_lane = self._env.unwrapped.vehicle.lane_index[2]

        # lane readings aren't always correct
        conf_lane = self.rng.choice([true_lane - 1, true_lane, true_lane + 1], p=[0.1, 0.8, 0.1])
        if conf_lane < 0  or conf_lane >= self.num_lanes:
            return true_lane

        return conf_lane

    def calc_I(self, D: float) -> int:
        front_vehicle = self._env.unwrapped.road.neighbour_vehicles(self._env.unwrapped.vehicle)[0]
        if front_vehicle is None:
            return 0

        acc = front_vehicle.acceleration(ego_vehicle=front_vehicle)
        is_braking = acc < -1.0 # m/s^2
        too_close = D < DANGER_DISTANCE

        if is_braking and too_close:
            return self.rng.choice(2, p=[0.1, 0.9])

        return self.rng.choice(2, p=[0.9, 0.1])

    def calc_A(self, L: int) -> int:
        if L == 0:
            return 0

        ego = self._env.unwrapped.vehicle
        road = self._env.unwrapped.road
        left_lane = (ego.lane_index[0], ego.lane_index[1], L - 1)
        left_front, left_back = road.neighbour_vehicles(self._env.unwrapped.vehicle, lane_index=left_lane)

        if left_front is not None \
            and MERGE_DANGER_DISTANCE > left_front.position[0] - ego.position[0] \
                - left_front.LENGTH / 2 - ego.LENGTH / 2:
            return 0
        
        if left_back is not None \
            and MERGE_DANGER_DISTANCE / 2 > ego.position[0] - left_back.position[0] \
                - ego.LENGTH / 2 - left_back.LENGTH / 2:
            return 0

        return 1
    
    def calc_B(self, L: int) -> int:
        if L == self.num_lanes - 1:
            return 0

        ego = self._env.unwrapped.vehicle
        road = self._env.unwrapped.road
        right_lane = (ego.lane_index[0], ego.lane_index[1], L + 1)
        right_front, right_back = road.neighbour_vehicles(self._env.unwrapped.vehicle, lane_index=right_lane)

        if right_front is not None \
            and MERGE_DANGER_DISTANCE > right_front.position[0] - ego.position[0] \
                - right_front.LENGTH / 2 - ego.LENGTH / 2:
            return 0
        
        if right_back is not None \
            and MERGE_DANGER_DISTANCE / 2 > ego.position[0] - right_back.position[0] \
                - ego.LENGTH / 2 - right_back.LENGTH / 2:
            return 0

        return 1
    
    def calc_W(self, I: int, U: int) -> int:
        # 0 = no warning, 1 = warning on
        if I == 1:
            if U == 1:
                return self.rng.choice(2, p=[0.1, 0.9]) # brake check and fog

            return self.rng.choice(2, p=[0.4, 0.6]) # only brake check
        else:
            if U == 1:
                return self.rng.choice(2, p=[0.6, 0.4]) # only fog

            return self.rng.choice(2, p=[0.9, 0.1]) # no reason for warning

    def sample_U(self) -> int:
        # 0 = clear vision, 1 = foggy weather
        return self.rng.choice(2, p=[0.8, 0.2])

    def reset(self, *, seed: int = None) -> Tuple[Any, dict]:
        self.rng = np.random.default_rng(seed)

        env_obs, env_info = self._env.reset()

        self.t = 0

        self._U = [self.sample_U()]
        self.D = [self.calc_D()]
        self._I = [self.calc_I(self.D[self.t])]
        self.W = [self.calc_W(self._I[self.t], self._U[self.t])]
        self.L = [self.calc_L()]
        self.A = [self.calc_A(self.L[self.t])]
        self.B = [self.calc_B(self.L[self.t])]

        self.X = []
        self._Y = []

        obs = self.observation()
        info = {'I': self._I, 'U': self._U, 'Y': self._Y, 'env_obs': env_obs, 'env_info': env_info}
        return obs, info

    def action(self, X: List[int], D: List[float], L: List[int], I: List[int], A: List[int], B: List[int], W: List[int]) -> ActType:
        # note that W is a red herring and should not be used        

        # verify lane reading using history
        drive_carefully = False
        if len(X) >= 1 and len(L) >= 2:
            last_lane_reading = L[-2]

            if X[-1] == self._actions_reverse['LANE_RIGHT']:
                last_lane_reading += 1
            elif X[-1] == self._actions_reverse['LANE_LEFT']:
                last_lane_reading -= 1

            if last_lane_reading != L[-1]:
                # detected lane reading anomaly
                drive_carefully = True

        if I[-1] == 1 and D[-1] < DANGER_DISTANCE:
            return self._actions_reverse['SLOWER']

        if D[-1] < DANGER_DISTANCE:
            if A[-1]:
                return self._actions_reverse['LANE_LEFT']

            if B[-1]:
                return self._actions_reverse['LANE_RIGHT']

            return self._actions_reverse['SLOWER']

        return self._actions_reverse['FASTER' if not drive_carefully else 'IDLE']

    def observation(self):
        return {'X': self.X, 'D': self.D, 'L': self.L, 'A': self.A, 'B': self.B, 'W': self.W}
    
    def _reward(self, X: int, D: float, A: int, B: int, U: int) -> float:
        action = self._meta_actions.ACTIONS_ALL[X]

        # punish crashes
        if D < 1.0 or action == 'LANE_LEFT' and not A or action == 'LANE_RIGHT' and not B:
            return -10.0

        speed = self._env.unwrapped.vehicle.velocity[0]

        # punish accelerating while tailgating
        if D < DANGER_DISTANCE and action == 'FASTER':
            return -10.0 * (2.0 if U == 1 else 1.0) # punish more in fog
        
        # half reward if speeding up in fog
        if U == 1 and action == 'FASTER':
            return speed * 0.5
        
        # otherwise reward with current velocity
        return speed

    def step(self, action: Any, show_reward = False) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        self.X.append(action)

        U_t = self._U[self.t]
        D_t = self.D[self.t]
        X_t = self.X[self.t]
        A_t = self.A[self.t]
        B_t = self.B[self.t]

        env_obs, _, terminated, _, env_info = self._env.step(action)

        Y_t = self._reward(X_t, D_t, A_t, B_t, U_t)
        if len(self._Y) == 0:
            self._Y = [Y_t]
        else:
            self._Y.append(self._Y[-1] + Y_t) # accumulate

        self.t += 1

        self._U.append(self.sample_U())
        self.D.append(self.calc_D())
        self.L.append(self.calc_L())
        self._I.append(self.calc_I(self.D[self.t]))
        self.A.append(self.calc_A(self.L[self.t]))
        self.B.append(self.calc_B(self.L[self.t]))
        self.W.append(self.calc_W(self._I[self.t], self._U[self.t]))

        obs = self.observation()
        info = {'I': self._I, 'U': self._U, 'Y': self._Y, 'env_obs': env_obs, 'env_info': env_info}

        return obs, Y_t if show_reward else None, terminated, self.t >= self.num_steps, info

    def render(self) -> ObsType:
        # still need this for one-step-at-a-time simulation
        frame = self._env.render()

        if self._env.render_mode != 'rgb_array':
            viewer = self._env.unwrapped.viewer
            screen = pygame.display.get_surface()

            ego = viewer.env.vehicle
            front = viewer.env.road.neighbour_vehicles(ego)[0]
            if front is not None and getattr(self, '_I', []) and self._I[-1] == 1:
                x, y = viewer.sim_surface.pos2pix(front.position[0], front.position[1])
                r = 4.5
                rect = pygame.Rect(int(x - 3*r), int(y - r), int(r), int(2*r))
                pygame.draw.rect(screen, (255, 100, 0), rect)

            if getattr(self, '_U', []) and self._U[-1] == 1:
                w, h = screen.get_size()
                alpha = self.rng.normal(loc=80, scale=30, size=(h, w))
                alpha = np.clip(alpha, 0, 255).astype(np.uint8)

                fog = np.empty((h, w, 4), dtype=np.uint8)
                fog[..., :3] = 200
                fog[...,  3] = alpha

                buf = fog.tobytes()
                fog_surf = pygame.image.frombuffer(buf, (w, h), 'RGBA').convert_alpha()
                screen.blit(fog_surf, (0, 0))

            if getattr(self, 'W', []) and self.W[-1] == 1:
                x, y = viewer.sim_surface.pos2pix(ego.position[0], ego.position[1])
                r = 4.5
                rect = pygame.Rect(int(x + 2*r), int(y - r), int(r), int(2*r))
                pygame.draw.rect(screen, (255, 200, 0), rect)

            # keep Uwy hidden for now

            pygame.display.flip()

        return frame

    @property
    def get_graph(self) -> Tuple[Dict[int, str], list[list[int]], list[list[int]]]:
        variables = ['D', 'L', 'I', 'A', 'B', 'W', 'X'] # U's are implicit, Y added at the end
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
            d, l, i, a, b, w, x = base, base + 1, base + 2, base + 3, base + 4, base + 5, base + 6
            y = n - 1

            base_graph[d][i] = 1 # close distance turns on indicator
            base_graph[d][x] = 1 # action chosen based on distance
            base_graph[d][y] = 1 # reward affected by distance
            base_graph[i][x] = 1 # expert uses indicator to avoid tailgating
            base_graph[x][y] = 1 # driving decisions affect reward function
            base_graph[l][x] = 1 # current lane restricts some actions e.g. veering offroad
            base_graph[l][a] = 1 # current lane used to check left lane
            base_graph[l][b] = 1 # current lane used to check right lane
            base_graph[a][x] = 1 # left lane availability restricts some actions
            base_graph[b][x] = 1 # right lane availability restricts some actions
            base_graph[a][y] = 1 # reward func checks if left lane is open
            base_graph[b][y] = 1 # reward func checks if right lane is open
            base_graph[i][w] = 1 # front car tail light triggers dashboard warning

            # fog confounds lane reading and reward func
            conf_graph[w][y] = 1
            conf_graph[y][w] = 1

        # inter-timstep edges
        for t in range(self.num_steps - 1):
            base = t * len(variables)
            base_next = (t + 1) * len(variables)

            d, l, i, a, b, w, x = base, base + 1, base + 2, base + 3, base + 4, base + 5, base + 6
            d2, l2, i2, a2, b2, w2, x2 = base_next, base_next + 1, base_next + 2, base_next + 3, base_next + 4, base_next + 5, base_next + 6

            base_graph[d][d2] = 1 # distance affects itself over time
            base_graph[x][d2] = 1 # acceleration/lane changes affect distance
            base_graph[l][l2] = 1 # lane dependent on previous lane
            base_graph[x][l2] = 1 # lange change action affects lane

            base_graph[x][x2] = 1 # expert uses previous action for lane inference
            base_graph[l][x2] = 1 # expert cross-checks lane readings with prev action

        return nodes, base_graph, conf_graph

    @property
    def observed_unobserved_vars(self) -> Tuple[list[str], list[str]]:
        return ['X', 'D', 'L', 'A', 'B', 'W'], ['I', 'U', 'Y']

class HighwayPCH(PCH):
    '''PCH wrapper for the HighwaySCM env'''

    def __init__(self, num_steps: int = 3, config: Dict[str, Any] = None, seed: int = None, render_mode = 'human'):
        # initialize underlying SCM
        self.env = HighwaySCM(num_steps=num_steps, config=config, seed=seed, render_mode=render_mode)
        super().__init__()

    def see(self, behavioral_policy=None, show_reward = False) -> Tuple[Any, Any, float, bool, bool, Dict[str, Any]]:
        X = self.env.X
        D = self.env.D
        L = self.env.L
        _I = self.env._I
        A = self.env.A
        B = self.env.B
        W = self.env.W

        if behavioral_policy is not None:
            action = behavioral_policy(X, D, L, _I, A, B, W)
        else:
            action = self.env.action(X, D, L, _I, A, B, W)

        obs, reward, terminated, truncated, info = self.env.step(action, show_reward=show_reward)
        return action, obs, reward, terminated, truncated, info

    def do(self, action: Any, show_reward = False) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        return self.env.step(action, show_reward=show_reward)

    def reset(self, *, seed: int = None) -> Tuple[Any, dict]:
        return self.env.reset(seed=seed)

    def render(self) -> Any:
        return self.env.render()