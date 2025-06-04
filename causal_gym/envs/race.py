import numpy as np
import math
from typing import Any, Tuple, Dict, List
import pygame

from causal_gym import SCM, PCH
from causal_gym.core import ObsType, ActType
import gymnasium as gym
from gymnasium import spaces

from highway_env.envs.common.action import ContinuousAction

class RaceSCM(SCM):
    ''' Causal environment for the sequential racetrack driving scenario.'''

    def __init__(self, num_steps: int, config: Dict[str, Any] = None, seed: int = None, render_mode = 'human', perception = 'truth', u_prob: float = 0.2, d_prob: float = 0.5, w_probs: List[float] = [0.5, 0.4, 0.3, 0.2]):
        super().__init__()

        self.rng = np.random.default_rng(seed)
        self.render_mode = render_mode

        self.num_steps = num_steps
        self.t = 0 # current timestep

        # configurations for environment
        self.config = config or {}
        self.config.update({'other_vehicles': 0}) # focus on staying on the track for now
        
        self.perception = perception # which GUI elements to show

        # internal env
        self._env = gym.make('racetrack-v0', config=self.config, render_mode=render_mode)
        self._env.reset(seed=seed)

        # show GUI for dashboard warning, drunkenness, and fog
        if self.render_mode == 'human':
            self._env.render()
            viewer = self._env.unwrapped.viewer

            orig_display = viewer.display

            def display_with_overlay():
                orig_display()

                screen = pygame.display.get_surface()

                ego = viewer.env.vehicle
                if self.perception != 'imitator'  and getattr(self, '_D') and self._D == 1:
                    font = pygame.font.Font(None, 36)
                    text_surface = font.render("Driver is impaired", True, (255, 255, 255))
                    screen.blit(text_surface, (10, 10))

                if getattr(self, '_U', []) and self._U[-1] == 1:
                    if self.perception == 'truth':
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
                    half_len = ego.LENGTH / 2.0
                    dx = math.cos(ego.heading) * half_len
                    dy = math.sin(ego.heading) * half_len
                    
                    x, y = viewer.sim_surface.pos2pix(ego.position[0] + dx, ego.position[1] + dy)
                    r = 4.5

                    rect = pygame.Rect(int(x - r), int(y - r), int(2*r), int(2*r))
                    pygame.draw.ellipse(screen, (255, 100, 0), rect)

                pygame.display.flip()

            viewer.display = display_with_overlay

        # covariates
        self.W = [] # dashboard warning
        self.C = [] # lane centering
        self.H = [] # heading error

        # latents
        self._D = None # driver is drunk

        # confounder
        self._U = [] # fog

        # action
        self.X = [] # steering from -1.0 (left) to 1.0 (right)

        # reward
        self._Y = [] # rewards staying on the road and in the center of the lane

        self.action_space = self._env.action_space # Box(-1.0, 1.0, (1,), float32)
        self.observation_space = spaces.Dict({
            'X': spaces.Sequence(spaces.Box(-1.0, 1.0, (1,))),
            'W': spaces.Sequence(spaces.Discrete(2)),
            'C': spaces.Sequence(spaces.Box(-1.0, 1.0, (1,))),
            'H': spaces.Sequence(spaces.Box(-1.0, 1.0, (1,))),
        })

        self.u_prob = u_prob
        self.d_prob = d_prob
        self.w_probs = w_probs

    def sample_U(self) -> int:
        # 0 = clear weather, 1 = fog
        return self.rng.choice(2, p=[1 - self.u_prob, self.u_prob])

    def sample_D(self) -> int:
        # 0 = sober, 1 = drunk
        return self.rng.choice(2, p=[1 - self.d_prob, self.d_prob])

    def calc_W(self, D: int, U: int) -> int:
        # 0 = no warning, 1 = warning on
        if D == 1:
            if U == 1:
                return self.rng.choice(2, p=[1 - self.w_probs[0], self.w_probs[0]]) # drunk and fog

            return self.rng.choice(2, p=[1 - self.w_probs[1], self.w_probs[1]]) # only drunk
        else:
            if U == 1:
                return self.rng.choice(2, p=[1 - self.w_probs[2], self.w_probs[2]]) # only fog

            return self.rng.choice(2, p=[1 - self.w_probs[3], self.w_probs[3]]) # no reason for warning

    def calc_C(self) -> float:
        # influence from previous X is implicit
        _, lateral = self._env.unwrapped.vehicle.lane.local_coordinates(self._env.unwrapped.vehicle.position)
        magnitude = 1 / (1 + self._env.unwrapped.config["lane_centering_cost"] * lateral**2)
        return magnitude if lateral >= 0.0 else -magnitude

    def calc_H(self) -> float:
        # how far the car's heading is from what it should be based on the track
        # influence from previous X and H is implicit
        ego  = self._env.unwrapped.vehicle
        lane = ego.lane

        lon, lat = lane.local_coordinates(ego.position)

        # ideal heading of the lane at s
        try:
            ideal_heading = lane.heading_at(lon)
        except AttributeError:
            # fallback: finite‐difference approximation
            p1 = lane.position(lon)
            p2 = lane.position(lon + 0.1)
            ideal_heading = math.atan2(p2[1] - p1[1], p2[0] - p1[0])

        # signed difference, wrapped to [-pi, pi]
        err = ego.heading - ideal_heading
        return math.atan2(math.sin(err), math.cos(err))

    def reset(self, *, seed: int = None) -> Tuple[Any, dict]:
        self.rng = np.random.default_rng(seed)

        env_obs, env_info = self._env.reset()

        self.t = 0

        self._D = self.sample_D()
        self._U = [self.sample_U()]
        self.W = [self.calc_W(self._D, self._U[self.t])]
        self.C = [self.calc_C()]
        self.H = [self.calc_H()]

        self.X = []
        self._Y = []

        obs = self.observation()
        info = {'D': self._D, 'U': self._U, 'Y': self._Y, 'env_obs': env_obs, 'env_info': env_info}
        return obs, info

    def action(self, D: int, W: List[int], C: List[float], H: List[float]) -> ActType:
        # note how W is not used

        # if centered enough, follow road; otherwise correct toward center
        center_threshold = 0.9
        steer = -float(H[-1]) if abs(C[-1]) >= center_threshold else -(1 - abs(C[-1])) * np.sign(C[-1])

        # if drunk, introduce noise
        if D == 1:
            steer += self.rng.normal(loc=0.0, scale=0.1)

        # expert drives more carefully if drunk to minimize effect of drinking
        max_steer = 1.0 if D == 0 else 0.6

        # match env action format
        steer = float(np.clip(steer, -max_steer, max_steer))
        return steer

    def observation(self):
        return {'X': self.X, 'W': self.W, 'C': self.C, 'H': self.H}

    def _reward(self, X: float, C: float, H: float, U: int) -> float:
        center_score = max(0.0, abs(C)) * 0.5 if U == 1 else 1.0
        steer_penalty = abs(X) * (0.1 if U == 1 else 0.2)
        heading_penalty = abs(H) * (0.05 if U == 0 else 0.1)
        return center_score - steer_penalty - heading_penalty

    def step(self, action: Any, show_reward = False) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        self.X.append(action)

        X_t = self.X[self.t]
        C_t = self.C[self.t]
        H_t = self.H[self.t]
        U_t = self._U[self.t]

        env_obs, _, terminated, _, env_info = self._env.step(np.array([action, 0], dtype=np.float32))

        Y_t = self._reward(X_t, C_t, H_t, U_t)
        if len(self._Y) == 0:
            self._Y = [Y_t]
        else:
            self._Y.append(self._Y[-1] + Y_t) # accumulate

        self.t += 1

        self._U.append(self.sample_U())
        self.W.append(self.calc_W(self._D, self._U[self.t]))
        self.C.append(self.calc_C())
        self.H.append(self.calc_H())

        obs = self.observation()
        info = {'D': self._D, 'U': self._U, 'Y': self._Y, 'env_obs': env_obs, 'env_info': env_info}

        return obs, Y_t if show_reward else None, terminated, self.t >= self.num_steps, info

    def render(self) -> ObsType:
        # still need this for one-step-at-a-time simulation
        frame = self._env.render()
        viewer = self._env.unwrapped.viewer
        if self._env.render_mode != 'rgb_array':
            screen = pygame.display.get_surface()
        else:
            screen = pygame.Surface((frame.shape[1], frame.shape[0]))
            image_surf = pygame.surfarray.make_surface(frame.transpose(1,0,2))
            screen.blit(image_surf, (0, 0))
        ego = viewer.env.vehicle
        if self.perception != 'imitator'  and getattr(self, '_D') and self._D == 1:
            font = pygame.font.Font(None, 36)
            text_surface = font.render("Driver is impaired", True, (255, 255, 255))
            screen.blit(text_surface, (10, 10))

        if getattr(self, '_U', []) and self._U[-1] == 1:
            if self.perception == 'truth':
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
            half_len = ego.LENGTH / 2.0
            dx = math.cos(ego.heading) * half_len
            dy = math.sin(ego.heading) * half_len
            
            x, y = viewer.sim_surface.pos2pix(ego.position[0] + dx, ego.position[1] + dy)
            r = 4.5

            rect = pygame.Rect(int(x - r), int(y - r), int(2*r), int(2*r))
            pygame.draw.ellipse(screen, (255, 100, 0), rect)

        if self._env.render_mode != 'rgb_array':
            pygame.display.flip()
        else:
            pixels = pygame.surfarray.array3d(screen)  # shape: (width, height, 3)
            frame = np.transpose(pixels, (1, 0, 2))   # reshape to (height, width, 3)

        return frame
    
    def close(self):
        self._env.close()

    @property
    def get_graph(self) -> Tuple[Dict[int, str], list[list[int]], list[list[int]]]:
        variables = ['X', 'W', 'C', 'H']
        n = (self.num_steps) * len(variables) + 2 # incl. D and Y

        nodes = {0: 'D0'} # ensure D comes first in temporal ordering
        i = 1 # offset for D
        for t in range(self.num_steps):
            for v in variables:
                nodes[i] = f'{v}{t}'
                i += 1

        nodes[i] = f'Y{self.num_steps}' # ensures Y comes last in temporal ordering

        base_graph = [[0]*n for _ in range(n)]
        conf_graph = [[0]*n for _ in range(n)]

        # intra-timestep edges
        for t in range(self.num_steps):
            base = t * len(variables) + 1 # offset for D
            d = 0
            x, w, c, h = base, base + 1, base + 2, base + 3
            y = n - 1

            base_graph[d][x] = 1
            base_graph[d][w] = 1
            base_graph[c][x] = 1
            base_graph[c][y] = 1
            base_graph[h][x] = 1
            base_graph[h][y] = 1
            base_graph[x][y] = 1
            
            conf_graph[w][y] = 1
            conf_graph[y][w] = 1

        # inter-timstep edges
        for t in range(self.num_steps - 1):
            base = t * len(variables) + 1
            base_next = (t + 1) * len(variables) + 1

            d = 0
            x1, w1, c1, h1 = base, base + 1, base + 2, base + 3
            x2, w2, c2, h2 = base_next, base_next + 1, base_next + 2, base_next + 3
            y = n - 1

            base_graph[x1][c2] = 1
            base_graph[h1][h2] = 1
            base_graph[x1][h2] = 1

        return nodes, base_graph, conf_graph

    @property
    def observed_unobserved_vars(self) -> Tuple[list[str], list[str]]:
        return ['X', 'W', 'C', 'H'], ['D', 'U', 'Y']

class RacePCH(PCH):
    '''
    PCH wrapper for the RaceSCM env.

    perception:
        1. 'truth' = show everything
        2. 'expert' = hide fog
        3. 'imitator' = hide fog and indicator
    '''

    def __init__(self, num_steps: int = 3, config: Dict[str, Any] = None, seed: int = None, render_mode = 'human', perception = 'truth', u_prob: float = 0.2, d_prob: float = 0.5, w_probs: List[float] = [0.5, 0.4, 0.3, 0.2]):
        # initialize underlying SCM
        self.env: RaceSCM = RaceSCM(num_steps=num_steps, config=config, seed=seed, render_mode=render_mode, perception=perception, u_prob=u_prob, d_prob=d_prob, w_probs=w_probs)
        super().__init__()

    def see(self, behavioral_policy=None, show_reward = False) -> Tuple[Any, Any, float, bool, bool, Dict[str, Any]]:
        D = self.env._D
        W = self.env.W
        C = self.env.C
        H = self.env.H

        if behavioral_policy is not None:
            action = behavioral_policy(D, W, C, H)
        else:
            action = self.env.action(D, W, C, H)

        obs, reward, terminated, truncated, info = self.env.step(action, show_reward=show_reward)
        return action, obs, reward, terminated, truncated, info

    def do(self, action: Any, show_reward = False) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        return self.env.step(action, show_reward=show_reward)
    
    # Counterfactual policy intervention
    def ctf_do(self, ctf_policy):
        intuition = self.env.action()
        action = ctf_policy(self.env.observation(), intuition)
        return self.env.step(action)

    def reset(self, *, seed: int = None) -> Tuple[Any, dict]:
        return self.env.reset(seed=seed)

    def render(self) -> Any:
        return self.env.render()
    
    def close(self) -> None:
        self.env.close()
    
    @property
    def get_graph(self) -> Tuple[Dict[int, str], list[list[int]], list[list[int]]]:
        return self.env.get_graph