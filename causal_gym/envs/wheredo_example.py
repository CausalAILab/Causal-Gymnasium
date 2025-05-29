import math
import numpy as np

from causal_gym import SCM, PCH
from causal_gym.core.causal_graph import CausalGraph
from causal_gym.core import PolicyType, ActType, ObsType

class ExampleSCM_9_5(SCM):
    def __init__(self, max_step=1):
        self._policy = self.F
        self._u1 = lambda: self.rng.choice(2, p=[.5, .5])
        self._u2 = lambda: self.rng.choice(2, p=[.5, .5])
        self.num_step = 0
        self._max_step = max_step

    def reset(self, *, seed: int = None, options: dict = None) -> tuple[dict]:
        self.rng = np.random.default_rng(seed)
        self.num_step = 0
        return {}
    
    def sample_u(self):
        """
        Sample exogeneous variables
        """
        return self._u1(), self._u2()
   
    def F(self, u1: int, u2: int):
        x1 = u1
        x2 = x1 ^ u2
        return x1, x2
    
    def action(self, u1: int, u2: int, policy = None):
        if not policy:
            policy = self._policy
        return policy(u1, u2)
    
    def step(self, x1: int, x2: int, u1: int, u2: int):
        self.num_step += 1
        self.y = x2 ^ u2
        return self.y, False, self.num_step > self._max_step, {}
    
    def get_graph(self):
        X1, X2, Y = 'X1', 'X2', 'Y'
        cdag = CausalGraph({X1, X2, Y}, [(X1, X2), (X2, Y)], [(X2, Y, 'U2')])
        return cdag

class ExamplePCH_9_5(PCH):
    def __init__(self, max_step=1):
        self.env = ExampleSCM_9_5(max_step=max_step)
        super().__init__()
    
    def see(self):
        u1, u2 = self.env.sample_u()
        x1, x2 = self.env.action(u1, u2)
        y, terminated, truncated, info = self.env.step(x1, x2, u1, u2)
        return x1, x2, y, terminated, truncated, info

    def do(self, policy):
        u1, u2 = self.env.sample_u()
        x1, x2 = self.env.action(u1, u2, policy)
        return self.env.step(x1, x2, u1, u2)