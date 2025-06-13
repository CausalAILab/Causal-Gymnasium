import numpy as np

from causal_gym import SCM, PCH
from causal_gym.core import PolicyType, ActType, ObsType

class MABExampleSCM(SCM):
    """
    A confounded MAB from CRL book Chapter 7 Example 7.1.
    By default, the behavioral policy observes the confounder between action and reward.

    Action and reward variables are binary, while the confounder U is continuous.
    The coefficient delta is also continuous.
    
    """
    def __init__(self, delta=0.1):
        self.delta = delta
        # behavioral policy
        self._policy = lambda u: int(u < 0.8)
        self._u = lambda: self.rng.uniform(0, 1)
        
    def reset(self, *, seed: int = None, options: dict = None) -> tuple[ObsType, dict]:
        self.rng = np.random.default_rng(seed)
        return None, {}
        
    def action(self, u: int) -> int:
        """
        Sample action from the behavioral policy
        """
        return self._policy(u)
    
    def step(self, x, u):
        y = int(u < 0.4 - self.delta * x)
        return None, y, False, True, {}
    
class MABExamplePCH(PCH):
    """
    PCH for the MAB example defined above.
    """
    def __init__(self, delta=0.1):
        self.env = MABExampleSCM(delta)
        super().__init__()
        
    def see(self):
        u = self.env._u()
        x = self.env.action(u)
        _, y, terminated, truncated, info = self.env.step(x, u)
        return x, _, y, terminated, truncated, info
    
    def do(self, action):
        u = self.env._u()
        return self.env.step(action, u)