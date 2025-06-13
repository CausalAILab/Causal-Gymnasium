import numpy as np

from causal_gym import SCM, PCH
from causal_gym.core import PolicyType, ActType, ObsType

class DTRExampleSCM(SCM):
    """
    A confounded DTR from CRL book Chapter 8 Example 8.1.
    By default, the behavioral policy observes both the state at that timestep
    and the confounder between the actions and reward.

    Actions, state, and reward variables are binary, while the confounders are continuous.
    The two coefficients alpha1 and alpha2 are also continuous.
    
    This is a two-stage DTR, where the episode ends after the second stage.
    
    """
    def __init__(self, alpha1=0.3, alpha2=0.3):
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.s1, self.x1, self.s2, self.x2 = None, None, None, None
        # behavioral policy 
        self._policy1 = lambda s1, u, u1: int(3*s1 + alpha1*u + u1 > 0)
        self._policy2 = lambda s2, u, u2: int(3*s2 + alpha2*u + u2 > 0)
        self.num_step = 0
        self._max_step = 2
        
    def reset(self, *, seed: int = None, options: dict = None) -> tuple[ObsType, dict]:
        self.rng = np.random.default_rng(seed)
        self._u = self.rng.uniform(0, 1)
        self._u1 = self.rng.logistic(0, 1)
        self._u2 = self.rng.logistic(0, 1)
        self._u3 = self.rng.logistic(0, 1)
        self._u4 = self.rng.logistic(0, 1)
        self.num_step = 0
        self.s1 = int(self._u3 > 0)
        return self.s1, {}
    
    def action(self, s):
        if self.num_step == 0:
            return self._policy1(s, self._u, self._u1)
        else:
            return self._policy2(s, self._u, self._u2)
        
    def step(self, x):
        if self.num_step == 0:
            self.x1 = x
            self.s2 = int(0.1 + 0.1*self.s1 + 0.1*self.x1 + self._u4 > 0)
            self.num_step += 1
            return self.s2, 0, False, False, {}
        else:
            self.x2 = x
            y = int(3*self._u - 3*self.s1 - 3*self.x1 - 3*self.s1*self.x1 \
                + 3*self.x2 - 3*self.s2*self.x2 + 3*self.x1*self.x2 > 0)
            self.num_step += 1
            return None, y, True, False, {}
    
class DTRExamplePCH(PCH):
    """
    PCH for the DTR Example defined above.
    """
    def __init__(self, alpha1=0.3, alpha2=0.3):
        self.env = DTRExampleSCM(alpha1=alpha1, alpha2=alpha2)
        super().__init__()

    def see(self):
        if self.env.num_step == 0:
            x = self.env.action(self.env.s1)
        else:
            x = self.env.action(self.env.s2)
        s, y, terminated, truncated, info = self.env.step(x)
        return x, s, y, terminated, truncated, info 

    def do(self, action):
        return self.env.step(action)