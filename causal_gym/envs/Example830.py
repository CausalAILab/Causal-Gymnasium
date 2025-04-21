import math
import numpy as np

from causal_gym import SCM, PCH
from causal_gym.core import PolicyType, ActType, ObsType

# Example 8.30: Do-calculus Learning - Fixed Implementation
# We'll implement the causal identification approach for the model in Figure 8.14

# Create a custom SCM for the model in Figure 8.14
class Example830SCM(SCM):
    """SCM for the causal diagram in Figure 8.14(a) from Example 8.30"""
    
    def __init__(self, max_step=1):
        self.s1 = 0
        self.x1 = 0
        self.x2 = 0
        self.y = 0
        self.num_step = 0
        self._max_step = max_step
    
    def reset(self, *, seed: int = None, options: dict = None) -> tuple[ObsType, dict]:
        self.rng = np.random.default_rng(seed)
        # According to the textbook, P(S1=0) = 0.9, P(S1=1) = 0.1
        self.s1 = self.rng.choice([0, 1], p=[0.9, 0.1])
        self.num_step = 0
        return self.s1, {}
    
    def sample_u(self):
        """Sample exogeneous variables"""
        # For simplicity, return random values that influence actions
        u1 = self.rng.random()
        u2 = self.rng.random()
        return u1, u2
    
    def action(self, s1, u1, u2):
        """Behavioral policy for actions X1 and X2"""
        # X1 depends on S1 and U1
        x1 = 1 if u1 < 0.5 else 0  # Random action for X1
        
        # X2 depends on S1 and U2
        x2 = 1 if u2 < 0.5 else 0  # Random action for X2
        
        return x1, x2
    
    def compute_reward(self, s1, x2):
        """
        Compute reward Y
        According to the textbook example, for the policy X2 ← ¬S1:
        - When S1=0, X2=1: Y=1 (90% of the time)
        - When S1=1, X2=0: Y=1 (10% of the time)
        This gives an expected reward of 0.9 for the policy
        """
        # Following the example, Y = 1 if X2 = ¬S1
        return int(x2 == (1 - s1))
    
    def step(self, action):
        """Take a step with the provided action (X2 value)"""
        self.num_step += 1
        
        # Set X2 to the provided action
        self.x2 = action
        
        # Compute reward
        self.y = self.compute_reward(self.s1, self.x2)
        
        return self.s1, self.y, self.num_step >= self._max_step, False, {}

class Example830PCH(PCH):
    """PCH for Example 8.30"""
    
    def __init__(self, max_step=1):
        self.env = Example830SCM(max_step=max_step)
        super().__init__()
    
    def see(self):
        """Passively observe the environment"""
        u1, u2 = self.env.sample_u()
        
        # Get actions based on behavioral policy
        x1, x2 = self.env.action(self.env.s1, u1, u2)
        
        # Record state, action, and compute reward
        self.env.x1 = x1
        self.env.x2 = x2
        self.env.y = self.env.compute_reward(self.env.s1, self.env.x2)
        
        return x2, self.env.s1, self.env.y, self.env.num_step >= self.env._max_step, False, {}
    
    def do(self, action):
        """Intervene on X2"""
        return self.env.step(action)