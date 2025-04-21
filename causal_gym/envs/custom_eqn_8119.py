import math
import numpy as np

from causal_gym import SCM, PCH
from causal_gym.core import PolicyType, ActType, ObsType

class Custom8119SCM(SCM):
    """SCM for Example 8.24 from the textbook described in Equation 8.119 in textbook"""
    
    def __init__(self, max_step=30):
        self.s = 0
        self.x = 0
        self.num_step = 0
        self._max_step = max_step
    
    def reset(self, *, seed: int = None, options: dict = None) -> tuple[ObsType, dict]:
        self.rng = np.random.default_rng(seed)
        self.s = self.rng.integers(0, 2)  # S_1 ← U_1 (binary)
        self.num_step = 0
        return self.s, {}
    
    def sample_u(self):
        """Sample exogeneous variables U_1, U_2, U_3, U_4, U_5"""
        # U_1, U_2, U_3 with P(U_i = 0) = 0.9
        u1 = self.rng.choice([0, 1], p=[0.9, 0.1])
        u2 = self.rng.choice([0, 1], p=[0.9, 0.1]) 
        u3 = self.rng.choice([0, 1], p=[0.9, 0.1])
        
        # U_4, U_5 uniform over {0, 1}
        u4 = self.rng.integers(0, 2)
        u5 = self.rng.integers(0, 2)
        
        return u1, u2, u3, u4, u5
    
    def action(self, s, u1, u4):
        """Action function X_i ← U_i ⊕ U_i+3"""
        return u1 ^ u4  # Use XOR operation
    
    def step(self, x, u1, u2, u3):
        self.num_step += 1
        
        # Reward: Y ← S_i ⊕ X_i ⊕ U_i+2
        y = self.s ^ x ^ u3
        
        # Next state: S_i+1 ← S_i ⊕ X_i ⊕ U_i+1
        next_s = self.s ^ x ^ u2
        
        # Update current state
        self.s = next_s
        
        return self.s, y, False, self.num_step > self._max_step, {}
    

    
# Create the SCM described in Equation 8.119
class Custom8119PCH(PCH):
    """PCH for the custom MDP Example with specific structure from Example 8.24
    
    This implements the SCM from Equation 8.119 in the textbook where:
    S_i ← U_i
    X_i ← U_i ⊕ U_i+3
    S_i+1 ← S_i ⊕ X_i ⊕ U_i+1
    Y ← S_i ⊕ X_i ⊕ U_i+2
    """
    def __init__(self, max_step=30):
        self.env = Custom8119SCM(max_step=max_step)
        super().__init__()

    def see(self):
        u1, u2, u3, u4, u5 = self.env.sample_u()
        x = self.env.action(self.env.s, u1, u4)
        s, y, terminated, truncated, info = self.env.step(x, u1, u2, u3)
        return x, s, y, terminated, truncated, info 

    def do(self, action):
        u1, u2, u3, u4, u5 = self.env.sample_u()
        return self.env.step(action, u1, u2, u3)