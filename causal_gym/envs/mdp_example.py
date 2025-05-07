import math
import numpy as np

from causal_gym import SCM, PCH
from causal_gym.core import PolicyType, ActType, ObsType

class MDPExampleSCM(SCM):
    """
    A confounded MDP from CRL book Chap. 7 Example 7.2.
    See also the inventory control example from Csaba 2010 - Algos for RL, Example 1.
    By default the behavioral policy observes both the state 
    and the confounder between state and reward.

    Note that this environment never terminates but truncate at max_step (default: 30)
    
    All variables are binary (state, action, reward, confouders). 

    ## Rewards

    At each time step, a reward of '1' is given for success, and '0' for failure.

    ## Termination

    The episode ends if any one of the following conditions is met:

    1. Timeout (see `max_steps`).

    """

    def __init__(self, init_dist=[.5,.5], max_step=30):
        assert sum(init_dist) == 1.0, f"Init state distribution must sum to 1!"
        self.init_dist = init_dist
        self.s, self.x = 0, 0
        self.prev_s, self.prev_x = self.s, self.x
        # behavioral policy
        self._policy = lambda s, u1: int(s != u1)
        self._u1 = lambda: self.rng.choice(2, p = [.1, .9])
        self._u2 = lambda: self.rng.choice(2, p = [.9, .1])
        self._u3 = lambda: self.rng.choice(2, p = [.9, .1])
        self.num_step = 0
        self._max_step = max_step

    def reset(self, *, seed: int = None, options: dict = None) -> tuple[ObsType, dict]:
        self.rng = np.random.default_rng(seed)
        self.s = self.rng.choice(2, p = self.init_dist)
        self.num_step = 0
        # empty info
        return self.s, {}
    
    def action(self, s: int, u1: int) -> int:
        """
        sample action from the behavioral policy
        """
        return self._policy(s, u1)
    
    def state_transition(self, u1: int, u2: int, s: int, x: int) -> int:
        return (u1 != u2) != (s | x)
    
    # def see(self):
    #     self.x = self.action(self.s, u1)
    #     self.num_step += 1
    #     u1 = self._u1()
    #     u2 = self._u2()
    #     u3 = self._u3()
    #     self.x = self.action(self.s, u1)
    #     self.y = ((self.s != self.x) != u1) != u3
    #     # next state
    #     self.s = self.state_transition(u1, u2, self.s, self.x)
    #     # Return action, next state, reward, terminated, truncated, info
    #     return self.x, self.s, self.y, False, self.num_step > self._max_step, {}

    def sample_u(self):
        """
        Sample exogeneous variables
        """
        return self._u1(), self._u2(), self._u3()
    
    def step(self, x, u1, u2, u3):
        self.num_step += 1
        # reward
        self.y = ((self.s != x) != u1) != u3
        # next state
        self.s = self.state_transition(u1, u2, self.s, x)
        # Return next state, reward, terminated, truncated, info
        return self.s, self.y, False, self.num_step > self._max_step, {}
    
    def change_policy(self, new_policy):
        if new_policy != None:
            self._policy = new_policy
        return None
    

class MDPExamplePCH(PCH):
    """PCH for the MDP Example defined above.
    """
    def __init__(self, init_dist=[.5,.5], max_step=30):
        self.env = MDPExampleSCM(init_dist, max_step)
        super().__init__()

    def see(self):
        u1, u2, u3 = self.env.sample_u()
        x = self.env.action(self.env.s, u1)
        s, y, terminated, truncated, info = self.env.step(x, u1, u2, u3)
        return x, s, y, terminated, truncated, info 

    def do(self, action):
        u1, u2, u3 = self.env.sample_u()
        return self.env.step(action, u1, u2, u3)