import math
import numpy as np

from causal_gym import SCM, PCH
from causal_gym.core import PolicyType, ActType, ObsType, Task, Graph

class MABSCM(SCM):
    """
    This implements a classic MAB scenario with optional confounding to demonstrate
    the difference between observational data and interventional data.
    
    The environment has:
    - X: arm choice (0 or 1)
    - Y: reward (binary)
    - U: optional unmeasured confounder affecting both X and Y
    
    The causal mechanisms follow the structure from the book:
    - X is determined by behavioral policy (or by intervention)
    - Y depends on X and potentially U (if confounding is enabled)
    
    ## Rewards
    A reward of '1' is given based on the success probability of the chosen arm.
    
    ## Termination
    Each episode consists of a single arm pull, after which it terminates.
    """

    def __init__(self, confounding_strength=0.0, arms_probs=None):
        """
        Initialize the Greedy Casino environment
        """
        self.x = None
        self.y = None
        self.b = None
        self.d = None

        self.payoff = {
            (0, 0, 0): 0.10,
            (0, 0, 1): 0.50,
            (0, 1, 0): 0.40,
            (0, 1, 1): 0.20,
            (1, 0, 0): 0.50,
            (1, 0, 1): 0.10,
            (1, 1, 0): 0.20,
            (1, 1, 1): 0.40,
        }
        
        # default behavioral policy - uniformly random
        self._policy = lambda b, d: b ^ d

    def sample_u(self):
        """Sample the unmeasured confounders B,D ~ Bernoulli(0.5)"""
        return bool(self.rng.binomial(1, 0.5)), bool(self.rng.binomial(1, 0.5))
        
    def reset(self, *, seed: int = None, options: dict = None) -> tuple[ObsType, dict]:
        """Reset the environment for a new stage"""
        self.rng = np.random.default_rng(seed)
        
        # Sample confounders B and D ~ Uniform(0, 1)
        self.b, self.d = self.sample_u()
        
        return None, {}
    
    def action(self):
        """Sample action from the behavioral policy"""
        return self._policy(self.b, self.d)
    
    def sample_y(self):
        mean_y = self.payoff[(self.x, self.b, self.d)]
        return self.rng.binomial(1, mean_y)

    def step(self, action):
        """
        Transition the environment based on the action
        
        Args:
            action: Arm to pull (0 or 1)
            
        Returns:
            observation: None (MAB has no state)
            reward: 1 for success, 0 for failure
            terminated: Always True (episode ends after one arm pull)
            truncated: Always False
            info: Additional information
        """
        self.x = action
        self.y = self.rng.binomial(1, self.payoff[(self.x, self.b, self.d)])
        
        return None, self.y, True, False, {"action": action, "reward": self.y}
    
    def change_policy(self, new_policy):
        if new_policy != None:
            self._policy = new_policy
        return None
    
    # Causal graph
    @property
    def get_graph(self):
        nodes = [
            # {'name': 'U', 'label': 'Confounder', 'type': 'latent'},
            {'name': 'X', 'label': 'Action'},
            {'name': 'Y', 'label': 'Reward'},
            {'name': 'B', 'label': 'Blinking', 'type': 'latent'},
            {'name': 'D', 'label': 'Drunkenness', 'type': 'latent'}      
        ]

        edges = [
            {'from_': 'X', 'to_': 'Y', 'type_': 'directed'},
            {'from_': 'B', 'to_': 'X', 'type_': 'directed'},
            {'from_': 'D', 'to_': 'X', 'type_': 'directed'},
            {'from_': 'B', 'to_': 'Y', 'type_': 'directed'},
            {'from_': 'D', 'to_': 'Y', 'type_': 'directed'}
        ]
        graph = Graph(nodes=nodes, edges=edges)
        return graph

class MABPCH(PCH):
    """PCH wrapper for the MAB Example environment."""
    
    def __init__(self, confounding_strength=0.0, arms_probs=None):
        self.env = MABSCM(confounding_strength, arms_probs)
        super().__init__()
    
    def see(self, see_policy=None):
        # Override default policy if provided
        if see_policy is not None:
            a = see_policy(self.env.b, self.env.d)
        else:
            a = self.env.action()
        
        o, r, term, trunc, info = self.env.step(a)
        info['natural_action'] = a
        return o, r, term, trunc, info
    
    def do(self, do_policy):
        action = do_policy()
        o, r, term, trunc, info = self.env.step(action)
        info['action'] = action
        return o, r, term, trunc, info
    
    # Counterfactual policy intervention
    def ctf_do(self, ctf_policy):
        intuition = self.env.action()
        action = ctf_policy(intuition)
        obs, r, terminated, truncated, info = self.env.step(action)
        info['natural_action'] = intuition
        info['action'] = action
        return obs, r, terminated, truncated, info