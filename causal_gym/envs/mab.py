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
        Initialize the MAB environment
        
        Args:
            confounding_strength: Strength of confounding between U -> X and U -> Y

            arms_probs: Optional override for arm reward probabilities
                       [p(Y=1|X=0), p(Y=1|X=1)] when confounding is disabled
        """
        self.confounding_strength = confounding_strength
        
        # By default arm 0 has higher reward (0.4 vs 0.3)
        if arms_probs is None:
            self.arms_probs = [0.4, 0.3]
        else:
            self.arms_probs = arms_probs
            
        self.x = None
        self.y = None
        self.u = None  # Unmeasured confounder
        
        # default behavioral policy - uniformly random
        self._policy = lambda u: int((0.8 + self.confounding_strength * u) > self.rng.random())

    def sample_u(self):
        """Sample the unmeasured confounder U ~ Uniform(0, 1)"""
        return self.rng.random()
        
    def reset(self, *, seed: int = None, options: dict = None) -> tuple[ObsType, dict]:
        """Reset the environment for a new stage"""
        self.rng = np.random.default_rng(seed)
        
        # Sample confounder U ~ Uniform(0, 1)
        self.u = self.sample_u()
        
        return None, {}
    
    def action(self):
        """Sample action from the behavioral policy"""
        return self._policy(self.u)
    
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
        
        # Determine reward based on arm choice and confounding
        if self.confounding_strength == 0:
            # No confounding - use fixed arm probabilities
            success_prob = self.arms_probs[action]
        else:
            # With confounding - reward depends on both arm and confounder
            # Using the mechanism from Example 7.1: Y ← I{U < 0.4 − D·X}
            # where D controls the gap between arms
            D = self.arms_probs[0] - self.arms_probs[1]  # Difference between arm probabilities
            threshold = self.arms_probs[0] - D * action
            success_prob = threshold - self.confounding_strength * self.u
            
        # Sample reward
        self.y = int(self.rng.random() < success_prob)
        
        return None, self.y, True, False, {"arm": action, "reward": self.y}
    
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
            {'name': 'Y', 'label': 'Reward'}
        ]

        edges = [
            {'from_': 'X', 'to_': 'Y', 'type_': 'directed'},
            {'from_': 'X', 'to_': "Y", 'type_': 'bidirected'}
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
            a = see_policy(self.env.u)
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
        action = ctf_policy(self.env.observation(), intuition)
        obs, r, terminated, truncated, info = self.env.step(action)
        info['natural_action'] = intuition
        info['action'] = action
        return obs, r, terminated, truncated, info