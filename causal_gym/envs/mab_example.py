import math
import numpy as np

from causal_gym import SCM, PCH
from causal_gym.core import PolicyType, ActType, ObsType

class MABExampleSCM(SCM):
    """
    A Multi-Armed Bandit (MAB) environment from CRL book Chap. 8 Example 8.8.
    
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
                                 (0 = no confounding, as in Example 8.8)
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
        
    def reset(self, *, seed: int = None, options: dict = None) -> tuple[ObsType, dict]:
        """Reset the environment for a new stage"""
        self.rng = np.random.default_rng(seed)
        
        # Sample confounder U ~ Uniform(0, 1)
        self.u = self.rng.random()
        
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


class MABExamplePCH(PCH):
    """PCH wrapper for the MAB Example environment.
    
    This implements two interaction modalities:
    - see: Passively observe the environment with behavioral policy
    - do: Actively intervene by pulling a specific arm
    """
    
    def __init__(self, confounding_strength=0.0, arms_probs=None):
        self.env = MABExampleSCM(confounding_strength, arms_probs)
        super().__init__()
    
    def see(self, bpolicy=None):
        """
        Observe the environment with behavioral policy
        
        Args:
            bpolicy: Optional custom behavioral policy
            
        Returns a tuple of:
            action: The arm pulled by the behavioral policy
            observation: None
            reward: The reward (0 or 1)
            terminated: Always True
            truncated: Always False
            info: Additional information
        """
        # Override default policy if provided
        if bpolicy is not None:
            original_policy = self.env._policy
            self.env._policy = lambda u: bpolicy()
        
        try:
            # Reset environment
            self.env.reset()
            
            # Sample action from behavioral policy
            action = self.env.action()
            
            # Take a step in the environment
            _, reward, terminated, truncated, info = self.env.step(action)
            
            return action, None, reward, terminated, truncated, info
        
        finally:
            # Restore original policy
            if bpolicy is not None:
                self.env._policy = original_policy
    
    def do(self, action):
        """
        Intervene in the environment by pulling a specific arm
        
        Args:
            action: The arm to pull (0 or 1)
            
        Returns tuple:
            observation: None
            reward: The reward (0 or 1)
            terminated: Always True
            truncated: Always False
            info: Additional information
        """
        # Reset environment
        self.env.reset()
        
        # Take a step with the specified action
        return self.env.step(action)