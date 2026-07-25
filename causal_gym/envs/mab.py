import math
import numpy as np
from gymnasium import spaces

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
            confounding_strength: Probability that the action and reward
                mechanisms share the same exogenous variable. ``0`` is
                unconfounded and ``1`` is the Chapter 7 SCM.

            arms_probs: Optional override for the two structural reward
                thresholds [p(Y=1|do(X=0)), p(Y=1|do(X=1))].
        """
        if not 0.0 <= confounding_strength <= 1.0:
            raise ValueError("confounding_strength must be between 0 and 1")
        self.confounding_strength = confounding_strength
        
        # By default arm 0 has higher reward (0.4 vs 0.3)
        if arms_probs is None:
            self.arms_probs = [0.4, 0.3]
        else:
            self.arms_probs = arms_probs

        if len(self.arms_probs) != 2 or any(
            probability < 0.0 or probability > 1.0
            for probability in self.arms_probs
        ):
            raise ValueError("arms_probs must contain two probabilities in [0, 1]")
            
        self.x = None
        self.y = None
        self.u = None
        self._reward_u = None
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Discrete(1)
        
        # Physician's behavioral policy from Chapter 7: X <- I[U < 0.8].
        self._policy = lambda u: int(u < 0.8)

    def sample_u(self):
        """Sample the unmeasured confounder U ~ Uniform(0, 1)"""
        return self.rng.random()
        
    def reset(self, *, seed: int = None, options: dict = None) -> tuple[ObsType, dict]:
        """Reset the environment for a new stage"""
        self.rng = np.random.default_rng(seed)
        
        # U drives the natural action. The reward uses the same U for a
        # confounded unit and an independent draw otherwise. This mixture gives
        # confounding_strength a continuous meaning while preserving the arm
        # marginals under intervention.
        self.u = self.sample_u()
        if self.confounding_strength == 1.0:
            self._reward_u = self.u
        elif self.confounding_strength == 0.0:
            self._reward_u = self.sample_u()
        elif self.rng.random() < self.confounding_strength:
            self._reward_u = self.u
        else:
            self._reward_u = self.sample_u()
        
        return self.observation(), {}

    def observation(self):
        return 0
    
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
        
        # Chapter 7 uses the structural equation Y <- I[U_Y < p_X]. It is a
        # deterministic function of the already sampled exogenous context; it
        # must not be interpreted as a probability and sampled a second time.
        threshold = self.arms_probs[action]
        self.y = int(self._reward_u < threshold)
        
        return self.observation(), self.y, True, False, {"arm": action, "reward": self.y}
    
    def change_policy(self, new_policy):
        if new_policy != None:
            self._policy = new_policy
        return None
    
    # Causal graph
    @property
    def get_graph(self):
        nodes = [
            {'name': 'X', 'label': 'Action'},
            {'name': 'Y', 'label': 'Reward'}
        ]

        edges = [
            {'from_': 'X', 'to_': 'Y', 'type_': 'directed'},
        ]
        if self.confounding_strength != 0:
            edges.append({'from_': 'X', 'to_': "Y", 'type_': 'bidirected'})
        graph = Graph(nodes=nodes, edges=edges)
        return graph

def _call_policy(policy, *args):
    try:
        return policy(*args)
    except TypeError:
        return policy()

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
        action = _call_policy(do_policy, self.env.observation())
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
