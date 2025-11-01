import math
import numpy as np

from causal_gym import SCM, PCH
from causal_gym.core import PolicyType, ActType, ObsType, Graph

class DTRSCM(SCM):
    """
    A 2-stage Dynamic Treatment Regime (DTR) example from CRL book Chap. 8 Equation 8.2
    
    This implements a health treatment scenario where a physician must decide on 
    an initial treatment X1 and a follow-up treatment X2 for a patient.
    
    All variables are binary:
    - S1: patient's initial condition (0/1)
    - X1: initial treatment (0 = behavioral therapy, 1 = medication)
    - S2: patient's response (0 = non-responder, 1 = responder)
    - X2: follow-up treatment (0 = continue initial, 1 = intensive combined therapy)
    - Y: primary outcome (days of abstinence over 12 months)
    
    The causal mechanisms follow the structure from the book:
    - S1 ~ Bernoulli
    - X1 is determined by the behavioral policy (or by intervention)
    - S2 depends on S1 and X1
    - X2 is determined by the behavioral policy (or by intervention)
    - Y depends on S1, X1, S2, X2 with interactions
    
    ## Rewards
    The reward is the binary outcome Y representing treatment success.
    
    ## Termination
    Each episode consists of two stages of treatment, after which it terminates.
    """

    def __init__(self, a1=0, a2=0, s1_function=None, s2_function=None, y_function=None, u_distribution=None):
        """
        Initialize the DTR environment
        
        Args:
            a1, a2: Coefficients that determine the strength of unmeasured confounding.
                    When both are 0, the NUC condition holds.
        """
        self.a1 = a1
        self.a2 = a2
        self.s1 = 0 
        self.x1 = 0
        self.s2 = 0
        self.x2 = 0
        self.y = 0
        self.stage = 0  # 0: initial, 1: after X1, 2: terminated
        self._max_stage = 2
        self.u_distribution = u_distribution
        self.s1_function = s1_function
        self.s2_function = s2_function
        self.y_function = y_function
        
        # Default behavioral policy - can be overridden
        self._policy1 = lambda s1, u: int(3*s1 + self.a1*u + self._logistic() > 0)
        self._policy2 = lambda s1, x1, s2, u: int(3*s2 + self.a2*u + self._logistic() > 0)
        
    def _logistic(self):
        """Sample from a logistic distribution"""
        u = self.rng.random()
        return math.log(u / (1 - u))
        
    def reset(self, *, seed: int = None, options: dict = None) -> tuple[ObsType, dict]:
        """Reset the environment to initial state"""
        self.rng = np.random.default_rng(seed)
        if self.u_distribution:
            self.u = self.u_distribution
            self.s1 = self.s1_function
        else:
            # Unmeasured confounder U ~ Uniform(0, 1)
            self.u = self.rng.random() 
            # Initial patient condition S1
            self.s1 = int(self._logistic() > 0)  # S1 = I{U3 > 0}

        self.stage = 0
        
        return self.s1, {"stage": self.stage}
    
    def action(self):
        """Sample action from the behavioral policy based on current stage"""
        if self.stage == 0:
            return self._policy1(self.s1, self.u)
        elif self.stage == 1:
            return self._policy2(self.s1, self.x1, self.s2, self.u)
        else:
            raise ValueError("Episode already terminated")
    
    def step(self, action):
        """
        Transition the environment forward based on the action
        
        Returns:
            observation: The next observation (S2 after X1, or None after X2)
            reward: Treatment outcome (0 for failure, 1 for success)
            terminated: Whether the episode is terminated
            truncated: Always False for this environment
            info: Additional information
        """
        if self.stage == 0:
            # First stage: Assign initial treatment X1
            self.x1 = action
            if self.s2_function:
                self.s2 = self.s2_function
            else:
                # Update patient's response to treatment (S2)
                # S2 = I{0.1 + 0.1*S1 + 0.1*X1 + U4 > 0}
                self.s2 = int(0.1 + 0.1*self.s1 + 0.1*self.x1 + self._logistic() > 0)
            
            self.stage = 1
            return self.s2, 0, False, False, {"s1": self.s1, "x1": self.x1, "s2": self.s2, "stage": self.stage}
            
        elif self.stage == 1:
            # Second stage: Assign follow-up treatment X2
            self.x2 = action
            
            if self.y_function:
                self.y = self.y_function
            else:
                # Calculate outcome Y
                # Y = I{3U - 3S1 - 3X1 - 3S1X1 + 3X2 - 3S2X2 + 3X1X2 > 0}
                outcome_prob = 3*self.u - 3*self.s1 - 3*self.x1 - 3*self.s1*self.x1 + \
                            3*self.x2 - 3*self.s2*self.x2 + 3*self.x1*self.x2
                self.y = int(outcome_prob > 0)
            
            self.stage = 2
            return None, self.y, True, False, {
                "s1": self.s1, "x1": self.x1, "s2": self.s2, "x2": self.x2, "y": self.y, "stage": self.stage
            }
        
        else:
            raise ValueError("Episode already terminated")

    def change_policy1(self, new_policy):
        if new_policy != None:
            self._policy1 = new_policy
        return None
    
    def change_policy2(self, new_policy):
        if new_policy != None:
            self._policy2 = new_policy
        return None
    
    @property
    def get_graph(self):
        nodes = [
            {'name': 'S1', 'label': 'Initial Condition'},
            {'name': 'X1', 'label': 'Initial Treatment'},
            {'name': 'S2', 'label': 'Treatment Response'},
            {'name': 'X2', 'label': 'Follow-up Treatment'},
            {'name': 'Y', 'label': 'Outcome'}
        ]

        edges = [
            # Direct causal relationships
            {'from_': 'S1', 'to_': 'X1', 'type_': 'directed'},
            {'from_': 'S1', 'to_': 'S2', 'type_': 'directed'},
            {'from_': 'X1', 'to_': 'S2', 'type_': 'directed'},
            {'from_': 'S1', 'to_': 'X2', 'type_': 'directed'},
            {'from_': 'X1', 'to_': 'X2', 'type_': 'directed'},
            {'from_': 'S2', 'to_': 'X2', 'type_': 'directed'},
            {'from_': 'S1', 'to_': 'Y', 'type_': 'directed'},
            {'from_': 'X1', 'to_': 'Y', 'type_': 'directed'},
            {'from_': 'S2', 'to_': 'Y', 'type_': 'directed'},
            {'from_': 'X2', 'to_': 'Y', 'type_': 'directed'},

            # Bidirected edges representing unmeasured confounding through U
            {'from_': 'U', 'to_': 'X1', 'type_': 'latent'},
            {'from_': 'U', 'to_': 'X2', 'type_': 'latent'},
            {'from_': 'U', 'to_': 'S1', 'type_': 'latent'},
            {'from_': 'U', 'to_': 'S2', 'type_': 'latent'},
            {'from_': 'U', 'to_': 'Y', 'type_': 'latent'}
        ]
        
        graph = Graph(nodes=nodes, edges=edges)
        return graph

class DTRPCH(PCH):
    """PCH wrapper for the DTR Example environment.
    
    This implements two interaction modalities:
    - see: Passively observe the environment with behavioral policies
    - do: Actively intervene with a specified policy
    """
    
    def __init__(self, a1=0, a2=0):
        self.env = DTRSCM(a1, a2)
        super().__init__()
    
    def see(self, see_policy=None):
        """
        Observe the environment with behavioral policies
        
        Args:
            see_policy: Optional custom behavioral policy

        Returns a tuple of:
            action: The action taken by the behavioral policy
            observation: The next state
            reward: The reward (0 until end of episode, then outcome Y)
            terminated: Whether the episode is terminated
            truncated: Always False for this environment
            info: Additional information
        """
        if see_policy is not None:
            if self.env.stage == 0:
                a = see_policy(self.env.s1, self.env.u)
            else:
                a = see_policy(self.env.s1, self.env.x1, self.env.s2, self.env.u)
        else:
            a = self.env.action()
        o, r, term, trunc, info = self.env.step(a)
        info['natural_action'] = a
        return o, r, term, trunc, info
    
    def do(self, do_policy):
        """
        Intervene in the environment with a specified action
        
        Args:
            action: The action to take
            
        Returns a tuple of:
            observation: The next state
            reward: The reward (0 until end of episode, then outcome Y)
            terminated: Whether the episode is terminated
            truncated: Always False for this environment
            info: Additional information
        """
        if self.env.stage == 0:
            action = do_policy(self.env.s1)
        else:
            action = do_policy(self.env.s1, self.env.x1, self.env.s2)

        o, r, term, trunc, info = self.env.step(action)
        info['action'] = action
        return o, r, term, trunc, info
    

    # Counterfactual policy intervention
    def ctf_do(self, ctf_policy):
        intuition = self.env.action()
        if self.env.stage == 0:
            action = ctf_policy(self.env.s1, intuition)
        else:
            action = ctf_policy(self.env.s1, self.env.x1, self.env.s2, intuition)

        obs, r, terminated, truncated, info = self.env.step(action)
        info['natural_action'] = intuition
        info['action'] = action
        return obs, r, terminated, truncated, info
