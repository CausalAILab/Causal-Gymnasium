import copy
import numpy as np
import gymnasium as gym

from ..core import PolicyType, ActType, ObsType, SCM, PCH, Task, Graph



class RobotWalkSCM(SCM):
    def __init__(self, length=10, timelimit=30, put=[.5, .5]):
        super().__init__()
        self.stable_u_dist = put
        # 1-dim maze, hallway
        self.goal_location = length
        self.time_limit = timelimit
        self.state_space = np.array([length+1, 2])
        self.observation_space = gym.spaces.Box(low=np.array([0, 0]), high=np.array([length, 1]), shape=[2,], dtype=int, )
        self.action_space = gym.spaces.Discrete(2)

    def reset(self, seed=None, options=None):
        super().reset(seed = seed)
        self.rng = np.random.default_rng(seed)
        self.steps_cnt = 0
        self.current_location = 0
        self.is_stable = 1
        self.ut = self.sample_u()
        return (self.current_location, self.is_stable), {'ut': self.ut}
    
    def sample_u(self):
        ut = self.rng.choice(2, p=self.stable_u_dist)
        return ut
    
    def step(self, action):
        self.steps_cnt += 1
        if (self.is_stable and action == 0) or (not self.is_stable and action == self.ut):
            next_location = self.current_location + 1
        else:
            next_location = self.current_location
        if self.is_stable:
            next_stable = 1 - action
        else:
            next_stable = int(action == self.ut)
        
        # +1 for moving forward, -1 for wrong movement when unstable, +1 for reaching goal
        discount = .5 if self.is_stable and action == 0 else 1
        reward = discount * (next_location - self.current_location) - int(not self.is_stable and action != self.ut)
        if next_location == self.goal_location:
            reward = 1
        self.current_location = next_location
        self.is_stable = next_stable
        truncated = self.steps_cnt > self.time_limit
        # update ut
        self.ut = self.sample_u()
        info = {
            'ut':self.ut
        }
        return (next_location, next_stable), reward, (next_location == self.goal_location), truncated, info
    
    def action(self):
        # A perfect behavioral policy that can walk like a human does
        if self.is_stable:
            return 1
        else:
            return self.ut
        
    def observation(self):
        return (copy.deepcopy(self.current_location), copy.deepcopy(self.is_stable))

    @property
    def get_graph(self):
        nodes = [
            {'name': 'L', 'label': 'Location'},
            {'name': 'F', 'label': 'Stability'},
            {'name': 'X', 'label': 'Action'},
            {'name': 'Y', 'label': 'Reward'},
            {'name': "L'", 'label': 'Next Location'},
            {'name': "F'", 'label': 'Next Stability'}
        ]

        edges = [
            # {'from_': 'U', 'to_': 'X', 'type_': 'directed'},
            # {'from_': 'U', 'to_': "S'", 'type_': 'directed'},
            {'from_': 'L', 'to_': 'X', 'type_': 'directed'},
            {'from_': 'F', 'to_': 'X', 'type_': 'directed'},
            {'from_': 'L', 'to_': 'Y', 'type_': 'directed'},
            {'from_': 'F', 'to_': 'Y', 'type_': 'directed'},
            {'from_': 'X', 'to_': 'Y', 'type_': 'directed'},
            {'from_': 'L', 'to_': "L'", 'type_': 'directed'},
            {'from_': 'F', 'to_': "F'", 'type_': 'directed'},
            {'from_': 'X', 'to_': "L'", 'type_': 'directed'},
            {'from_': 'X', 'to_': "F'", 'type_': 'directed'},
            # Bidirected confounding between Action and Next State
            {'from_': 'X', 'to_': "L'", 'type_': 'bidirected'},
            {'from_': 'X', 'to_': "F'", 'type_': 'bidirected'},
            {'from_': 'Y', 'to_': "L'", 'type_': 'bidirected'},
            {'from_': 'Y', 'to_': "F'", 'type_': 'bidirected'},
        ]
        return Graph(nodes=nodes, edges=edges)
    
    def render(self):
        return self.observation()


class RobotWalkPCH(PCH):
    """
    RobotWalk is a simple 1-dimensional grid environment designed to study sequential decision making under uncertainty. 
    An agent (robot) must traverse a hallway to reach a goal, but its ability to move forward depends on its internal stability state and a latent variable. 
    The environment supports observational, interventions and counterfactual reasoning, making it suitable for causal reinforcement learning experiments.
    """
    metadata = {"render_modes": ["rgb_array"]}
    def __init__(self, length=10, timelimit=30, put=[.5, .5], task=Task()):
        self.env: RobotWalkSCM = RobotWalkSCM(length, timelimit, put)
        super().__init__(task=task)

    def __getattr__(self, name: str):
        if name == "_np_random":
            raise AttributeError(
                "Can't access `_np_random` of a wrapper, use `self.unwrapped._np_random` or `self.np_random`."
            )
        elif name.startswith("_"):
            raise AttributeError(f"accessing private attribute '{name}' is prohibited")
        if hasattr(self.env, name):
            return getattr(self.env, name)
        else:
            return self.env.__getattr__(name)

    def do(self, do_policy):
        action = do_policy((self.env.current_location, self.env.is_stable))
        state, reward, terminated, truncated, info = self.env.step(action)
        info['action'] = action
        return state, reward, terminated, truncated, info
    
    # Counterfactual policy intervention
    def ctf_do(self, ctf_policy):
        intuition = self.env.action()
        action = ctf_policy(self.env.observation(), intuition)
        state, r, terminated, truncated, info = self.env.step(action)
        info['natural_action'] = intuition
        info['action'] = action
        return state, r, terminated, truncated, info

    def see(self, see_policy=None):
        if see_policy is None:
            action = self.env.action()
        else:
            action = see_policy((self.env.current_location, self.env.is_stable), self.env.ut)
        state, reward, terminated, truncated, info = self.env.step(action)
        info['natural_action'] = action
        return state, reward, terminated, truncated, info
    
    @staticmethod
    def PolicyMapping(q_table: np.array):
        # Translate q-value to interpretable policy string matrix
        return np.argmax(q_table, axis=2)

    

    
