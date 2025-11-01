import numpy as np

from causal_gym import SCM, PCH, Task, Graph

class ExampleSCM_9_5(SCM):
    def __init__(self):
        self._policy = self.F
        self._u1 = lambda: self.rng.choice(2, p=[.5, .5])
        self._u2 = lambda: self.rng.choice(2, p=[.5, .5])

    def reset(self, *, seed: int = None, options: dict = None) -> tuple[dict]:
        self.rng = np.random.default_rng(seed)
        return None, {}
    
    def sample_u(self):
        """
        Sample exogeneous variables
        """
        return self._u1(), self._u2()
   
    def F(self, u1: int, u2: int):
        x1 = u1
        x2 = x1 ^ u2
        return x1, x2
    
    def action(self, u1: int, u2: int, policy = None):
        if not policy:
            policy = self._policy
        return policy(u1, u2)
    
    def observation(self):
        # MAB alike environment, instrument variable with bow graph 
        return None
    
    def step(self, u1: int, u2: int, x1: int = None, x2: int = None):
        if x1 is None:
            x1 = u1
        if x2 is None: 
            x2 = x1 ^ u2
        self.y = x2 ^ u2
        # This is a single step environment
        return None, self.y, 1, 1, {}
    
    def get_graph(self):
        nodes = [
            {'name': 'X1', 'label': 'Action1'},
            {'name': 'X2', 'label': 'Action2'},
            {'name': 'Y', 'label': 'Reward'},
        ]
        edges = [
            {'from_': 'X1', 'to_': 'X2', 'type_': 'directed'},
            {'from_': 'X2', 'to_': 'Y', 'type_': 'directed'},
            {'from_': 'X2', 'to_': 'Y', 'type_': 'bidirected'},
        ]
        return Graph(nodes=nodes, edges=edges)


class ExamplePCH_9_5(PCH):
    def __init__(self, task=Task()):
        self.env: ExampleSCM_9_5 = ExampleSCM_9_5()
        super().__init__(task=task)
    
    # Observational step under behaviour policy
    def see(self, see_policy=None):
        u1, u2 = self.env.sample_u()
        if see_policy is not None:
            a = see_policy(self.env.observation(), (u1, u2))
        else:
            a = self.env.action(u1, u2)
        o, r, term, trunc, info = self.env.step(u1, u2, a[0], a[1])
        info['natural_action'] = a
        return o, r, term, trunc, info

    # Interventional step with forced action
    def do(self, do_policy):
        u1, u2 = self.env.sample_u()
        do_action: dict = do_policy(self.env.observation())
        action = []
        action.append(do_action.pop('X1', None))
        action.append(do_action.pop('X2', None))
        o, r, term, trunc, info = self.env.step(u1, u2, action[0], action[1])
        info['action'] = action
        return o, r, term, trunc, info
    
    # Counterfactual policy intervention
    def ctf_do(self, ctf_policy):
        u1, u2 = self.env.sample_u()
        intuition = self.env.action(u1, u2)
        do_action = ctf_policy(self.env.observation(), intuition)
        action = []
        action.append(do_action.pop('X1', None))
        action.append(do_action.pop('X2', None))
        obs, r, terminated, truncated, info = self.env.step(u1, u2, action[0], action[1])
        info['natural_action'] = intuition
        info['action'] = action
        return obs, r, terminated, truncated, info