import numpy as np
from typing import Any, Tuple, Dict

from causal_gym import SCM, PCH
from causal_gym.core import ObsType, ActType, PolicyType
from gymnasium import spaces

from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import Subset
import os


class MNISTSCM(SCM):
    '''Causal environment for the MNIST digits experiment.'''

    def __init__(self, seed: int = None):
        super().__init__()
        self.rng = np.random.default_rng(seed)

        # load dataset
        self._load_binary_mnist()

        self._u = None
        self.x = None
        self.w = None
        self.s = None
        self.y = None

        # SCM says to set these no matter what but I don't see a use for them yet
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Dict({
            'x': spaces.Discrete(2),
            'w': spaces.Box(low=0, high=1, shape=(1, 28, 28), dtype=np.float32),
            's': spaces.Discrete(2)
        })

    def _load_binary_mnist(self):
        '''Load and filter MNIST to only include digits 0 and 1.'''
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.numpy())
        ])

        dataset = MNIST(root=os.path.expanduser('~/.mnist'), train=True, download=True, transform=transform)

        binary_indices = [i for i, (img, label) in enumerate(dataset) if label in [0, 1]]
        self.binary_dataset = Subset(dataset, binary_indices)

        self.digit_0 = [img for img, label in self.binary_dataset if label == 0]
        self.digit_1 = [img for img, label in self.binary_dataset if label == 1]

    def reset(self, *, seed: int = None) -> Tuple[ObsType, dict]:
        self.rng = np.random.default_rng(seed)

        self._u = None
        self.x = None
        self.w = None
        self.s = None
        self.y = None

        return self.s, {}

    def action(self, u) -> ActType:
        return self.rng.choice(2, p=[0.9, 0.1]) if u == 0 else self.rng.choice(2, p=[0.1, 0.9])

    def observation(self):
        return {'x': self.x, 'w': self.w, 's': self.s}

    def sample_u(self) -> int:
        '''Sample u from P(u).'''
        self._u = self.rng.choice(2, p=[0.1, 0.9])
        return self._u

    def step(self, action, show_reward = False) -> Tuple[ObsType, float, bool, bool, Dict[str, Any]]:
        # sample u from P(u)
        if self._u is None:
            self.sample_u()

        # sample x from the action
        self.x = action

        # sample w from P(w|x)
        digit = None
        if self.x == 0:
            digit = self.rng.choice(2, p=[0.9, 0.1])
        else:
            digit = self.rng.choice(2, p=[0.1, 0.9])

        self.w = self.rng.choice(self.digit_0 if digit == 0 else self.digit_1)

        # sample s from P(s|u, w)
        if self._u == 0 and digit == 0:
            prob_s_1 = 0.1
        elif self._u == 0 and digit == 1:
            prob_s_1 = 0.9
        elif self._u == 1 and digit == 0:
            prob_s_1 = 0.9
        else:
            prob_s_1 = 0.1

        self.s = self.rng.choice(2, p=[1 - prob_s_1, prob_s_1])

        # set Y using Y <- !s
        # this should be latent
        self.y = 1 - self.s

        reward = self.y if show_reward else None

        # observation, reward, terminated, truncated, info
        obs = {'x': self.x, 's': self.s}
        return obs, reward, True, True, {'u': self._u, 'y': self.y}

    def render(self, render_mode = 'human') -> ObsType:
        '''
        Render the environment. In this case, we will render the image of the digit and print other variables.
        render_mode: 'human' (show the image) or 'rgb_array' (return the image as an array).
        '''
        import matplotlib.pyplot as plt
        import numpy as np

        if self.w is None:
            raise ValueError('Nothing to render. Run `do()` or `see()` first.')
        
        fig, ax = plt.subplots()
        ax.imshow(self.w.squeeze(), cmap='gray')
        ax.set_title(f'W (digit {self.x})')
        ax.axis('off')

        if render_mode == 'human':
            return fig, self.x, self.s
        elif render_mode == 'rgb_array':
            fig.canvas.draw()
            img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            plt.close(fig)
        else:
            raise ValueError(f'Unknown render mode: {render_mode}. Please choose "human" or "rgb_array".')

    @property
    def get_graph(self) -> Tuple[Dict[int, str], list[list[int]], list[list[int]]]:
        nodes = {
            0: 'X',
            1: 'W',
            2: 'S',
            3: 'Y'
        }

        base_graph = [
            [0, 1, 0, 0],  # X
            [0, 0, 1, 0],  # W
            [0, 0, 0, 1],  # S
            [0, 0, 0, 0],  # Y
        ]

        conf_graph = [
            [0, 0, 1, 0],  # X
            [0, 0, 0, 0],  # W
            [1, 0, 0, 0],  # S
            [0, 0, 0, 0],  # Y
        ]

        return nodes, base_graph, conf_graph
    
    @property
    def observed_unobserved_vars(self) -> Tuple[list[str], list[str]]:
        return ['X', 'W', 'S'], ['Y']


class MNISTPCH(PCH):
    '''PCH wrapper for MNISTSCM.'''

    def __init__(self, seed: int = None):
        self.env: MNISTSCM = MNISTSCM(seed=seed)  # Ensure env is set before base class init
        super().__init__()

    def see(self, behavioral_policy = None) -> Tuple[ActType, ObsType, float, bool, bool, Dict[str, Any]]:
        if behavioral_policy is not None:
            # step thru expert's policy
            action = behavioral_policy(self.env.sample_u())
        else:
            u = self.env.sample_u()
            action = self.env.action(u)

        obs, reward, terminated, truncated, info = self.env.step(action)
        return action, obs, reward, terminated, truncated, info

    def do(self, action: ActType) -> Tuple[ObsType, float, bool, bool, Dict[str, Any]]:
        return self.env.step(action)
    
    # Counterfactual policy intervention
    def ctf_do(self, ctf_policy):
        intuition = self.env.action()
        action = ctf_policy(self.env.observation(), intuition)
        return self.env.step(action)

    def reset(self, *, seed: int = None) -> Tuple[ObsType, dict]:
        return self.env.reset(seed=seed)

    def render(self, render_mode = 'human') -> ObsType:
        return self.env.render(render_mode=render_mode)