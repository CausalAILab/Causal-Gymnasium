import numpy as np
from typing import Any, Tuple, Dict

from causal_gym import SCM, PCH
from causal_gym.core import ObsType, ActType, PolicyType
from gymnasium import spaces

from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
import torch
import os


class MNISTSCM(SCM):
    """Causal environment for the MNIST digits experiment."""

    def __init__(self, seed: int = None):
        super().__init__()
        self._dataset_loaded = False
        self._load_binary_mnist()
        self._np_random = np.random.default_rng(seed)
        self.u = None
        self.x = None
        self.w = None
        self.s = None
        self.y = None

        # Define action space: X ∈ {0, 1}
        self.action_space = spaces.Discrete(2)
        # Observation space: S is an image, shape (1, 28, 28), values in [0, 1]
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(1, 28, 28), dtype=np.float32)

    def _load_binary_mnist(self):
        """Load and filter MNIST to only include digits 0 and 1."""
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.numpy())
        ])

        dataset = MNIST(root=os.path.expanduser("~/.mnist"), train=True, download=True, transform=transform)

        binary_indices = [i for i, (img, label) in enumerate(dataset) if label in [0, 1]]
        self.binary_dataset = Subset(dataset, binary_indices)

        self.digit_0 = [img for img, label in self.binary_dataset if label == 0]
        self.digit_1 = [img for img, label in self.binary_dataset if label == 1]

        self._dataset_loaded = True

    def reset(self, *, seed: int = None, options: dict = None) -> Tuple[ObsType, dict]:
        if seed is not None:
            self._np_random = np.random.default_rng(seed)
        self.u = self._np_random.integers(0, 2)  # U ~ {0, 1}
        self.x = None
        self.w = None
        self.s = None
        self.y = None
        return None, {"u": self.u}

    def action(self) -> ActType:
        return self.x

    def observation(self) -> ObsType:
        return self.s.astype(np.float32) if self.s is not None else None

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, bool, Dict[str, Any]]:
        self.x = action
        if self.x == 0:
            self.w = self._np_random.choice(self.digit_0)
        else:
            self.w = self._np_random.choice(self.digit_1)

        if self.u == 0:
            self.s = self.w
        else:
            noise = self._np_random.normal(loc=0.0, scale=0.1, size=self.w.shape)
            self.s = np.clip(self.w + noise, 0.0, 1.0)

        obs = self.s.astype(np.float32)
        info = {
            "x": self.x,
            "u": self.u,
            "w": self.w,
            "s": self.s,
        }
        return obs, 0.0, True, False, info

    def render(self) -> ObsType:
        import matplotlib.pyplot as plt

        if self.w is None or self.s is None:
            print("Nothing to render. Run `do()` or `see()` first.")
            return

        fig, axs = plt.subplots(1, 2)
        axs[0].imshow(self.w.squeeze(), cmap='gray')
        axs[0].set_title(f'W (digit {self.x})')
        axs[0].axis('off')

        axs[1].imshow(self.s.squeeze(), cmap='gray')
        axs[1].set_title('S (surrogate)')
        axs[1].axis('off')

        plt.tight_layout()
        plt.show()

    @property
    def get_graph(self) -> Tuple[Dict[int, str], list[list[int]], list[list[int]]]:
        nodes = {
            0: "X",
            1: "W",
            2: "S",
            3: "Y"
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


class MNISTPCH(PCH):
    """PCH wrapper for MNISTSCM."""

    def __init__(self, seed: int = None):
        super().__init__()
        self.env = MNISTSCM(seed=seed)

    def see(self) -> Tuple[ActType, ObsType, float, bool, bool, Dict[str, Any]]:
        self.env.u = self.env._np_random.integers(0, 2)
        self.env.x = 1 - self.env.u  # behavior policy: x = 1 - u
        return self.env.step(self.env.x)

    def do(self, action: ActType) -> Tuple[ObsType, float, bool, bool, Dict[str, Any]]:
        return self.env.step(action)

    def reset(self, *, seed: int = None, options: dict = None) -> Tuple[ObsType, dict]:
        return self.env.reset(seed=seed, options=options)

    def render(self) -> ObsType:
        return self.env.render()
