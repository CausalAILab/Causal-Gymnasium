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
        # Define action and observation spaces here

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
        pass

    def action(self) -> ActType:
        pass

    def observation(self) -> ObsType:
        pass

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, bool, Dict[str, Any]]:
        pass

    def render(self) -> ObsType:
        pass

    @property
    def get_graph(self) -> Tuple[Dict[int, str], list[list[int]], list[list[int]]]:
        pass


class MNISTPCH(PCH):
    """PCH wrapper for MNISTSCM."""

    def __init__(self, seed: int = None):
        super().__init__()
        # Initialize MNISTSCM instance here

    def see(self) -> Tuple[ActType, ObsType, float, bool, bool, Dict[str, Any]]:
        pass

    def do(self, action: ActType) -> Tuple[ObsType, float, bool, bool, Dict[str, Any]]:
        pass

    def reset(self, *, seed: int = None, options: dict = None) -> Tuple[ObsType, dict]:
        pass

    def render(self) -> ObsType:
        pass
