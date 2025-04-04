import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from causal_gym import SCM, PCH

# MNIST dataset loader
class MNISTLoader:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        self.train_set = datasets.MNIST(root='./data', train=True, download=True, transform=self.transform)
    
    def get_batch(self, batch_size=64):
        loader = torch.utils.data.DataLoader(self.train_set, batch_size=batch_size, shuffle=True)
        return next(iter(loader))

# GAN components (from R-66 Section 4.2)
class Generator(nn.Module):
    def __init__(self, latent_dim=100):
        super().__init__()

        self.main = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(True),
            nn.Linear(256, 28 * 28),
            nn.Tanh()
        )

    def forward(self, z):
        return self.main(z).view(-1, 1, 28, 28)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.main = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x)