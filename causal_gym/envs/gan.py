import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as spectral_norm
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# ---------------------------
# GUMBEL SOFTMAX UTILITIES
# ---------------------------
def sample_gumbel(shape, eps=1e-20, device='cpu'):
    U = torch.rand(shape, device=device)
    return -torch.log(-torch.log(U + eps) + eps)

def gumbel_softmax_sample(logits, temperature):
    y = logits + sample_gumbel(logits.size(), device=logits.device)
    return F.softmax(y / temperature, dim=-1)

def gumbel_softmax(logits, temperature=1.0, hard=False):
    y = gumbel_softmax_sample(logits, temperature)
    if hard:
        shape = y.size()
        _, ind = y.max(dim=-1)
        y_hard = torch.zeros_like(y).view(-1, shape[-1])
        y_hard.scatter_(1, ind.view(-1, 1), 1)
        y_hard = y_hard.view(*shape)
        y = (y_hard - y).detach() + y
    return y


# ---------------------------
# GENERATORS
# ---------------------------
class GeneratorX(nn.Module):
    def __init__(self, noise_dim=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(noise_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, z, temperature=0.5):
        logits = self.fc(z)
        return gumbel_softmax(logits, temperature, hard=True)


class GeneratorW(nn.Module):
    def __init__(self, input_dim=16 + 2, output_channels=1):
        super().__init__()
        self.fc = nn.Linear(input_dim, 128 * 7 * 7)
        self.deconv = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, output_channels, 4, 2, 1),
            nn.Tanh(),
        )

    def forward(self, z, x_onehot):
        zx = torch.cat([z, x_onehot], dim=1)
        out = self.fc(zx)
        out = out.view(-1, 128, 7, 7)
        img = self.deconv(out)
        return img


class GeneratorS(nn.Module):
    def __init__(self, input_dim=28 * 28):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, w_img, temperature=0.5):
        flat = w_img.view(w_img.size(0), -1)
        logits = self.fc(flat)
        return gumbel_softmax(logits, temperature, hard=True)


# ---------------------------
# DISCRIMINATORS
# ---------------------------
class DiscriminatorW(nn.Module):
    def __init__(self, input_channels=1):
        super().__init__()
        self.model = nn.Sequential(
            spectral_norm(nn.Conv2d(input_channels, 64, 4, 2, 1)),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Conv2d(64, 128, 4, 2, 1)),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            spectral_norm(nn.Linear(128 * 7 * 7, 1))
        )

    def forward(self, x):
        return self.model(x)


class DiscriminatorX(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(2, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, x_onehot):
        return self.fc(x_onehot)


class DiscriminatorS(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(2, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, s_onehot):
        return self.fc(s_onehot)


# ---------------------------
# TRAINING LOOP
# ---------------------------
def train_gan(g_x, g_w, g_s, d_x, d_w, d_s, device='cpu', epochs=20, batch_size=64, lr=1e-4):
    transform = transforms.ToTensor()
    dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    dataset = [(x, y) for x, y in dataset if y in [0, 1]]
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    opt_g = torch.optim.Adam(list(g_x.parameters()) + list(g_w.parameters()) + list(g_s.parameters()), lr=lr)
    opt_d_x = torch.optim.Adam(d_x.parameters(), lr=lr)
    opt_d_w = torch.optim.Adam(d_w.parameters(), lr=lr)
    opt_d_s = torch.optim.Adam(d_s.parameters(), lr=lr)

    loss_fn = nn.BCEWithLogitsLoss()

    g_x.to(device), g_w.to(device), g_s.to(device)
    d_x.to(device), d_w.to(device), d_s.to(device)

    for epoch in range(epochs):
        for real_imgs, labels in loader:
            real_imgs = real_imgs.to(device)
            batch_size = real_imgs.size(0)

            # Real and fake labels
            valid = torch.ones(batch_size, 1, device=device)
            fake = torch.zeros(batch_size, 1, device=device)

            # -------- Train Discriminators --------
            z = torch.randn(batch_size, 16, device=device)
            x_fake = g_x(z)
            w_fake = g_w(z, x_fake)
            s_fake = g_s(w_fake)

            # X discriminator
            d_x_real = d_x(F.one_hot(torch.randint(0, 2, (batch_size,), device=device), num_classes=2).float())
            d_x_fake = d_x(x_fake.detach())
            loss_d_x = loss_fn(d_x_real, valid) + loss_fn(d_x_fake, fake)
            opt_d_x.zero_grad(); loss_d_x.backward(); opt_d_x.step()

            # W discriminator
            d_w_real = d_w(real_imgs)
            d_w_fake = d_w(w_fake.detach())
            loss_d_w = loss_fn(d_w_real, valid) + loss_fn(d_w_fake, fake)
            opt_d_w.zero_grad(); loss_d_w.backward(); opt_d_w.step()

            # S discriminator
            s_real = F.one_hot(torch.randint(0, 2, (batch_size,), device=device), num_classes=2).float()
            d_s_real = d_s(s_real)
            d_s_fake = d_s(s_fake.detach())
            loss_d_s = loss_fn(d_s_real, valid) + loss_fn(d_s_fake, fake)
            opt_d_s.zero_grad(); loss_d_s.backward(); opt_d_s.step()

            # -------- Train Generators --------
            x_fake = g_x(z)
            w_fake = g_w(z, x_fake)
            s_fake = g_s(w_fake)

            loss_g = (
                loss_fn(d_x(x_fake), valid) +
                loss_fn(d_w(w_fake), valid) +
                loss_fn(d_s(s_fake), valid)
            )
            opt_g.zero_grad(); loss_g.backward(); opt_g.step()

        print(f"Epoch [{epoch+1}/{epochs}] | D_x: {loss_d_x.item():.4f}, D_w: {loss_d_w.item():.4f}, D_s: {loss_d_s.item():.4f}, G: {loss_g.item():.4f}")