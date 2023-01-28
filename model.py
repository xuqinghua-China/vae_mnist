"""
    @Time    : 26/01/2023
    @Author  : qinghua
    @Software: PyCharm
    @File    : model.py

"""
import torchvision
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, hidden_dim=2):
        super(Encoder, self).__init__()
        self.linear1 = nn.Linear(28 * 28, 512)
        self.linear2 = nn.Linear(512, hidden_dim)

    def forward(self, x):
        """x:[N,1,28,28]"""
        x = torch.flatten(x, start_dim=1)  # [N,764]
        x = self.linear1(x)  # [N, 512]
        x = F.relu(x)
        return self.linear2(x)  # [N,2]


class Decoder(nn.Module):
    def __init__(self, hidden_dim):
        super(Decoder, self).__init__()
        self.linear1 = nn.Linear(hidden_dim, 512)
        self.linear2 = nn.Linear(512, 28 * 28)

    def forward(self, x):
        """x:[N,2]"""
        hidden = self.linear1(x)  # [N, 512]
        hidden = torch.relu(hidden)
        hidden = self.linear2(hidden)  # [N,764]
        hidden = torch.sigmoid(hidden)
        return torch.reshape(hidden, (-1, 1, 28, 28))


class AutoEncoder(nn.Module):
    def __init__(self, hidden_dim=2):
        super(AutoEncoder, self).__init__()
        self.name = "ae"
        self.encoder = Encoder(hidden_dim)
        self.decoder = Decoder(hidden_dim)

    def forward(self, x):
        return self.decoder(self.encoder(x))


class VAEEncoder(nn.Module):
    def __init__(self, hidden_dim=2):
        super(VAEEncoder, self).__init__()
        self.linear1 = nn.Linear(28 * 28, 512)
        self.linear2 = nn.Linear(512, hidden_dim)
        self.linear3 = nn.Linear(512, hidden_dim)
        self.noise_dist = torch.distributions.Normal(0, 1)
        self.kl = 0

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = self.linear1(x)
        x = torch.relu(x)
        mu = self.linear2(x)
        sigma = torch.exp(self.linear3(x))
        hidden = mu + self.noise_dist.sample(mu.shape) * sigma
        self.kl = (sigma ** 2 + mu ** 2 - torch.log(sigma) - 1 / 2).sum()
        return hidden


class VAE(nn.Module):
    def __init__(self, hidden_dim=2):
        super(VAE, self).__init__()
        self.name = "vae"
        self.encoder = VAEEncoder(hidden_dim=hidden_dim)
        self.decoder = Decoder(hidden_dim)
        self.kl = 0

    def forward(self, x):
        hidden = self.encoder(x)
        self.kl = self.encoder.kl
        return self.decoder(hidden)


if __name__ == '__main__':
    dataset = torchvision.datasets.MNIST("data", transform=torchvision.transforms.ToTensor(), download=True)
    print(dataset[0][0].shape)
