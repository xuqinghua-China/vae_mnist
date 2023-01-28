"""
    @Time    : 26/01/2023
    @Author  : qinghua
    @Software: PyCharm
    @File    : train.py

"""
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import AutoEncoder, VAE
import torch.nn as nn
import torch.optim as optim
import torchvision
import matplotlib.pyplot as plt
import seaborn as sn
import numpy as np


def train(model, train_dataloader, opitimizer, writer, valid_dataloader, valid_every=10):
    global global_step
    for x, y in train_dataloader:
        optimizer.zero_grad()
        x_hat = model(x)
        if model.name == "ae":
            loss = torch.sum((x_hat - x) ** 2)
        else:
            loss = torch.sum((x_hat-x)**2)+model.kl
        loss.backward()
        opitimizer.step()
        writer.add_scalar("Train/loss", loss.item(), global_step)
        global_step += 1
        if global_step % valid_every == 0:
            valid(model, valid_dataloader, writer)
            generate(model, writer)


def valid(model, valid_dataloader, writer):
    global epoch_i
    all_hidden, all_y = [], []
    plt.figure()
    with torch.no_grad():
        for x, y in valid_dataloader:
            hidden = model.encoder(x)  # [batch_size, hidden_dim]
            all_hidden.append(hidden)
            all_y.append(y)
        hidden = torch.cat(all_hidden, dim=0)
        y = torch.cat(all_y, dim=0)
        y = y.detach().numpy()
        hidden = hidden.detach().numpy()
        fig=plt.figure()
        plt.scatter(x=hidden[:, 0], y=hidden[:, 1], c=y, cmap="tab10").get_figure()
        plt.colorbar()
        writer.add_figure("Valid/Hidden", fig, epoch_i)
        plt.close(fig)


def generate(model, writer, n=12):
    global epoch_i
    left, right, bottom, top = -5, 10, -10, 5
    w = 28
    image_matrix = np.zeros((w * n, w * n))
    decoder = model.decoder
    for i, y in enumerate(np.linspace(bottom, top, n)):
        for j, x in enumerate(np.linspace(left, right, n)):
            hidden = torch.tensor([x, y], dtype=torch.float)
            x_hat = decoder(hidden).reshape(w, w).detach().numpy()
            image_matrix[(n - 1 - i) * w:(n - 1 - i + 1) * w, j * w:(j + 1) * w] = x_hat
    fig = plt.figure()
    plt.imshow(image_matrix, extent=[left, right, bottom, top])
    writer.add_figure("Valid/Generate", fig, epoch_i)
    plt.close(fig)


if __name__ == '__main__':
    hidden_dim = 2
    # model = AutoEncoder(hidden_dim)
    model = VAE(hidden_dim)
    optimizer = optim.Adam(model.parameters())
    data = torchvision.datasets.MNIST("data", download=True, transform=torchvision.transforms.ToTensor())
    train_data, test_data = torch.utils.data.random_split(data, [0.9, 0.1])
    train_dataloader = DataLoader(train_data, batch_size=128, shuffle=True)
    valid_dataloader = DataLoader(test_data, batch_size=16)
    writer = SummaryWriter()
    n_epochs = 10
    global_step = 0
    for epoch_i in range(n_epochs):
        print("Training for epoch ", epoch_i)
        train(model, train_dataloader, optimizer, writer, valid_dataloader)
