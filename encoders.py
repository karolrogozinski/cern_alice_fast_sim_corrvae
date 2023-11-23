"""
Module containing the encoders.
"""
import numpy as np

import torch
from torch import nn


class EncoderControlVAE(nn.Module):
    def __init__(
            self,
            img_size,
            latent_dim_z=10,
            latent_dim_w=10,
            hidden_dim=256,
            hid_channels=32,
            device='cpu',
            ):
        """
        Encoder based on CorrVAE, adjusted to 44x44 ZDC images
        """
        super(EncoderControlVAE, self).__init__()

        self.hid_channels = hid_channels
        self.hidden_dim = hidden_dim
        self.latent_dim_z = latent_dim_z
        self.latent_dim_w = latent_dim_w
        self.img_size = img_size

        kernel_size = 4
        cnn_kwargs = dict(
            kernel_size=kernel_size,
            stride=2,
            padding=1,
        )
        self.reshape = (self.hid_channels, kernel_size, kernel_size)

        self.conv1 = nn.Conv2d(
            self.img_size[0], self.hid_channels, **cnn_kwargs).to(device)
        self.conv2 = nn.Conv2d(
            self.hid_channels, self.hid_channels, **cnn_kwargs).to(device)
        self.conv3 = nn.Conv2d(
            hid_channels, hid_channels, kernel_size,
            stride=(3, 3), padding=(1, 1)).to(device)

        self.lin1 = nn.Linear(
            np.product(self.reshape), self.hidden_dim).to(device)
        self.lin2 = nn.Linear(self.hidden_dim, self.hidden_dim).to(device)

        # Fully connected layers for mean and variance
        self.mu_logvar_gen = nn.Linear(
            self.hidden_dim,
            (self.latent_dim_z+self.latent_dim_w) * 2).to(device)

    def forward(self, x):
        batch_size = x.size(0)

        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))

        x_z = x.view((batch_size, -1))
        x_z = torch.relu(self.lin1(x_z))
        x_z = torch.relu(self.lin2(x_z))

        mu_logvar = self.mu_logvar_gen(x_z)
        mu, logvar = mu_logvar.view(
            -1, self.latent_dim_z+self.latent_dim_w, 2).unbind(-1)

        return (
            mu[:, :self.latent_dim_z],
            mu[:, self.latent_dim_z:],
            logvar[:, :self.latent_dim_z],
            logvar[:, self.latent_dim_z:]
        )
