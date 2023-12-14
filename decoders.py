"""
Module containing the decoders.
"""
import numpy as np

import torch
from torch import nn
from utils.spectral_norm_fc import spectral_norm_fc


class DecoderControlVAE(nn.Module):
    def __init__(
            self,
            img_size,
            latent_dim_z=10,
            latent_dim_w=10,
            latent_dim_cond=10,
            num_prop=2,
            hid_channels=32,
            hidden_dim=512,
            hidden_dim_prop=512,
            device='cpu',
            ):
        """
        Decoder base on CorrVAE, adjusted to 44x44 ZDC images
        """
        super(DecoderControlVAE, self).__init__()

        self.hid_channels = hid_channels
        self.hiddem_dim = hidden_dim
        self.hidden_dim_prop = hidden_dim_prop
        self.img_size = img_size
        self.num_prop = num_prop
        self.latent_dim_z = latent_dim_z
        self.latent_dim_w = latent_dim_w
        self.latent_dim_cond = latent_dim_cond
        self.device = device

        kernel_size = 4
        self.reshape = (hid_channels, kernel_size, kernel_size)

        # decoder for the property
        self.property_lin_list = nn.ModuleList()
        for _ in range(num_prop):
            layers = []
            layers.append(spectral_norm_fc(
                nn.Linear(1, hidden_dim_prop).to(self.device)))
            layers.append(nn.ReLU())
            layers.append(spectral_norm_fc(
                nn.Linear(hidden_dim_prop, 1).to(self.device)))
            layers.append(nn.Sigmoid())
            self.property_lin_list.append(nn.Sequential(*layers))

        self.wp_lin_list = nn.ModuleList()
        for _ in range(num_prop):
            layers = nn.Sequential(
                nn.Linear(self.latent_dim_w, self.hiddem_dim),
                nn.ReLU(),
                nn.Linear(self.hiddem_dim, self.hiddem_dim),
                nn.ReLU(),
                nn.Linear(self.hiddem_dim, 1)
                ).to(device)
            self.wp_lin_list.append(nn.Sequential(*layers))

        self.lin1 = nn.Linear(
            self.latent_dim_z + self.latent_dim_w + self.latent_dim_cond,
            hidden_dim).to(self.device)
        self.lin2 = nn.Linear(hidden_dim, hidden_dim).to(self.device)
        self.lin3 = nn.Linear(
            hidden_dim, np.product(self.reshape)).to(self.device)

        cnn_kwargs = dict(
            kernel_size=kernel_size,
            stride=2,
            padding=1
        )

        if self.img_size[1] == self.img_size[2] == 64:
            self.convT_64 = nn.ConvTranspose2d(
                hid_channels, hid_channels, **cnn_kwargs).to(device)

        self.convT1 = nn.ConvTranspose2d(
            hid_channels, hid_channels, kernel_size,
            stride=(3, 3), padding=(1, 1)).to(device)
        self.convT2 = nn.ConvTranspose2d(
            hid_channels, hid_channels, **cnn_kwargs).to(device)
        self.convT3 = nn.ConvTranspose2d(
            hid_channels, self.img_size[0], **cnn_kwargs).to(device)

    def mask(self, w, w_mask):
        w = w.view(w.shape[0], 1, -1)
        w = w.repeat(1, self.num_prop, 1)

        w = w * w_mask

        wp = [self.wp_lin_list[i](w[:, i, :].to(self.device))
              for i in range(self.num_prop)]

        return torch.cat(wp, dim=-1)

    def forward(self, z, w, cond, w_mask):
        batch_size = z.size(0)
        wzc = torch.cat([w, z, cond], dim=-1)
        prop = []

        wp = self.mask(w, w_mask)

        # fully connected process for reconstruct the properties
        for idx in range(self.num_prop):
            w_ = wp[:, idx].view(-1, 1)
            prop.append(self.property_lin_list[idx](w_) + w_)

        # Fully connected layers with ReLu activations
        x = torch.relu(self.lin1(wzc))
        x = torch.relu(self.lin2(x))
        x = torch.relu(self.lin3(x))
        x = x.view(batch_size, *self.reshape)

        # Convolutional layers with ReLu activations
        if self.img_size[1] == self.img_size[2] == 64:
            x = torch.relu(self.convT_64(x))
        x = torch.relu(self.convT1(x))
        x = torch.relu(self.convT2(x))

        # final layer
        x = torch.relu(self.convT3(x))

        return x, torch.cat(prop, dim=-1), wp
