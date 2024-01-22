"""
Module containing the decoders.
"""
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

from utils.spectral_norm_fc import spectral_norm_fc


class DecoderControlVAE(nn.Module):
    def __init__(
            self,
            img_size,
            latent_dim_z=10,
            latent_dim_w=10,
            latent_dim_cond=10,
            num_prop=2,
            num_prop_cond=2,
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
        self.hidden_dim = hidden_dim
        self.hidden_dim_prop = hidden_dim_prop
        self.img_size = img_size
        self.num_prop = num_prop
        self.num_prop_cond = num_prop_cond
        self.latent_dim_z = latent_dim_z
        self.latent_dim_w = latent_dim_w
        self.latent_dim_cond = latent_dim_cond
        self.device = device

        kernel_size = 4
        self.reshape = (hid_channels, kernel_size, kernel_size)

        # decoder for the property
        self.property_lin_list = nn.ModuleList()
        for _ in range(num_prop):
            # layers = []
            # layers.append(spectral_norm_fc(
            #     nn.Linear(1, hidden_dim_prop).to(self.device)))
            # layers.append(nn.LeakyReLU(negative_slope=0.1))
            # layers.append(spectral_norm_fc(
            #     nn.Linear(hidden_dim_prop, 1).to(self.device)))
            # layers.append(nn.Sigmoid())
            # self.property_lin_list.append(nn.Sequential(*layers))

            layers = []
            layers.append(spectral_norm_fc(
                nn.Linear(1, 16).to(self.device)))
            layers.append(nn.LeakyReLU(negative_slope=0.1))
            layers.append(spectral_norm_fc(
                nn.Linear(16, 1).to(self.device)))
            layers.append(nn.Sigmoid())
            self.property_lin_list.append(nn.Sequential(*layers))

        self.wp_lin_list = nn.ModuleList()
        for _ in range(num_prop):
            # layers = nn.Sequential(
            #     nn.Linear(self.latent_dim_w, self.hidden_dim),
            #     nn.LeakyReLU(negative_slope=0.1),
            #     nn.Linear(self.hidden_dim, self.hidden_dim),
            #     nn.LeakyReLU(negative_slope=0.1),
            #     nn.Linear(self.hidden_dim, 1)
            # ).to(device)
            # self.wp_lin_list.append(nn.Sequential(*layers))

            layers = nn.Sequential(
                nn.Linear(self.latent_dim_w, 16),
                nn.LeakyReLU(negative_slope=0.1),
                nn.Linear(16, 1)
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

        # cond prop decoder
        self.lin1_cond = spectral_norm_fc(nn.Linear(
            self.latent_dim_cond,
            self.hidden_dim).to(device))
        self.lin2_cond = spectral_norm_fc(nn.Linear(
            self.hidden_dim, self.num_prop_cond).to(device))

    def mask(self, w, w_mask):
        w = w.view(w.shape[0], 1, -1)
        w = w.repeat(1, self.num_prop, 1)

        w = w * w_mask

        wp = [self.wp_lin_list[i](w[:, i, :].to(self.device))
              for i in range(self.num_prop)]

        return torch.cat(wp, dim=-1)

    def forward(self, z, w, c, w_mask):
        batch_size = z.size(0)
        wzc = torch.cat([w, z, c], dim=-1)
        prop = []

        wp = self.mask(w, w_mask)

        # fully connected process for reconstruct the properties
        for idx in range(self.num_prop):
            w_ = wp[:, idx].view(-1, 1)
            prop.append(self.property_lin_list[idx](w_) + w_)

        c_ = F.leaky_relu(self.lin1_cond(c), negative_slope=0.1)
        prop_c = F.leaky_relu(self.lin2_cond(c_), negative_slope=0.1)

        # Fully connected layers with LeakyReLU activations
        x = F.leaky_relu(self.lin1(wzc), negative_slope=0.1)
        x = F.leaky_relu(self.lin2(x), negative_slope=0.1)
        x = F.leaky_relu(self.lin3(x), negative_slope=0.1)
        x = x.view(batch_size, *self.reshape)

        # Convolutional layers with LeakyReLU activations
        x = F.leaky_relu(self.convT1(x), negative_slope=0.1)
        x = F.leaky_relu(self.convT2(x), negative_slope=0.1)
        x = F.leaky_relu(self.convT3(x), negative_slope=0.1)

        return x, torch.cat(prop, dim=-1), wp, prop_c
