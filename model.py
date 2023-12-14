# -*- coding: utf-8 -*-
"""
Created on Sat Dec 11 21:17:20 2021

@author: Shiyu
"""
import torch
from torch import nn
from torch.nn import functional as F

from utils.model_init import weights_init


class ControlVAE(nn.Module):
    def __init__(
            self,
            img_size,
            encoder,
            decoder,
            latent_dim,
            latent_dim_prop,
            latent_dim_cond,
            num_prop,
            hid_channels=32,
            device='cpu'):
        """
        Class which defines model and forward pass.

        Parameters
        ----------
        img_size : tuple of ints
            Size of images. E.g. (1, 32, 32) or (3, 64, 64).
        encoder: encoder
        decoder: decoder
        latent_dim: latent dimension
        num_prop: number of properties
        device: device
        """
        super(ControlVAE, self).__init__()

        self.num_prop = num_prop
        self.latent_dim_z = latent_dim
        self.latent_dim_w = latent_dim_prop
        self.latent_dim_cond = latent_dim_cond

        self.img_size = img_size
        self.num_pixels = self.img_size[1] * self.img_size[2]
        self.device = device

        self.encoder = encoder(img_size, self.latent_dim_z,
                               self.latent_dim_w, self.latent_dim_cond,
                               hid_channels, device=device)
        self.decoder = decoder(img_size, self.latent_dim_z,
                               self.latent_dim_w, self.latent_dim_cond,
                               self.num_prop, hid_channels, device=device)

        self.apply(weights_init)
        self.w_mask = torch.nn.Parameter(
            torch.randn(self.num_prop, self.latent_dim_w, 2))

    def reparameterize(self, mean, logvar):
        """
        Samples from a normal distribution using the reparameterization trick.

        Parameters
        ----------
        mean : torch.Tensor
            Mean of the normal distribution. Shape (batch_size, latent_dim)

        logvar : torch.Tensor
            Diagonal log variance of the normal distribution. Shape (batch_size
            latent_dim)
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + std * eps

    def forward(self, x, cond, tau,
                mask=None, w2=None, z2=None, w_mask=None, label=None):
        """
        Forward pass of model.

        Parameters
        ----------
        x : torch.Tensor
            Batch of data. Shape (batch_size, n_chan, height, width)
        """

        latent_dist_z_mean, latent_dist_w_mean, \
            latent_dist_z_std, latent_dist_w_std = self.encoder(x, cond)

        latent_sample_z = self.reparameterize(
            latent_dist_z_mean, latent_dist_z_std)
        latent_sample_w = self.reparameterize(
            latent_dist_w_mean, latent_dist_w_std)

        if w2 is not None:
            latent_sample_w = w2.repeat(latent_sample_z.shape[0], 1)

        if z2 is not None:
            latent_sample_z = z2

        if mask is None:
            logit = torch.sigmoid(self.w_mask) / (
                1 - torch.sigmoid(self.w_mask))
            mask = F.gumbel_softmax(
                logit.to(self.device), tau, hard=True)[:, :, 1]

        reconstruct, y_reconstruct, _ = self.decoder(
            latent_sample_z, latent_sample_w, cond, mask)

        latent_dist_z = (latent_dist_z_mean, latent_dist_z_std)
        latent_dist_w = (latent_dist_w_mean, latent_dist_w_std)
        return (reconstruct, y_reconstruct), latent_dist_z, latent_dist_w, \
            latent_sample_z, latent_sample_w, mask, cond, self.w_mask

    def sample_latent(self, x):
        """
        Returns a sample from the latent distribution.

        Parameters
        ----------
        x : torch.Tensor
            Batch of data. Shape (batch_size, n_chan, height, width)
        """
        latent_dist_z_mean, latent_dist_w_mean, \
            latent_dist_z_std, latent_dist_w_std, _ = self.encoder(x)
        latent_dist_z = (latent_dist_z_mean, latent_dist_z_std)
        latent_dist_w = (latent_dist_w_mean, latent_dist_w_std)
        latent_sample_z = self.reparameterize(*latent_dist_z)
        latent_sample_w = self.reparameterize(*latent_dist_w)

        return latent_sample_z, latent_sample_w

    def iterate_get_w(self, label, w_latent_idx, maxIter=20):
        """
        Get the w for a kind of given property

        Note:
        It's not the w from laten space but the w' reversed from y.
        Dim is same as y!
        """
        w_n = label.view(-1, 1).to(self.device).float()  # [N]
        for _ in range(maxIter):
            summand = self.decoder.property_lin_list[w_latent_idx](w_n)
            w_n1 = label.view(-1, 1).to(self.device).float() - summand
            print('Iteration of difference:' +
                  str(torch.abs(w_n-w_n1).mean().item()))
            w_n = w_n1.clone()
        return w_n1.view(-1)
