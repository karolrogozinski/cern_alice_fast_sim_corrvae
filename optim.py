# -*- coding: utf-8 -*-
"""
Created on Sun Jan 16 15:07:56 2022

@author: Shiyu
"""

import torch
from torch import nn


class Optimization(nn.Module):
    def __init__(self, dim_w, model, num_prop, device='cpu'):
        super(Optimization, self).__init__()
        self.dim_w = dim_w
        self.num_prop = num_prop

        self.device = device

        self.w = nn.Parameter(torch.randn(1, self.dim_w), requires_grad=True)
        self.decoder = model.decoder
        self.decoder.requires_grad = False
        self.lambda1 = nn.Parameter(torch.randn(1), requires_grad=True)
        self.lambda2 = nn.Parameter(torch.randn(1), requires_grad=True)
        self.lambda3 = nn.Parameter(torch.randn(1), requires_grad=True)
        self.lambda4 = nn.Parameter(torch.randn(1), requires_grad=True)

    def forward(self, prop, mask):
        w = self.w.repeat(self.num_prop, 1).to(self.device)
        w = w * mask

        wp = []
        for idx in range(self.num_prop):
            wp.append(self.decoder.wp_lin_list[idx](w[idx, :].to(self.device)))

        prop_pred = []
        for idx in range(self.num_prop):
            # TODO why do we add w_ to prediction?
            w_ = wp[idx].view(-1, 1)
            prop_pred.append(self.decoder.property_lin_list[idx](w_) + w_)

        prop_pred = torch.cat(prop_pred, dim=-1)

        loss_range1 = self.lambda1.to(self.device) * (0.5 - prop_pred[0, 1]) +\
            self.lambda2.to(self.device) * (prop_pred[0, 1] - 0.8)

        # TODO rewrite this in as loop based on prop num
        
        # loss_value1 = torch.abs(prop[0] - prop_pred[0, 0]) + torch.abs(
        #     prop[1] - prop_pred[0, 1]) + torch.abs(prop[2] - prop_pred[0, 2])
        # loss_value2 = torch.abs(
        #     prop[0] - prop_pred[0, 0]) + torch.abs(prop[2] - prop_pred[0, 2])
        # loss_inf = - prop_pred[0, 2]

        loss_value1 = torch.abs(prop[0] - prop_pred[0, 0]) + torch.abs(
            prop[1] - prop_pred[0, 1])
        loss_value2 = torch.abs(
            prop[0] - prop_pred[0, 0])
        loss_inf = - prop_pred[0, 1]

        return (prop_pred, loss_range1, loss_value1, loss_value2, loss_inf,
                self.lambda1, self.lambda2, self.lambda3, self.lambda4, self.w)
