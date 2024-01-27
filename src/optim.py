import torch
from torch import nn
from torch.nn import functional as F


class Optimization(nn.Module):
    def __init__(self, dim_w, model, num_prop, batch_size):
        super(Optimization, self).__init__()
        self.dim_w = dim_w
        self.num_prop = num_prop

        self.w = nn.Parameter(torch.randn(batch_size, self.dim_w),
                              requires_grad=True)

        for param in model.decoder.parameters():
            param.requires_grad = False
        self.decoder = model.decoder

    def forward(self, prop, mask):
        w = self.w.view(self.w.shape[0], 1, -1)
        w = w.repeat(1, self.num_prop, 1)

        w = w * mask

        wp = []
        for idx in range(self.num_prop):
            wp.append(self.decoder.wp_lin_list[idx](w[:, idx, :]))

        prop_pred = []
        for idx in range(self.num_prop):
            w_ = wp[idx].view(-1, 1)
            prop_pred.append(self.decoder.property_lin_list[idx](w_)+w_)

        prop_pred = torch.cat(prop_pred, dim=-1)
        loss_value1 = F.mse_loss(prop, prop_pred, reduction='sum')
        loss_value1 /= self.w.shape[0]

        return prop_pred, loss_value1, self.w, prop_pred, wp
