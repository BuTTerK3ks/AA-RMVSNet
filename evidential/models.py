import torch
import math
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class EvidentialModule(nn.Module):
    def __init__(self):
        super(EvidentialModule, self).__init__()

        # gamma, nu, alpha, beta
        self.convolution = nn.Conv2d(100, 4, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.convolution(x)

        y = torch.zeros_like(x)
        y[:, 0, :, :] = x[:, 0, :, :]
        y[:, 1:4, :, :] = F.softplus(x[:, 1:4, :, :])
        # Add +1 to alpha channel
        x = y
        x[:, 1, :, :] = torch.add(y[:, 1, :, :], 1)

        return x


def loss_der(prediction, depth_gt, mask, depth_value, coeff=0.01):


    gamma, nu, alpha, beta = prediction[:, 0, :, :], prediction[:, 1, :, :], prediction[:, 2, :, :], prediction[:, 3, :, :]
    error = gamma - depth_gt
    omega = 2.0 * beta * (1.0 + nu)

    calculated_loss = 0.5 * torch.log(math.pi / nu) - alpha * torch.log(omega) + (alpha + 0.5) * torch.log(error ** 2 * nu + omega) + torch.lgamma(alpha) - torch.lgamma(alpha + 0.5) + coeff * torch.abs(error) * (2.0 * nu + alpha)
    calculated_loss = torch.mean(calculated_loss)

    # TODO check if right
    return calculated_loss, gamma