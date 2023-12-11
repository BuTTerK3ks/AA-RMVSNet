import torch
import math
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

def normalize_target(target, target_min, target_max):
    return (target - target_min) / (target_max - target_min)

def denormalize_target(normalized_target, target_min, target_max):
    return normalized_target * (target_max - target_min) + target_min

def map_to_0_1(tensor, min_value, max_value):
    mapped_tensor = (tensor - min_value) / (max_value - min_value)
    return mapped_tensor

def map_to_original_range(mapped_tensor, min_value, max_value):
    original_tensor = mapped_tensor * (max_value - min_value) + min_value
    return original_tensor

class EvidentialModule(nn.Module):
    def __init__(self):
        super(EvidentialModule, self).__init__()

        # gamma, nu, alpha, beta
        self.convolution = nn.Conv2d(100, 4, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.convolution(x)

        y = torch.zeros_like(x)
        y[:, 0, :, :] = x[:, 0, :, :]
        y[:, 1:4, :, :] = F.softplus(x[:, 1:4, :, :])
        # Add +1 to alpha channel
        x = y
        x[:, 1, :, :] = torch.add(y[:, 1, :, :], 1)

        return x



def loss_der(prediction, depth_gt, mask, depth_value, coeff=0.01, use_mask=True):


    gamma, nu, alpha, beta = prediction[:, 0, :, :], prediction[:, 1, :, :], prediction[:, 2, :, :], prediction[:, 3, :, :]
    depth = map_to_0_1(depth_gt, 425, 687.35)
    #depth = depth_gt
    error = gamma - depth
    omega = 2.0 * beta * (1.0 + nu)

    calculated_loss = 0.5 * torch.log(math.pi / nu) - alpha * torch.log(omega) + (alpha + 0.5) * torch.log(error ** 2 * nu + omega) + torch.lgamma(alpha) - torch.lgamma(alpha + 0.5) + coeff * torch.abs(error) * (2.0 * nu + alpha)
    if use_mask:
        masked_loss = torch.mul(mask, calculated_loss)  # valid pixel
        loss = torch.mean(masked_loss)
    else:
        loss = torch.mean(calculated_loss)

    aleatoric = torch.sqrt(beta * (nu + 1) / nu / alpha)
    epistemic = 1. / torch.sqrt(nu)
    #TODO Change for other dataset
    depth_est = map_to_original_range(gamma, 425, 687.35)
    #depth_est = gamma



    # TODO check if right
    return loss, depth_est, aleatoric, epistemic