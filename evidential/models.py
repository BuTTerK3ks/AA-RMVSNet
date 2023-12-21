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

def map_to_original_range_0_1(mapped_tensor, min_value, max_value):
    original_tensor = mapped_tensor * (max_value - min_value) + min_value
    return original_tensor


class EvidentialModule(nn.Module):
    def __init__(self):
        super(EvidentialModule, self).__init__()

        # nu, alpha, beta
        self.convolution = nn.Conv2d(100, 3, kernel_size=1, stride=1, padding=0)

    def forward(self, probability_volume):
        y = self.convolution(probability_volume)

        z = torch.zeros_like(y)
        z = F.softplus(y)
        # Add +1 to alpha channel
        x = torch.zeros_like(z)
        x[:, 0, :, :] = z[:, 0, :, :]
        x[:, 1, :, :] = torch.add(z[:, 1, :, :], 1)
        x[:, 2, :, :] = z[:, 2, :, :]

        return x


def loss_der(outputs, depth_gt, mask, depth_value, coeff=0.01):

    evidential_prediction = outputs['evidential_prediction']
    probability_volume = outputs['probability_volume']

    # take max probability and get depth
    probability_map = torch.argmax(probability_volume, dim=1).type(torch.long)
    depth_map = torch.take(depth_value, probability_map)

    nu, alpha, beta = evidential_prediction[:, 0, :, :], evidential_prediction[:, 1, :, :], evidential_prediction[:, 2, :, :]

    # map errors to relative range [0,1]
    t_min = torch.min(depth_value.flatten())
    t_max = torch.max(depth_value.flatten())
    depth_map_normalized = normalize_target(depth_map, torch.min(depth_value.flatten()), torch.max(depth_value.flatten())) * mask
    depth_gt_normalized = normalize_target(depth_gt, torch.min(depth_value.flatten()), torch.max(depth_value.flatten())) * mask
    error = depth_map_normalized - depth_gt_normalized

    omega = 2.0 * beta * (1.0 + nu)

    calculated_loss = 0.5 * torch.log(math.pi / nu) - alpha * torch.log(omega) + (alpha + 0.5) * torch.log(error ** 2 * nu + omega) + torch.lgamma(alpha) - torch.lgamma(alpha + 0.5) + coeff * torch.abs(error) * (2.0 * nu + alpha)
    #TODO Check if masked loss is right
    masked_loss = calculated_loss * mask
    valid_pixel_num = torch.sum(mask, dim=[1, 2]) + 1e-6
    loss = torch.sum(masked_loss / valid_pixel_num)



    aleatoric = torch.sqrt(beta * (nu + 1) / nu / alpha)
    epistemic = 1. / torch.sqrt(nu)


    return loss, depth_map, aleatoric, epistemic

