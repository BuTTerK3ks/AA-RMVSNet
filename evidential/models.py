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
    def __init__(self, depth):
        super(EvidentialModule, self).__init__()
        # one layer
        # nu, alpha, beta
        # First convolutional layer
        self.conv1 = nn.Conv2d(depth, 256, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ELU()

        # Second convolutional layer
        self.conv2 = nn.Conv2d(256, 128, kernel_size=5, stride=1, padding=2)
        self.relu2 = nn.ELU()

        # Third convolutional layer
        self.conv3 = nn.Conv2d(128, 256, kernel_size=9, stride=1, padding=4)
        self.relu3 = nn.ELU()

        # Final layer to bring the channel size to 3
        self.conv4 = nn.Conv2d(256, 3, kernel_size=1, stride=1, padding=0)

        # 1D Conv layer
        self.conv1d = nn.Conv1d(in_channels=100, out_channels=3, kernel_size=1)

        self.linear = nn.Sequential(
            nn.Linear(100, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 3),
        )


    def forward(self, probability_volume):
        #y = self.convolution(probability_volume)

        x = probability_volume

        # Use cascade of Convolutional and ELU layers
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.relu3(self.conv3(x))
        x = self.conv4(x)


        '''
        # Use fully connected layer
        x = x.view(1, 128*160, 100).transpose(1, 2)
        x = self.conv1d(x)
        x = x.transpose(1, 2).reshape(1, 3, 128, 160)
        '''

        '''
        # Use linear layer
        x = self.linear(x)
        '''

        z = F.softplus(x)

        # Add +1 to alpha channel
        x = torch.zeros_like(z)
        # nu
        x[:, 0, :, :] = z[:, 0, :, :]
        # alpha
        x[:, 1, :, :] = torch.add(z[:, 1, :, :], 1)
        # beta
        x[:, 2, :, :] = z[:, 2, :, :]

        return x


def loss_der(outputs, depth_gt, mask, depth_value, coeff=0.01):

    evidential_prediction = outputs['evidential_prediction']
    probability_volume = outputs['probability_volume']

    # take max probability and get depth
    probability_map = torch.argmax(probability_volume, dim=1).type(torch.long)
    depth_map = torch.take(depth_value, probability_map)

    # get EDL parameters
    nu, alpha, beta = evidential_prediction[:, 0, :, :], evidential_prediction[:, 1, :, :], evidential_prediction[:, 2, :, :]

    torch.set_printoptions(profile="full")

    error = (depth_map - depth_gt) * mask
    #error = normalize_target(error, torch.min(error.flatten()), torch.max(error.flatten()))

    omega = 2.0 * beta * (1.0 + nu)
    # Formula 8 from Deep Evidential Regression Paper
    calculated_loss = 0.5 * torch.log(math.pi / nu) - alpha * torch.log(omega) + (alpha + 0.5) * torch.log(error ** 2 * nu + omega) + torch.lgamma(alpha) - torch.lgamma(alpha + 0.5) + coeff * torch.abs(error) * (2.0 * nu + alpha)




    # mask loss and weight regarding the effective amount of valid pixels
    masked_loss = calculated_loss * mask
    valid_pixel_num = torch.sum(mask, dim=[1, 2]) + 1e-6
    loss = torch.sum(masked_loss / valid_pixel_num)

    # get aleatoric and epistemic uncertainty
    aleatoric = torch.sqrt(beta * (nu + 1) / nu / alpha)
    epistemic = 1. / torch.sqrt(nu)


    return loss, depth_map, aleatoric, epistemic

