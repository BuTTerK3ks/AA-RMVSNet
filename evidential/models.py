import torch
import math
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class EvidentialModule(nn.Module):
    def __init__(self, depth):
        super(EvidentialModule, self).__init__()
        '''
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
        '''

        # Initial Convolution Layer
        self.conv1 = nn.Conv3d(1, 16, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn1 = nn.BatchNorm3d(16)
        self.relu = nn.ReLU()

        # Additional Convolution Layers
        self.conv2 = nn.Conv3d(16, 32, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn2 = nn.BatchNorm3d(32)

        self.conv3 = nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn3 = nn.BatchNorm3d(64)

        # Reduce depth and adjust channels
        self.conv_depth_reduction = nn.Conv3d(64, 4, kernel_size=(100, 1, 1), stride=1)
        self.bn = nn.BatchNorm3d(4)

    def forward(self, x):
        # Assuming x is the input tensor with shape [Batch, Channels, Depth, Height, Width]
        # Ensure input tensor has 5 dimensions, with the second dimension being 1 (for single-channel input)

        # enable anomaly detection
        torch.autograd.set_detect_anomaly(True)

        x = x.unsqueeze(1)  # Add channel dimension if not present
        # batch size, chanel, depth, height, width

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.conv_depth_reduction(x)
        x = self.bn(x)
        x = self.relu(x)

        x = torch.squeeze(x, dim=0)
        x = torch.squeeze(x, dim=1)



        '''
        # Use cascade of Convolutional and ELU layers
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.relu3(self.conv3(x))
        x = self.conv4(x)
        '''

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

        # Apply sigmoid to the first channel
        first_channel_sigmoid = torch.sigmoid(x[0:1, :, :])  # Keeps the tensor 3D

        x_softplus = F.softplus(x)

        x_modified = torch.cat((first_channel_sigmoid, x_softplus[1:, :, :]), dim=0)
        y = x_modified.clone()

        y[2, :, :] += 1  # Modify the copy

        return y


def loss_der(outputs, depth_gt, mask, depth_value, coeff=0.01):

    evidential_prediction = outputs['evidential_prediction']
    probability_volume = outputs['probability_volume']

    # get EDL parameters
    gamma, nu, alpha, beta = evidential_prediction[0, :, :], evidential_prediction[1, :, :], evidential_prediction[2, :, :], evidential_prediction[3, :, :]
    gamma = torch.unsqueeze(gamma, 0)
    nu = torch.unsqueeze(nu, 0)
    alpha = torch.unsqueeze(alpha, 0)
    beta = torch.unsqueeze(beta, 0)


    #highest_prob = torch.argmax(gamma, dim=1).type(torch.long)
    #depth_map = torch.take(depth_value, highest_prob)
    depth = depth_value.size(1)
    indices = (gamma * (depth - 1)).long()
    #selected_depth_values = torch.gather(depth_value, 1, indices.expand(-1, 1, -1, -1))
    selected_depth_values = torch.take(depth_value, indices)

    depth_map = selected_depth_values

    torch.set_printoptions(profile="full")

    error = depth_map - depth_gt

    omega = 2.0 * beta * (1.0 + nu)
    # Formula 8 from Deep Evidential Regression Paper
    calculated_loss = 0.5 * torch.log(math.pi / nu) - alpha * torch.log(omega) + (alpha + 0.5) * torch.log(error ** 2 * nu + omega) + torch.lgamma(alpha) - torch.lgamma(alpha + 0.5) + coeff * torch.abs(error) * (2.0 * nu + alpha)




    # mask loss and weight regarding the effective amount of valid pixels
    masked_loss = calculated_loss * mask
    valid_pixel_num = torch.sum(mask, dim=[1, 2]) + 1e-6
    loss = torch.sum(masked_loss)/valid_pixel_num

    # get aleatoric and epistemic uncertainty
    aleatoric = torch.sqrt(beta * (nu + 1) / nu / alpha)
    epistemic = 1. / torch.sqrt(nu)


    return loss, depth_map, aleatoric, epistemic

