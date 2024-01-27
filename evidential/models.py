import torch
import math
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class EvidentialModule(nn.Module):
    def __init__(self, depth):
        super(EvidentialModule, self).__init__()
        '''
        # Layout 3D
        #_______________________________________________________________________________________________

        # Initial 3D Convolution Layer to process depth
        self.conv3d1 = nn.Conv3d(1, 16, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn3d1 = nn.BatchNorm3d(16)
        self.relu = nn.ReLU(inplace=True)

        # Convert 3D feature maps to 2D
        self.conv3d_to_2d = nn.Conv3d(16, 16, kernel_size=(100, 1, 1), stride=(100, 1, 1))

        # 2D Convolution Layers to process spatial information
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        # Output layer for EDL parameters
        self.edl_params = nn.Conv2d(64, 4, kernel_size=1)  # 4 channels for EDL parameters

        # Layout 2D
        #_______________________________________________________________________________________________
        # Convolutional layers
        self.conv1 = nn.Conv2d(100, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(16)
        self.conv4 = nn.Conv2d(16, 8, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(8)

        # Output layer
        self.output_layer = nn.Conv2d(8, 4, kernel_size=1)
        '''
        # Layout Mixed
        #_______________________________________________________________________________________________
        self.conv3d1 = nn.Conv3d(1, 16, kernel_size=(5, 3, 3), padding=(2, 1, 1))
        self.bn3d1 = nn.BatchNorm3d(16)

        # Transition from 3D to 2D, we'll use a 1x1x1 conv to reduce the depth dimension without losing spatial dimensions
        self.conv3d_to_2d = nn.Conv3d(16, 4, kernel_size=(100, 1, 1))

        # 2D convolutional layers to process the spatial information
        self.conv2d1 = nn.Conv2d(4, 16, kernel_size=3, padding=1)
        self.bn2d1 = nn.BatchNorm2d(16)
        self.conv2d2 = nn.Conv2d(16, 8, kernel_size=3, padding=1)
        self.bn2d2 = nn.BatchNorm2d(8)

        # Output layer to produce 4 EDL parameters per pixel
        self.output_layer = nn.Conv2d(8, 4, kernel_size=1)


    def forward(self, x):
        # Layout 3D
        #_______________________________________________________________________________________________
        '''

        x = x.unsqueeze(1)  # Add channel dimension if not present
        # batch size, chanel, depth, height, width

        # Existing conv layers
        # 3D Convolution to process depth
        x = self.relu(self.bn3d1(self.conv3d1(x)))

        # Convert 3D feature maps to 2D by collapsing the depth dimension
        x = self.conv3d_to_2d(x)
        x = torch.squeeze(x, 2)  # Remove the depth dimension

        # 2D Convolution Layers
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))

        # Output EDL parameters
        x = self.edl_params(x)  # Output shape: [batch_size, 4, 128, 160]


        # Layout 2D
        #_______________________________________________________________________________________________
        # Apply convolutional layers with ReLU and batch normalization
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))

        # Output layer to get 4 EDL parameters per pixel
        x = self.output_layer(x)
        '''
        # Layout Mixed
        #_______________________________________________________________________________________________
        # Add a channel dimension for 3D convolution (N, C, D, H, W)
        x = x.unsqueeze(1)

        # Apply 3D convolution and batch normalization
        x = F.relu(self.bn3d1(self.conv3d1(x)))

        # Transition from 3D to 2D by reducing depth to 4
        x = self.conv3d_to_2d(x)
        # Remove the depth dimension, results in (N, C, H, W)
        x = x.squeeze(2)

        # Apply 2D convolutions and batch normalization
        x = F.relu(self.bn2d1(self.conv2d1(x)))
        x = F.relu(self.bn2d2(self.conv2d2(x)))

        # Output layer to get 4 EDL parameters per pixel
        x = self.output_layer(x)


        x = torch.squeeze(x, dim=0)
        #x = torch.squeeze(x, dim=1)

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

