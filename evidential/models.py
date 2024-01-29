import torch
import math
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return self.relu(x)

class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class EDLNet(nn.Module):
    def __init__(self):
        super(EDLNet, self).__init__()
        # Define initial convolutions, SEBlocks, and other necessary layers
        self.conv1 = ConvBlock(1, 64)
        self.se1 = SEBlock(64)
        # Continue defining layers, including downsample and upsample paths for U-Net structure
        # Final layers for EDL parameter estimation
        self.final_conv = nn.Conv2d(64, 4, kernel_size=1)  # Adjust the input channels accordingly

    def forward(self, x):
        # Forward pass through the network, incorporating attention and multi-scale features
        x = self.conv1(x)
        x = self.se1(x)
        # Continue forward pass, combining features from different scales
        x = self.final_conv(x)
        return x

def convbn_3d(in_channels, out_channels, kernel_size, stride, pad):
    return nn.Sequential(nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                   padding=pad, bias=False),
                         nn.BatchNorm3d(out_channels))

class Mish(nn.Module):
    def __init__(self):
        super().__init__()
        # print("Mish activation loaded...")

    def forward(self, x):
        # save 1 second per epoch with no x= x*() and then return x...just inline it.
        return x * (torch.tanh(F.softplus(x)))

def FMish(x):
    '''

    Applies the mish function element-wise:

    mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))

    See additional documentation for mish class.

    '''

    return x * torch.tanh(F.softplus(x))

def disparity_regression(x, maxdisp):
    assert len(x.shape) == 4
    disp_values = torch.arange(0, maxdisp, dtype=x.dtype, device=x.device)
    disp_values = disp_values.view(1, maxdisp, 1, 1)
    return torch.sum(x * disp_values, 1, keepdim=False)



class hourglass(nn.Module):
    def __init__(self, in_channels):
        super(hourglass, self).__init__()

        self.conv1 = nn.Sequential(convbn_3d(in_channels, in_channels * 2, 3, 2, 1),
                                   Mish())

        self.conv2 = nn.Sequential(convbn_3d(in_channels * 2, in_channels * 2, 3, 1, 1),
                                   Mish())

        self.conv3 = nn.Sequential(convbn_3d(in_channels * 2, in_channels * 4, 3, 2, 1),
                                   Mish())

        self.conv4 = nn.Sequential(convbn_3d(in_channels * 4, in_channels * 4, 3, 1, 1),
                                   Mish())

        self.conv5 = nn.Sequential(
            nn.ConvTranspose3d(in_channels * 4, in_channels * 2, 3,
                               padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(in_channels * 2))

        self.conv6 = nn.Sequential(
            nn.ConvTranspose3d(in_channels * 2, in_channels, 3,
                               padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(in_channels))

        self.redir1 = convbn_3d(in_channels, in_channels,
                                kernel_size=1, stride=1, pad=0)
        self.redir2 = convbn_3d(
            in_channels * 2, in_channels * 2, kernel_size=1, stride=1, pad=0)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)

        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)

        conv5 = FMish(self.conv5(conv4) + self.redir2(conv2))
        conv6 = FMish(self.conv6(conv5) + self.redir1(x))

        return conv6


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
        
        # Layout Mixed
        #_______________________________________________________________________________________________
        self.conv3d1 = nn.Conv3d(1, 16, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn3d1 = nn.BatchNorm3d(16)
        self.pool3d1 = nn.MaxPool3d(kernel_size=(2, 1, 1))

        self.conv3d2 = nn.Conv3d(16, 8, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn3d2 = nn.BatchNorm3d(8)
        self.pool3d2 = nn.MaxPool3d(kernel_size=(2, 1, 1))

        # Transition to 2D
        self.conv3d_to_2d = nn.Conv3d(8, 4, kernel_size=(3, 1, 1))

        # 2D convolutional layers
        self.conv2d1 = nn.Conv2d(4, 16, kernel_size=3, padding=1)
        self.bn2d1 = nn.BatchNorm2d(16)
        self.residual1 = nn.Conv2d(16, 16, kernel_size=3, padding=1)

        self.conv2d2 = nn.Conv2d
        '''
        # ELFNet inspired
        #_______________________________________________________________________________________________

        self.dres0 = nn.Sequential(convbn_3d(1, 32, 3, 1, 1),
                                   Mish(),
                                   convbn_3d(32, 32, 3, 1, 1),
                                   Mish())

        self.dres1 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                   Mish(),
                                   convbn_3d(32, 32, 3, 1, 1))

        self.dres2 = hourglass(32)

        self.classif0 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      Mish(),
                                      nn.Conv3d(32, 4, kernel_size=3, padding=1, stride=1, bias=False))

    def forward(self, input):
        # Add a channel dimension for 3D convolution (N, C, D, H, W)
        x = input.unsqueeze(1)

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

        # Layout Mixed
        #_______________________________________________________________________________________________

        # 3D Convolutional layers
        x = F.relu(self.bn3d1(self.conv3d1(x)))
        x = self.pool3d1(x)

        x = F.relu(self.bn3d2(self.conv3d2(x)))
        x = self.pool3d2(x)

        # Transition from 3D to 2D
        x = self.conv3d_to_2d(x)
        # Remove the depth dimension which is now reduced to 1
        x = torch.squeeze(x, dim=2)  # Assuming the depth dimension is the second one


        # 2D Convolutional layers
        x = F.relu(self.bn2d1(self.conv2d1(x)))

        # Residual connection example
        identity = x
        out = F.relu(self.bn2d1(self.residual1(x)))
        out += identity
        x = F.relu(out)

        x = F.relu(self.bn2d2(self.conv2d2(x)))

        # Output layer
        x = self.output_layer(x)
        '''
        # ELFNet inspired
        # _______________________________________________________________________________________________

        cost0 = self.dres0(x)
        #cost0 = self.dres1(cost0) + cost0

        out1 = self.dres2(cost0)

        out1 = self.classif0(out1)

        (cost0, logla0, logalpha0, logbeta0) = torch.split(self.classif0(cost0), 1, dim=1)

        def get_pred(cost):
            cost_upsample = F.upsample(cost, [self.maxdisp, input.size()[2], input.size()[
                3]], mode='trilinear', align_corners=True)
            cost_upsample = torch.squeeze(cost_upsample, 1)
            prob = F.softmax(cost_upsample, dim=1)
            pred = disparity_regression(prob, self.maxdisp)
            return pred, prob

        def get_logits(cost, prob):
            cost_upsample = F.upsample(cost, [self.maxdisp, input.size()[2], input.size()[
                3]], mode='trilinear', align_corners=True)
            cost_upsample = torch.squeeze(cost_upsample, 1)
            pred = torch.sum(cost_upsample * prob, 1, keepdim=False)
            return pred

        pred0, prob0 = get_pred(cost0)
        logla0 = get_logits(logla0, prob0)
        logalpha0 = get_logits(logalpha0, prob0)
        logbeta0 = get_logits(logbeta0, prob0)
        la0, alpha0, beta0 = self.get_uncertainty(
            logla0, logalpha0, logbeta0)

        (u, la, alpha, beta) = pred0, la0, alpha0, beta0

        u = torch.unsqueeze(u, 1)
        refinenet_feature_left = features_left["finetune_feature"]
        refinenet_feature_left = F.upsample(refinenet_feature_left, [left.size()[
                                            2], left.size()[3]], mode='bilinear', align_corners=True)
        refinenet_feature_right = features_right["finetune_feature"]
        refinenet_feature_right = F.upsample(refinenet_feature_right, [left.size()[
                                             2], left.size()[3]], mode='bilinear', align_corners=True)
        refinenet_feature_right_warp = warp(refinenet_feature_right, u)
        refinenet_costvolume = build_corrleation_volume(
            refinenet_feature_left, refinenet_feature_right_warp, 24, 1)
        refinenet_costvolume = torch.squeeze(refinenet_costvolume, 1)
        feature = self.dispupsample(u)
        refinenet_combine = torch.cat((refinenet_feature_left - refinenet_feature_right_warp,
                                      refinenet_feature_left, feature, u, refinenet_costvolume), dim=1)
        disp_finetune = self.refinenet3(refinenet_combine, u)
        disp_finetune = torch.squeeze(disp_finetune, 1)
        u = torch.squeeze(u, 1)



        x = evidential
        #x = torch.squeeze(x, dim=1)

        # Apply sigmoid to the first channel
        first_channel_sigmoid = torch.sigmoid(x[0:1, :, :])  # Keeps the tensor 3D

        x_softplus = F.softplus(x)

        x_modified = torch.cat((first_channel_sigmoid, x_softplus[1:, :, :]), dim=0)
        y = x_modified.clone()

        y = torch.squeeze(y, dim=0)

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

