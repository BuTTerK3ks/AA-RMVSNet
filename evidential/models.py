import torch
import math
import numpy as np
import torch.nn as nn
from torch import Tensor
import torch.optim as optim
import torch.nn.functional as F


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


def disparity_regression(x, depth_values, max_d=60):
    assert len(x.shape) == 4
    disp_values = torch.arange(0, max_d, dtype=x.dtype, device=x.device)
    depth_1 = depth_values.size()[1]
    disp_values = depth_values.view(1, depth_values.size()[1], 1, 1)
    return torch.sum(x * disp_values, 1, keepdim=False)


def disparity_classification(x, depth_values):
    assert len(x.shape) == 4
    max_idx = torch.argmax(x, dim=1)
    pred = torch.take(depth_values, max_idx)
    return pred


class HourGlassUp(nn.Module):
    def __init__(self, in_channels):
        super(HourGlassUp, self).__init__()

        self.conv1 = nn.Conv3d(in_channels, in_channels * 2, kernel_size=3, stride=2,
                               padding=1, bias=False)

        self.conv2 = nn.Sequential(convbn_3d(in_channels * 2, in_channels * 2, 3, 1, 1),
                                   Mish())

        # self.conv3 = nn.Sequential(convbn_3d(in_channels * 2, in_channels * 4, 3, 2, 1),
        #                            Mish())
        self.conv3 = nn.Conv3d(in_channels * 2, in_channels * 4, kernel_size=3, stride=2,
                               padding=1, bias=False)

        self.conv4 = nn.Sequential(convbn_3d(in_channels * 4, in_channels * 4, 3, 1, 1),
                                   Mish())

        # self.conv5 = nn.Sequential(convbn_3d(in_channels * 4, in_channels * 4, 3, 2, 1),
        #                            Mish())
        # self.conv5 = nn.Conv3d(in_channels * 4, in_channels * 4, kernel_size=3, stride=2,
        #                        padding=1, bias=False)

        # self.conv6 = nn.Sequential(convbn_3d(in_channels * 4, in_channels * 4, 3, 1, 1),
        #                            Mish())
        # self.conv7 = nn.Sequential(
        #     nn.ConvTranspose3d(in_channels * 4, in_channels * 4, 3,
        #                        padding=1, output_padding=1, stride=2, bias=False),
        #     nn.BatchNorm3d(in_channels * 4))

        self.conv8 = nn.Sequential(
            nn.ConvTranspose3d(in_channels * 4, in_channels * 2, 3,
                               padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(in_channels * 2))

        self.conv9 = nn.Sequential(
            nn.ConvTranspose3d(in_channels * 2, in_channels, 3,
                               padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(in_channels))

        self.combine1 = nn.Sequential(convbn_3d(in_channels * 3, in_channels * 2, 3, 1, 1),
                                      Mish())
        self.combine2 = nn.Sequential(convbn_3d(in_channels * 5, in_channels * 4, 3, 1, 1),
                                      Mish())
        # self.combine3 = nn.Sequential(convbn_3d(in_channels * 6, in_channels * 4, 3, 1, 1),
        #                               Mish())

        self.redir1 = convbn_3d(in_channels, in_channels,
                                kernel_size=1, stride=1, pad=0)
        self.redir2 = convbn_3d(
            in_channels * 2, in_channels * 2, kernel_size=1, stride=1, pad=0)
        self.redir3 = convbn_3d(
            in_channels * 4, in_channels * 4, kernel_size=1, stride=1, pad=0)

        self.dropout = nn.Dropout3d(p=0.5)

    def forward(self, x, feature4, feature5):
        conv1 = self.conv1(x)  # 1/8
        conv1 = torch.cat((conv1, feature4), dim=1)  # 1/8

        conv1 = self.combine1(conv1)  # 1/8
        conv2 = self.conv2(conv1)  # 1/8

        conv3 = self.conv3(conv2)  # 1/16
        conv3 = torch.cat((conv3, feature5), dim=1)  # 1/16
        conv3 = self.combine2(conv3)  # 1/16
        conv4 = self.conv4(conv3)  # 1/16

        conv7 = FMish(self.redir3(conv4))
        conv8 = FMish(self.conv8(conv7) + self.redir2(conv2))
        conv9 = FMish(self.conv9(conv8) + self.redir1(x))

        # Apply dropout
        conv9 = self.dropout(conv9)

        return conv9


class HourGlass(nn.Module):
    def __init__(self, in_channels):
        super(HourGlass, self).__init__()

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

        self.dropout = nn.Dropout3d(p=0.5)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)

        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)

        conv5 = FMish(self.conv5(conv4) + self.redir2(conv2))
        conv6 = FMish(self.conv6(conv5) + self.redir1(x))

        # Apply dropout
        conv6 = self.dropout(conv6)

        return conv6


class EvidentialWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.original_model = EvidentialModule(depth=32).cuda()

    def forward(self, x):
        # Create dummy proj_matrices and depth_values with the expected shape and type
        dummy_depth_values = torch.randn(1, 32).cuda()  # Adjust the shape and values as needed
        return self.original_model(x, dummy_depth_values)


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
        # _______________________________________________________________________________________________
        self.maxdisp = 32

        self.dres0 = nn.Sequential(convbn_3d(1, 32, 3, 1, 1),
                                   Mish(),
                                   convbn_3d(32, 32, 3, 1, 1),
                                   Mish())

        self.dres1 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                   Mish(),
                                   convbn_3d(32, 32, 3, 1, 1),
                                   Mish())

        self.conv_vol2 = nn.Sequential(convbn_3d(1, 32, 3, 1, 1),
                                       Mish(),
                                       convbn_3d(32, 32, 3, 1, 1))

        self.conv_vol3 = nn.Sequential(convbn_3d(1, 32, 3, 1, 1),
                                       Mish(),
                                       convbn_3d(32, 32, 3, 1, 1))

        self.combine1 = HourGlassUp(32)
        self.dres2 = HourGlass(32)
        self.dres3 = HourGlass(32)

        self.classif0 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      Mish(),
                                      nn.Conv3d(32, 4, kernel_size=3, padding=1, stride=1, bias=False))

        self.classif1 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      Mish(),
                                      nn.Conv3d(32, 4, kernel_size=3, padding=1, stride=1, bias=False))

        self.classif2 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      Mish(),
                                      nn.Conv3d(32, 4, kernel_size=3, padding=1, stride=1, bias=False))

        self.dropout = nn.Dropout3d(p=0.5)

    def get_uncertainty(self, logv, logalpha, logbeta):
        v = self.evidence(logv)
        alpha = self.evidence(logalpha) + 1
        beta = self.evidence(logbeta)
        return v, alpha, beta

    def moe_nig(self, u1, la1, alpha1, beta1, u2, la2, alpha2, beta2):
        # Eq. 9
        la = la1 + la2
        u = (la1 * u1 + u2 * la2) / la
        # u[la == 0] = (u1[la == 0] + u2[la == 0]) * 0.5
        alpha = alpha1 + alpha2 + 0.5
        beta = beta1 + beta2 + 0.5 * \
               (la1 * (u1 - u) ** 2 + la2 * (u2 - u) ** 2)
        return u, la, alpha, beta

    def combine_uncertainty(self, ests):
        [u, la, alpha, beta] = ests[0]
        for i in range(1, len(ests)):
            [u1, la1, alpha1, beta1] = ests[i]
            u, la, alpha, beta = self.moe_nig(
                u, la, alpha, beta, u1, la1, alpha1, beta1)
        return (u, la, alpha, beta)

    def evidence(self, x):
        # return tf.exp(x)
        return F.softplus(x)

    def forward(self, input, depth_value):
        # Add a channel dimension for 3D convolution (N, C, D, H, W)
        # x = input.unsqueeze(1)

        # Layout 3D
        # _______________________________________________________________________________________________
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

        x = input.unsqueeze(0)

        volume1 = F.interpolate(x, [self.maxdisp, input.size()[2], input.size()[
            3]], mode='trilinear', align_corners=True)
        volume1 = F.softmax(volume1, dim=2)

        volume2 = F.interpolate(x, [self.maxdisp // 2, input.size()[2] // 2, input.size()[
            3] // 2], mode='trilinear', align_corners=True)
        volume2 = F.softmax(volume2, dim=2)

        volume3 = F.interpolate(x, [self.maxdisp // 4, input.size()[2] // 4, input.size()[
            3] // 4], mode='trilinear', align_corners=True)
        volume3 = F.softmax(volume3, dim=1)

        cost0 = self.dres0(volume1)
        cost0 = self.dres1(cost0) + cost0

        volume2 = self.conv_vol2(volume2)
        volume3 = self.conv_vol3(volume3)

        combine = self.combine1(cost0, volume2, volume3)
        out1 = self.dres2(combine)
        out2 = self.dres3(out1)

        def get_pred(cost, depth_value):
            cost_upsample = F.interpolate(cost, [self.maxdisp, input.size()[2], input.size()[
                3]], mode='trilinear', align_corners=True)
            cost_upsample = torch.squeeze(cost_upsample, 1)
            prob = F.softmax(cost_upsample, dim=1)
            # TODO Regression or classification based
            pred = disparity_regression(prob, depth_value)
            # pred = disparity_classification(prob, depth_value)
            return pred, prob

        def get_logits(cost, prob):
            cost_upsample = F.interpolate(cost, [self.maxdisp, input.size()[2], input.size()[
                3]], mode='trilinear', align_corners=True)
            cost_upsample = torch.squeeze(cost_upsample, 1)
            pred = torch.sum(cost_upsample * prob, 1, keepdim=False)
            return pred

        (cost0, logla0, logalpha0, logbeta0) = torch.split(
            self.classif0(cost0), 1, dim=1)
        (cost1, logla1, logalpha1, logbeta1) = torch.split(
            self.classif1(out1), 1, dim=1)
        (cost2, logla2, logalpha2, logbeta2) = torch.split(
            self.classif2(out2), 1, dim=1)

        pred0, prob0 = get_pred(cost0, depth_value=depth_value)
        logla0 = get_logits(logla0, prob0)
        logalpha0 = get_logits(logalpha0, prob0)
        logbeta0 = get_logits(logbeta0, prob0)
        la0, alpha0, beta0 = self.get_uncertainty(
            logla0, logalpha0, logbeta0)

        pred1, prob1 = get_pred(cost1, depth_value=depth_value)
        logla1 = get_logits(logla1, prob1)
        logalpha1 = get_logits(logalpha1, prob1)
        logbeta1 = get_logits(logbeta1, prob1)
        la1, alpha1, beta1 = self.get_uncertainty(
            logla1, logalpha1, logbeta1)

        pred2, prob2 = get_pred(cost2, depth_value=depth_value)
        logla2 = get_logits(logla2, prob2)
        logalpha2 = get_logits(logalpha2, prob2)
        logbeta2 = get_logits(logbeta2, prob2)
        la2, alpha2, beta2 = self.get_uncertainty(
            logla2, logalpha2, logbeta2)

        (u, la, alpha, beta) = self.combine_uncertainty([[pred0, la0, alpha0, beta0], [
            pred1, la1, alpha1, beta1], [pred2, la2, alpha2, beta2]])

        evidential = torch.cat((u, la, alpha, beta))
        prob_combine = torch.stack((prob0, prob1, prob2))
        prob_combine = torch.mean(prob_combine, dim=0)

        # Apply dropout
        evidential = self.dropout(evidential)
        prob_combine = self.dropout(prob_combine)

        return evidential, prob_combine


def criterion_uncertainty(u, la, alpha, beta, y, mask, weight_reg=0.1):
    # our loss function
    om = 2 * beta * (1 + la)
    mask = mask.bool()

    # len(u): size
    loss = torch.sum(
        (0.5 * torch.log(np.pi / la) - alpha * torch.log(om) +
         (alpha + 0.5) * torch.log(la * (u - y) ** 2 + om) +
         torch.lgamma(alpha) - torch.lgamma(alpha + 0.5))[mask]
    ) / torch.sum(mask == True)

    lossr = weight_reg * (torch.sum((torch.abs(u - y) * (2 * la + alpha))[mask])) / torch.sum(mask == True)
    loss = loss + lossr

    return loss


def compute_uncertainty(self, u, la, alpha, beta):
    aleatoric = beta / (alpha - 1)
    epistemic = beta / (alpha - 1) / la
    return aleatoric, epistemic


def loss_der(outputs, depth_gt, mask, depth_value, coeff=0.01):
    evidential_prediction = outputs['evidential_prediction']
    probability_volume = outputs['probability_volume']

    # get EDL parameters
    gamma, nu, alpha, beta = evidential_prediction[0, :, :], evidential_prediction[1, :, :], evidential_prediction[2, :,
                                                                                             :], evidential_prediction[
                                                                                                 3, :, :]
    gamma = torch.unsqueeze(gamma, 0)
    nu = torch.unsqueeze(nu, 0)
    alpha = torch.unsqueeze(alpha, 0)
    beta = torch.unsqueeze(beta, 0)

    loss = criterion_uncertainty(gamma, nu, alpha, beta, depth_gt, mask, weight_reg=0.1)

    # TODO Analyze the difference in performance
    # get aleatoric and epistemic uncertainty
    # method from "unreasonable effective der"
    aleatoric_1 = torch.sqrt(beta * (nu + 1) / nu / alpha)
    epistemic_1 = 1. / torch.sqrt(nu)

    # traditional method
    aleatoric_2 = beta / (alpha - 1)
    epistemic_2 = beta / (alpha - 1) / nu

    total = beta / (alpha - 1)

    evidential = {
        'gamma': gamma,
        'nu': nu,
        'alpha': alpha,
        'beta': beta,
        'aleatoric_1': aleatoric_1,
        'epistemic_1': epistemic_1,
        'aleatoric_2': aleatoric_2,
        'epistemic_2': epistemic_2,
        'total': total
    }

    return loss, gamma, evidential


def monte_carlo_dropout(model, x, depth_value, n_samples=10):
    model.train()  # Set the model to training mode to enable dropout
    predictions = []

    for _ in range(n_samples):
        with torch.no_grad():
            evidential, prob_combine = model(x, depth_value)
        predictions.append(evidential)

    # Stack predictions to compute the mean and variance
    predictions = torch.stack(predictions)
    mean_prediction = predictions.mean(dim=0)
    variance_prediction = predictions.var(dim=0)

    return mean_prediction, variance_prediction
