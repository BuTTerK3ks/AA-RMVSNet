import torch
import math
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

def normalize_target(target, target_min, target_max):
    return (target - target_min) / (target_max - target_min)

def denormalize_target(normalized_target, target_min, target_max):
    return normalized_target * (target_max - target_min) + target_min


class EvidentialModule(nn.Module):
    def __init__(self):
        super(EvidentialModule, self).__init__()
        # one layer
        # nu, alpha, beta
        # First convolutional layer
        self.conv1 = nn.Conv2d(100, 256, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ELU()

        # Second convolutional layer
        self.conv2 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ELU()

        # Third convolutional layer
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ELU()

        # Final layer to bring the channel size to 3
        self.conv4 = nn.Conv2d(256, 4, kernel_size=1, stride=1, padding=0)

    def evidence(self, x):
        # return tf.exp(x)
        return F.softplus(x)

    def get_uncertainty(self, logv, logalpha, logbeta):
        v = self.evidence(logv)
        alpha = self.evidence(logalpha) + 1
        beta = self.evidence(logbeta)
        return v, alpha, beta

    def forward(self, evidential_parameters):
        x = evidential_parameters
        y = torch.zeros_like(x)
        y[:, :, 0, :, :] = x[:, :, 0, :, :]
        y[:, :, 1:4, :, :] = F.softplus(x[:, :, 1:4, :, :])
        # Add +1 to alpha channel
        x = y
        x[:, :, 1, :, :] = torch.add(y[:, :, 1, :, :], 1)

        return x


def loss_der(outputs, depth_gt, mask, depth_value, coeff=0.01, use_mask=True):

    evidential_prediction = outputs['evidential_prediction']
    probability_volume = outputs['probability_volume']

    gamma, nu, alpha, beta = evidential_prediction[:, :, 0, :, :], evidential_prediction[:, :, 1, :, :], evidential_prediction[:, :, 2, :, :], evidential_prediction[:, :, 3, :, :]

    # take max probability and get depth
    highest_prob = torch.argmax(gamma, dim=1).type(torch.long)

    depth_map = torch.take(depth_value, highest_prob)
    nu = torch.take(nu, highest_prob)
    alpha = torch.take(alpha, highest_prob)
    beta = torch.take(beta, highest_prob)

    #TODO use masked?
    error = depth_map - depth_gt
    #error = normalize_target(error, torch.min(error.flatten()), torch.max(error.flatten()))

    omega = 2.0 * beta * (1.0 + nu)

    loss = 0.5 * torch.log(math.pi / nu) - alpha * torch.log(omega) + (alpha + 0.5) * torch.log(error ** 2 * nu + omega) + torch.lgamma(alpha) - torch.lgamma(alpha + 0.5) + coeff * torch.abs(error) * (2.0 * nu + alpha)
    loss = loss * mask
    valid_pixel_num = torch.sum(mask, dim=[1, 2]) + 1e-6
    loss = torch.sum(loss) / valid_pixel_num
    

    '''
    mask = mask == 1
    loss_nll = torch.sum((0.5 * torch.log(math.pi / nu) - alpha * torch.log(omega) +(alpha + 0.5) * torch.log(nu * error ** 2 + omega) + torch.lgamma(alpha) - torch.lgamma(alpha + 0.5))[mask]) / torch.sum(mask == True)
    loss_r = coeff * (torch.sum((torch.abs(error) * (2 * nu + alpha))[mask])) / torch.sum(mask == True)
    loss = loss_nll + loss_r
    '''


    '''
    if use_mask:
        masked_loss = torch.mul(mask, calculated_loss)  # valid pixel
        loss = torch.mean(masked_loss)
    else:
        loss = torch.mean(calculated_loss)
    '''




    aleatoric = torch.sqrt(beta * (nu + 1) / nu / alpha)
    epistemic = 1. / torch.sqrt(nu)
    #TODO Change for other dataset
    #depth_est = map_to_original_range(gamma, 425, 687.35)
    depth_est = depth_map



    # TODO check if right
    return loss, depth_est, aleatoric, epistemic