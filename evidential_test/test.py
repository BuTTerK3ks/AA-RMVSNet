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


def loss_der(prediction, ground_truth, coeff):
    gamma, nu, alpha, beta = prediction[:, 0, :, :], prediction[:, 1, :, :], prediction[:, 2, :, :], prediction[:, 3, :, :]
    error = gamma - ground_truth
    omega = 2.0 * beta * (1.0 + nu)

    '''
    e1 = 0.5 * torch.log(math.pi / nu)
    e2 = -1 * alpha * torch.log(omega)
    e3 = (alpha + 0.5) * torch.log(error ** 2 * nu + omega)
    e4 = -1 * torch.lgamma(alpha) - torch.lgamma(alpha + 0.5)
    e5 = coeff * torch.abs(error) * (2.0 * nu + alpha)
    '''

    calculated_loss = 0.5 * torch.log(math.pi / nu) - alpha * torch.log(omega) + (alpha + 0.5) * torch.log(error ** 2 * nu + omega) + torch.lgamma(alpha) - torch.lgamma(alpha + 0.5) + coeff * torch.abs(error) * (2.0 * nu + alpha)
    calculated_loss = torch.mean(calculated_loss)
    return calculated_loss


# Pass your tensor through the model
input_tensor = torch.randn(1, 100, 128, 160)
ground_truth = torch.randn(1, 128, 160)
number_steps = 5000

# Create an instance of your model
model = EvidentialModule()

optimizer = optim.Adam(model.parameters(), lr=0.01)
model.train()

losses = []
for step in range(number_steps):
    torch.autograd.set_detect_anomaly(True)
    optimizer.zero_grad()

    prediction = model(input_tensor)
    loss = loss_der(prediction, ground_truth, coeff=0.01)

    loss.backward()
    optimizer.step()

    print(loss)

# Check the output shape
print(prediction.shape)