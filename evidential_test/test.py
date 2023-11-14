import torch
import math
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class DERLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        gamma = x[:, 0]
        nu = nn.functional.softplus(x[:, 1])
        alpha = nn.functional.softplus(x[:, 2]) + 1.0
        beta = nn.functional.softplus(x[:, 3])
        return torch.stack((gamma, nu, alpha, beta), dim=1)


class TestModel(nn.Module):
    def __init__(self):
        super(TestModel, self).__init__()

        # gamma, nu, alpha, beta
        self.convolution = nn.Conv2d(100, 4, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.convolution(x)
        x[:, 1:3, :, :] = F.softplus(x[:, 1:3, :, :])
        # Add +1 to alpha channel
        x[:, 1, :, :] = torch.add(x[:, 1, :, :], 1)

        return x

def loss_der(y, y_pred, coeff):
    gamma, nu, alpha, beta = y[:, 0, :, :], y[:, 1, :, :], y[:, 2, :, :], y[:, 3, :, :]
    error = gamma - y_pred
    omega = 2.0 * beta * (1.0 + nu)

    loss = 0.5 * torch.log(math.pi / nu) - alpha * torch.log(omega) + (alpha + 0.5) * torch.log(error ** 2 * nu + omega) + torch.lgamma(alpha) - torch.lgamma(alpha + 0.5)+ coeff * torch.abs(error) * (2.0 * nu + alpha)


    loss = torch.mean(
        0.5 * torch.log(math.pi / nu)
        - alpha * torch.log(omega)
        + (alpha + 0.5) * torch.log(error ** 2 * nu + omega)
        + torch.lgamma(alpha)
        - torch.lgamma(alpha + 0.5)
        + coeff * torch.abs(error) * (2.0 * nu + alpha)
    )

    return loss


# Pass your tensor through the model
input_tensor = torch.randn(1, 100, 128, 160)
ground_truth = torch.randn(1, 128, 160)
number_steps = 50

# Create an instance of your model
model = TestModel()

optimizer = optim.Adam(model.parameters(), lr=0.001)
model.train()
for step in range(number_steps):
    optimizer.zero_grad()


    prediction = model(input_tensor)
    loss = loss_der(prediction, ground_truth, coeff=0.01)


    loss.backward()
    optimizer.step()

# Check the output shape
print(prediction.shape)