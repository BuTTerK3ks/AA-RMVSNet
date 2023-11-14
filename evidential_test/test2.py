import torch
import math
import torch.nn as nn
import torch.optim as optim

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

        # Define linear layer
        self.linear_layer = nn.Linear(128 * 160 * 100, 500)  # Adjust input and output dimensions as needed
        self.relu = nn.ReLU()

        # Define hidden layer
        self.hidden_layer = nn.Linear(500, 4)  # Adjust input and output dimensions as needed
        self.der_layer = DERLayer()

    def forward(self, x):
        # Flatten the input tensor if it's not already flat
        x = x.view(x.size(0), -1)

        # Pass through linear layer and apply activation function
        x = self.relu(self.linear_layer(x))

        # Pass through hidden layer
        x = self.hidden_layer(x)

        x = self.der_layer(x)

        return x

def loss_der(y, y_pred, coeff):
    gamma, nu, alpha, beta = y[:, 0], y[:, 1], y[:, 2], y[:, 3]
    error = gamma - y_pred
    omega = 2.0 * beta * (1.0 + nu)

    return torch.mean(
        0.5 * torch.log(math.pi / nu)
        - alpha * torch.log(omega)
        + (alpha + 0.5) * torch.log(error**2 * nu + omega)
        + torch.lgamma(alpha)
        - torch.lgamma(alpha + 0.5)
        + coeff * torch.abs(error) * (2.0 * nu + alpha)
    )


# Pass your tensor through the model
input_tensor = torch.randn(1, 100, 128, 160)
ground_truth = torch.randn(1)
number_steps = 50

# Create an instance of your model
model = TestModel()
optimizer = optim.Adam(model.parameters(), lr=0.001)

optimizer.zero_grad()

prediction = model(input_tensor)
loss = loss_der(prediction, 1, coeff=0.01)

loss.backward()
optimizer.step()

# Check the output shape
print(prediction.shape)