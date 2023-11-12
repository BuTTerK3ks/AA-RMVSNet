import torch
import torch.nn as nn

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

# Create an instance of your model
model = TestModel()

# Pass your tensor through the model
input_tensor = torch.randn(1, 100, 128, 160)
output_tensor = model(input_tensor)

# Check the output shape
print(output_tensor.shape)