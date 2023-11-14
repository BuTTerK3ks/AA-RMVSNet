import torch
import torch.nn as nn


class YourNetwork(nn.Module):
    def __init__(self):
        super(YourNetwork, self).__init__()

        # Define your layers here
        self.conv1 = nn.Conv2d(100, 64, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 160, 128 * 160)
        self.fc2 = nn.Linear(128 * 160, 128 * 160)
        self.fc3 = nn.Linear(128 * 160, 128 * 160)
        self.fc4 = nn.Linear(128 * 160, 128 * 160)

    def forward(self, x):
        # Define the forward pass
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        x = x.view(-1, 128, 160)  # Reshape to the desired output shape

        return x


# Instantiate your network
your_network = YourNetwork()

# Create a random input tensor with the specified shape
input_tensor = torch.randn(1, 100, 128, 160)

# Forward pass
output_tensor = your_network(input_tensor)
print(output_tensor.shape)