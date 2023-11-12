import torch
import torch.nn as nn
class Model(nn.Module):
    def __init__(self, n_output, n_hidden=64):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(100, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_output),
        )

    def forward(self, x):
        return self.model(x)

# Create a random tensor
random_tensor = torch.rand(1, 100, 128, 160)