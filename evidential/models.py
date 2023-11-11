import math
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


class EvidentialModule(nn.Module):
    def __init__(self, in_channels=3, n_hidden=128, bias=True):
        super(EvidentialModule, self).__init__()
        self.evidential_model = nn.Sequential(
            nn.Linear(in_channels, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, int(n_hidden/2)),
            nn.ReLU(),
            nn.Linear(int(n_hidden/2), 4),
            DERLayer()
        )
        self.der_layer = DERLayer()

    def forward(self, x):
        #return self.model(x)
        test = nn.Sequential(self.evidential_model, self.der_layer)
        return test(x)
