import math
import torch
import models
import utils
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import functools

# Step 1: Define a Dataset with random data
class RandomDataset(Dataset):
    def __init__(self, num_samples):
        self.num_samples = num_samples
        self.inputs = torch.randn(num_samples, 1)  # Assuming 1-dimensional input
        self.outputs = torch.randn(num_samples, 4)  # Assuming 4-dimensional output

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.inputs[idx], self.outputs[idx]

LR = 5e-4    # learning rate
COEFF = .01  # lambda


# Step 2: Set up a DataLoader
dataset = RandomDataset(num_samples=1000)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

model = torch.nn.Sequential(models.Model(4), models.DERLayer())
loss_fct = functools.partial(models.loss_der, coeff=COEFF)

model, loss, x_scan = utils.train(n_epochs=n_epochs,
                 model=model,
                 loss_fct=loss_fct,
                 error_fct=error_fct,  # only used for logging
                 lr=LR,
                 train_dl=train_dl,
                 test_dl=test_dl,
                 scan_lim=(-7, 7.),
                 device=DEVICE)