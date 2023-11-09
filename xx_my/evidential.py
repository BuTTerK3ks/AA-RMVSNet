import math
import torch
import models
import utils
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import functools


SEED = 42
NPRNG = np.random.default_rng(SEED)
LR = 5e-4    # learning rate
COEFF = .01  # lambda
BATCH_SIZE = 100
N_SAMPLES = 1
N_EPOCHS = 5
DEVICE = utils.get_best_device()


def generate_data(x_min, x_max, n, *, rng, train=True, dtype=None):
    if dtype is None:
        dtype = np.float32

    x = np.linspace(x_min, x_max, n)
    x = np.expand_dims(x, -1).astype(dtype)

    sigma = 3.0 * np.ones_like(x) if train else np.zeros_like(x)
    y = x**3 + rng.normal(0.0, sigma).astype(dtype)

    return np.concatenate((x, y), axis=1)

def error_fct(y, y_pred):
    return torch.mean(torch.abs(y[:, 0] - y_pred))

def get_dataloader(n):
    kwargs = {
        'rng': NPRNG,
        'dtype': np.float32,
    }
    train_data = generate_data(-4, 4, n, **kwargs)
    test_data = generate_data(-4, 4, n, **kwargs)

    dl_kwargs = {
        'batch_size': BATCH_SIZE,
        'shuffle': True,
        'drop_last': True
    }

    train_dl = torch.utils.data.DataLoader(train_data, **dl_kwargs)
    test_dl = torch.utils.data.DataLoader(test_data, **dl_kwargs)

    return train_dl, test_dl

train_dl, test_dl = get_dataloader(n=1000)



def train_der(n_epochs):
    model = torch.nn.Sequential(models.Model(4), models.DERLayer())
    loss_fct = functools.partial(models.loss_der, coeff=COEFF)
    return utils.train(n_epochs=n_epochs,
                 model=model,
                 loss_fct=loss_fct,
                 error_fct=error_fct,  # only used for logging
                 lr=LR,
                 train_dl=train_dl,
                 test_dl=test_dl,
                 scan_lim=(-7, 7.),
                 device=DEVICE)




model, loss, x_scan = utils.train_loop(train_der, n_samples=N_SAMPLES, n_epochs=N_EPOCHS, quiet=False)



print("end")