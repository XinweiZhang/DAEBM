import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision as tv
from torch.utils.data import Dataset


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Accumulator:
    """For accumulating sums over `n` variables."""

    def __init__(self, n):
        self.data = [0.0] * n
        self.len = 0

    def add(self, len, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args[0])]
        self.len += len

    def reset(self):
        self.data = [0.0] * len(self.data)
        self.len = 0

    def average(self):
        if self.len > 0:
            return [_ / float(self.len) for _ in self.data]
        else:
            return [0.0] * len(self.data)
            # raise ValueError("accumulator length is 0, can not do average")

    def __getitem__(self, idx):
        return self.data[idx]


def make_figure_grid(
    imgs,
    lbls=None,
    energies=None,
    nrow=2,
    ncol=10,
    figsize=(10, 4),
    show=False,
    suptilte=None,
):
    """Return a 2x10 grid of the MNIST images as a matplotlib figure."""
    # Create a figure to contain the plot.
    figure = plt.figure(figsize=figsize)
    for i in range(min(nrow * ncol, len(imgs))):
        # Start next subplot.
        plt.subplot(nrow, ncol, i + 1)
        if lbls is not None and energies is not None:
            tilte = f"{lbls[i]}:{energies[i]:.2f}"
        elif lbls is None and energies is not None:
            tilte = f"{energies[i]:.2f}"
        elif lbls is not None and energies is None:
            tilte = f"{lbls[i]}"
        else:
            tilte = None

        if tilte is not None:
            plt.title(tilte)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        if imgs.shape[1] == 1:
            plt.imshow(imgs[i,].squeeze(0), cmap="gray")
        else:
            plt.imshow(imgs[i,].permute(1, 2, 0))
    if suptilte:
        plt.suptitle(suptilte)
    plt.tight_layout()
    if show:
        plt.show()
    plt.close()
    return figure


def cycle(loader):
    while True:
        for data in loader:
            yield data


class AddGaussianNoise(nn.Module):
    def __init__(self, std=0):
        super(AddGaussianNoise, self).__init__()
        self.std = std

    def forward(self, x):
        x += torch.randn_like(x) * self.std
        return x


class AddUniformNoise(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, precision=256):
        x = (x * (precision - 1) + torch.rand_like(x)) / precision
        return x


class DataSubset(Dataset):
    def __init__(self, base_dataset, inds=None, size=-1, transform=None):
        self.base_dataset = base_dataset
        if inds is None:
            inds = np.random.choice(list(range(len(base_dataset))), size, replace=False)
        self.inds = inds
        self.transform = transform

    def __getitem__(self, index):
        base_ind = self.inds[index]
        x, y = self.base_dataset[base_ind]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.inds)


def inv_data_transform(data, data_transform):
    if "center_and_scale" in data_transform:
        image = (torch.clamp(data, -1, 1) + 1) / 2
    else:
        image = data

    return image


def imshow(imgs, ncol, normalize=True, scale_each=True, save_dir=None):
    imgs = tv.utils.make_grid(
        imgs, nrow=ncol, normalize=normalize, scale_each=scale_each
    )
    npimgs = imgs.numpy()

    if save_dir is None:
        plt.imshow(np.transpose(npimgs, (1, 2, 0)))
        # plt.show()
        plt.axis("off")
    else:
        plt.imsave(save_dir, np.transpose(npimgs, (1, 2, 0)))


class EMA(object):
    "from DDIM"

    def __init__(self, mu=0.999):
        self.mu = mu
        self.shadow = {}

    def register(self, module):
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, module):
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name].data = (
                    1.0 - self.mu
                ) * param.data + self.mu * self.shadow[name].data

    def ema(self, module):
        for name, param in module.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name].data)

    def ema_copy(self, module):
        module_copy = type(module)(module.config).to(module.config.device)
        module_copy.load_state_dict(module.state_dict())
        self.ema(module_copy)
        return module_copy

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state_dict):
        self.shadow = state_dict
