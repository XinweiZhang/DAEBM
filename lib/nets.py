import math
import re

import torch
from torch import nn
from torch.nn import LeakyReLU, ReLU, Sigmoid, SiLU, Softplus


def get_act_func_by_name(act_func_name, inplace=False):
    if act_func_name == "SiLU":
        act_func = SiLU(inplace=inplace)
    elif act_func_name == "ReLU":
        act_func = ReLU(inplace=inplace)
    elif act_func_name.startswith("lReLU"):
        try:
            act_func = LeakyReLU(
                float(re.findall(r"[-+]?\d*\.\d+|\d+", act_func_name)[0]),
                inplace=inplace,
            )
        except Exception:
            act_func = LeakyReLU(0.1, inplace=inplace)
    elif act_func_name == "Sigmoid":
        act_func = Sigmoid()
    elif act_func_name == "Softplus":
        act_func = Softplus()
    else:
        raise TypeError("No such activation function")
    return act_func


def get_timestep_embedding(timesteps, embedding_dim):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models:
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


class ToyModel(nn.Module):
    def __init__(self, n_output=2, act_func="SiLU"):
        super().__init__()
        self.lin1 = nn.Linear(2, 128)
        self.lin2 = nn.Linear(128, 128)
        self.lin3 = nn.Linear(128, 128)
        self.lin4 = nn.Linear(128, n_output)

        self.act_func = get_act_func_by_name(act_func)

    def forward(self, x):
        x = self.act_func(self.lin1(x))
        x = self.act_func(self.lin2(x))
        x = self.act_func(self.lin3(x))
        x = self.lin4(x)
        return x

    def f(self, x):
        output = self.forward(x)

        return output

    def energy_output(self, x, t=None):

        output = self.f(x)

        if t is None:
            energy = -output[:, 0]
        else:
            energy = -output.gather(1, t.long().view(-1, 1)).squeeze()

        return energy


class ToyTembModel(nn.Module):
    def __init__(self, n_class=1, act_func="SiLU", hidden_num=128):
        super().__init__()
        self.ch = hidden_num
        self.temb_ch = hidden_num
        self.lin1 = nn.Linear(2, hidden_num)
        self.lin2 = nn.Linear(hidden_num, hidden_num)
        self.lin3 = nn.Linear(hidden_num, hidden_num)
        self.lin4 = nn.Linear(hidden_num, 1)
        self.temb_proj1 = nn.Linear(self.temb_ch, hidden_num)
        self.temb_proj2 = nn.Linear(self.temb_ch, hidden_num)
        self.temb_proj3 = nn.Linear(self.temb_ch, hidden_num)

        self.act_func = get_act_func_by_name(act_func)
        self.temb = nn.Module()
        self.temb.dense = nn.ModuleList(
            [nn.Linear(self.ch, self.temb_ch), nn.Linear(self.temb_ch, self.temb_ch),]
        )
        self.n_class = n_class

    def forward(self, x, t=torch.tensor(0)):
        temb = get_timestep_embedding(t.to(x.device), self.ch)
        temb = self.temb.dense[0](temb)
        temb = self.act_func(temb)
        temb = self.temb.dense[1](temb)
        x = self.act_func(
            self.lin1(x.flatten(1)) * self.temb_proj1(self.act_func(temb))
        )
        x = self.act_func(self.lin2(x)) * self.temb_proj2(self.act_func(temb))
        x = self.act_func(self.lin3(x)) * self.temb_proj3(self.act_func(temb))
        x = self.lin4(x).squeeze()
        return x

    def f(self, x):
        output = [
            self.forward(x, t.long().repeat(x.shape[0]).to(x.device))
            for t in torch.arange(self.n_class)
        ]
        output = torch.stack(output, 1)

        return output

    def energy_output(self, x, t=None):
        if t is None:
            t = torch.zeros(x.shape[0]).long().to(x.device)

        energy = -self.forward(x, t)

        return energy


def test(net):
    import numpy as np

    total_params = 0

    for x in filter(lambda p: p.requires_grad, net.parameters()):
        total_params += np.prod(x.data.numpy().shape)
    print("Total number of params", total_params)
    print(
        "Total layers",
        len(
            list(
                filter(
                    lambda p: p.requires_grad and len(p.data.size()) > 1,
                    net.parameters(),
                )
            )
        ),
    )
