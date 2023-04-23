import math
import re

import torch
import torch.nn as nn
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


def Normalize(in_channels, norm=None):
    if norm is None:
        return nn.Identity()
    elif norm == "Group":
        return nn.GroupNorm(
            num_groups=32, num_channels=in_channels, eps=1e-6, affine=True
        )
    elif norm == "batch":
        return nn.BatchNorm2d(in_channels, momentum=0.9)
    elif norm == "instance":
        return nn.InstanceNorm2d(in_channels, affine=True)


def SpectralNorm(module, mode=True):
    if mode:
        return nn.utils.spectral_norm(module)

    return module


class ResidualBlock(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv, use_spectral_norm):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = SpectralNorm(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0),
                mode=use_spectral_norm,
            )

    def forward(self, x):
        B, C, H, W = x.shape
        if self.with_conv:
            pad = (0, 1, 0, 1)
            x = nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        assert [x.shape[_] for _ in range(len(x.shape))] == [B, C, H // 2, W // 2]

        return x


class AttnBlock(nn.Module):
    def __init__(self, in_channels, norm=None):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels, norm=norm)
        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, h, w = q.shape
        q = q.reshape(b, c, h * w)
        q = q.permute(0, 2, 1)  # b,hw,c
        k = k.reshape(b, c, h * w)  # b,c,hw
        w_ = torch.bmm(q, k)  # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c) ** (-0.5))
        w_ = nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b, c, h * w)
        w_ = w_.permute(0, 2, 1)  # b,hw,hw (first hw of k, second of q)
        # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = torch.bmm(v, w_)
        h_ = h_.reshape(b, c, h, w)

        h_ = self.proj_out(h_)

        assert h_.shape == x.shape

        return x + h_


class ResnetBlock(nn.Module):
    def __init__(
        self,
        *,
        in_channels,
        out_channels=None,
        conv_shortcut=False,
        use_spectral_norm=True,
        dp_prob=0.0,
        temb_channels=512,
        act_func=None,
        norm=None,
    ):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_convshortcut = conv_shortcut
        self.act_func = act_func

        if temb_channels > 0:
            self.temb_proj = SpectralNorm(
                nn.Linear(temb_channels, out_channels), mode=use_spectral_norm
            )

        self.norm1 = Normalize(in_channels, norm=norm)
        self.conv1 = SpectralNorm(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            mode=use_spectral_norm,
        )

        self.norm2 = Normalize(out_channels, norm=norm)
        self.dp_prob = nn.Dropout(dp_prob)
        self.conv2 = SpectralNorm(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            mode=use_spectral_norm,
        )
        if self.in_channels != self.out_channels:
            if self.use_convshortcut:
                self.conv_shortcut = SpectralNorm(
                    nn.Conv2d(
                        in_channels, out_channels, kernel_size=3, stride=1, padding=1
                    ),
                    mode=use_spectral_norm,
                )
            else:
                self.nin_shortcut = SpectralNorm(
                    nn.Conv2d(
                        in_channels, out_channels, kernel_size=1, stride=1, padding=0
                    ),
                    mode=use_spectral_norm,
                )

    def forward(self, x, temb=None):
        h = x
        h = self.norm1(h)
        h = self.act_func(h)
        h = self.conv1(h)

        if temb is not None:
            h += self.temb_proj(self.act_func(temb))[:, :, None, None]

        h = self.norm2(h)
        h = self.act_func(h)
        h = self.dp_prob(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_convshortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x + h


class ResNetTemb(nn.Module):
    def __init__(
        self,
        *,
        in_channels=3,
        resolution=32,
        num_res_blocks=2,
        ch=128,
        ch_mult=(1, 2, 2, 2),
        use_attention=False,
        attn_resolutions=(16,),
        resamp_with_conv=False,
        conv_shortcut=True,
        use_spectral_norm=False,
        norm=None,
        dp_prob=0.0,
        act_func="lReLU[0.2]",
        n_class=1,
    ):
        super().__init__()
        self.ch, self.ch_mult = ch, ch_mult
        self.num_res_blocks = num_res_blocks
        self.attn_resolutions = attn_resolutions
        self.num_resolutions = len(self.ch_mult)
        self.conv_shortcut = conv_shortcut
        self.resamp_with_conv = resamp_with_conv
        self.use_attention = use_attention
        self.use_spectral_norm = use_spectral_norm
        self.temb_ch = 4 * self.ch
        self.n_class = n_class
        if resolution == 28 or resolution == 32:
            self.resolution = 32
        self.sum_pool = True
        self.act_func = get_act_func_by_name(act_func)
        # timestep embedding
        self.temb = nn.Module()
        self.temb.dense = nn.ModuleList(
            [
                SpectralNorm(
                    nn.Linear(self.ch, self.temb_ch), mode=self.use_spectral_norm
                ),
                SpectralNorm(
                    nn.Linear(self.temb_ch, self.temb_ch), mode=self.use_spectral_norm
                ),
            ]
        )
        self.temb.final_dense = SpectralNorm(
            nn.Linear(self.temb_ch, self.ch * self.ch_mult[-1]), mode=False
        )

        # downsampling
        self.conv_in = SpectralNorm(
            nn.Conv2d(
                in_channels,
                self.ch,
                kernel_size=3,
                stride=1,
                padding=3 if resolution == 28 else 1,
            ),
            mode=self.use_spectral_norm,
        )

        curr_res = self.resolution
        in_ch_mult = (1,) + ch_mult
        self.down = nn.ModuleList()
        block_in = None
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(
                    ResnetBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                        conv_shortcut=self.conv_shortcut,
                        use_spectral_norm=self.use_spectral_norm,
                        temb_channels=self.temb_ch,
                        dp_prob=dp_prob,
                        act_func=self.act_func,
                        norm=norm,
                    )
                )
                block_in = block_out
                if self.use_attention and curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in, norm=norm))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(
                    block_in, resamp_with_conv, self.use_spectral_norm
                )
                curr_res = curr_res // 2
            self.down.append(down)
        self.normalize_last = Normalize(self.ch * self.ch_mult[-1], norm=norm)
        self.last = nn.Linear(self.ch * self.ch_mult[-1], 1)

    def features(self, x, t):
        # assert x.shape[2] == x.shape[3] == self.resolution

        # timestep embedding
        temb = get_timestep_embedding(t, self.ch)
        temb = self.temb.dense[0](temb)
        temb = self.act_func(temb)
        temb = self.temb.dense[1](temb)

        # hs = [self.conv_in(x)]
        h = self.conv_in(x)
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                # h = self.down[i_level].block[i_block](hs[-1], temb)
                h = self.down[i_level].block[i_block](h, temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                # hs.append(h)
            if i_level != self.num_resolutions - 1:
                # hs.append(self.down[i_level].downsample(hs[-1]))
                h = self.down[i_level].downsample(h)

        h = self.act_func(h)
        if self.sum_pool:
            h = h.sum((2, 3))
        else:
            h = nn.functional.avg_pool2d(h, h.shape[2])
        temb_final = self.temb.final_dense(self.act_func(temb))
        return h, temb_final

    def forward(self, x, t):
        features, temb_final = self.features(x, t)
        h = features * temb_final
        h = torch.sum(h, 1)

        return h

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

    def score_output(self, x, labels):
        if x.requires_grad is False:
            x_grad = torch.autograd.Variable(x.clone(), requires_grad=True)
        else:
            x_grad = x
        output = torch.autograd.grad(
            self.energy_output(x_grad, labels).sum(), [x_grad], create_graph=True
        )[0]

        return output


def test(net):
    import numpy as np

    total_params = 0

    for name, x in filter(lambda p: p[1].requires_grad, net.named_parameters()):
        print(name + str(x.data.numpy().shape))
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


if __name__ == "__main__":

    x = torch.randn(10, 3, 32, 32)
    t = torch.randint(size=(10,), high=10)

    net = ResNetTemb(
        ch=128,
        ch_mult=(1, 2, 2),
        num_res_blocks=1,
        n_class=10,
        use_spectral_norm=False,
        conv_shortcut=False,
    )
    h1 = net(x, t)
    assert h1.shape == torch.Size([10])
    h2 = net.f(x)
    assert h2.shape == torch.Size([10, 10])
    h3 = net.energy_output(x, t)
    assert h3.shape == torch.Size([10])
    h4 = net.energy_output(x)
    assert h3.shape == torch.Size([10])
