# Code borrow from score_sde_pytorch
"""DDPM model.

This code is the pytorch equivalent of:
https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/models/unet.py
"""
import functools

import torch
import torch.nn as nn

import lib.unet_layers as layers
from lib.nets import get_act_func_by_name

ResnetBlockDDPM = layers.ResnetBlockDDPM
Upsample = layers.Upsample
Downsample = layers.Downsample
conv3x3 = layers.ddpm_conv3x3
default_initializer = layers.default_init


class UNet(nn.Module):
    def __init__(
        self,
        *,
        nf=128,
        num_res_blocks=2,
        ch_mult=(1, 2, 2, 2),
        channels=3,
        attn_resolutions=(16,),
        resolution=32,
        dropout=0.0,
        act_func="SiLU",
        use_attention=False,
        conditional=True,
        centered=True,
        resamp_with_conv=False,
        conv_shortcut=False,
    ):
        super().__init__()

        self.act = act = get_act_func_by_name(act_func)
        self.nf = nf
        self.num_res_blocks = num_res_blocks
        self.attn_resolutions = attn_resolutions
        self.num_resolutions = num_resolutions = len(ch_mult)
        self.all_resolutions = all_resolutions = [
            resolution // (2 ** i) for i in range(num_resolutions)
        ]
        self.conditional = conditional
        self.use_attention = use_attention
        self.centered = centered

        AttnBlock = functools.partial(layers.AttnBlock)
        ResnetBlock = functools.partial(
            ResnetBlockDDPM,
            act=act,
            temb_dim=4 * nf,
            dropout=dropout,
            conv_shortcut=conv_shortcut,
        )

        modules = []
        if conditional:
            # Condition on noise levels.
            modules = [nn.Linear(nf, nf * 4)]
            modules[0].weight.data = default_initializer()(modules[0].weight.data.shape)
            nn.init.zeros_(modules[0].bias)
            modules.append(nn.Linear(nf * 4, nf * 4))
            modules[1].weight.data = default_initializer()(modules[1].weight.data.shape)
            nn.init.zeros_(modules[1].bias)

        # Downsampling block
        modules.append(conv3x3(channels, nf))
        hs_c = [nf]
        in_ch = nf
        for i_level in range(num_resolutions):
            # Residual blocks for this resolution
            for i_block in range(num_res_blocks):
                out_ch = nf * ch_mult[i_level]
                modules.append(ResnetBlock(in_ch=in_ch, out_ch=out_ch))
                in_ch = out_ch
                if self.use_attention and all_resolutions[i_level] in attn_resolutions:
                    modules.append(AttnBlock(channels=in_ch))
                hs_c.append(in_ch)
            if i_level != num_resolutions - 1:
                modules.append(Downsample(channels=in_ch, with_conv=resamp_with_conv))
                hs_c.append(in_ch)

        in_ch = hs_c[-1]
        modules.append(ResnetBlock(in_ch=in_ch))
        if self.use_attention:
            modules.append(AttnBlock(channels=in_ch))
        modules.append(ResnetBlock(in_ch=in_ch))

        # Upsampling block
        for i_level in reversed(range(num_resolutions)):
            for i_block in range(num_res_blocks + 1):
                out_ch = nf * ch_mult[i_level]
                modules.append(ResnetBlock(in_ch=in_ch + hs_c.pop(), out_ch=out_ch))
                in_ch = out_ch
            if self.use_attention and all_resolutions[i_level] in attn_resolutions:
                modules.append(AttnBlock(channels=in_ch))
            if i_level != 0:
                modules.append(Upsample(channels=in_ch, with_conv=resamp_with_conv))

        assert not hs_c
        modules.append(nn.GroupNorm(num_channels=in_ch, num_groups=32, eps=1e-6))
        modules.append(conv3x3(in_ch, channels, init_scale=0.0))
        self.all_modules = nn.ModuleList(modules)

    def forward(self, x, labels):
        modules = self.all_modules
        m_idx = 0
        if self.conditional:
            # timestep/scale embedding
            timesteps = labels
            temb = layers.get_timestep_embedding(timesteps, self.nf)
            temb = modules[m_idx](temb)
            m_idx += 1
            temb = modules[m_idx](self.act(temb))
            m_idx += 1
        else:
            temb = None

        if self.centered:
            # Input is in [-1, 1]
            h = x
        else:
            # Input is in [0, 1]
            h = 2 * x - 1.0

        # Downsampling block
        hs = [modules[m_idx](h)]
        m_idx += 1
        for i_level in range(self.num_resolutions):
            # Residual blocks for this resolution
            for i_block in range(self.num_res_blocks):
                h = modules[m_idx](hs[-1], temb)
                m_idx += 1
                if self.use_attention and h.shape[-1] in self.attn_resolutions:
                    h = modules[m_idx](h)
                    m_idx += 1
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(modules[m_idx](hs[-1]))
                m_idx += 1

        h = hs[-1]
        h = modules[m_idx](h, temb)
        m_idx += 1
        if self.use_attention:
            h = modules[m_idx](h)
            m_idx += 1
        h = modules[m_idx](h, temb)
        m_idx += 1

        # Upsampling block
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = modules[m_idx](torch.cat([h, hs.pop()], dim=1), temb)
                m_idx += 1
            if self.use_attention and h.shape[-1] in self.attn_resolutions:
                h = modules[m_idx](h)
                m_idx += 1
            if i_level != 0:
                h = modules[m_idx](h)
                m_idx += 1

        assert not hs
        h = self.act(modules[m_idx](h))
        m_idx += 1
        h = modules[m_idx](h)
        m_idx += 1
        assert m_idx == len(modules)

        return h

    def score_output(self, x, labels):
        return self.forward(x, labels)


if __name__ == "__main__":
    net = UNet(
        nf=128,
        ch_mult=(1, 2, 2, 2),
        channels=1,
        num_res_blocks=1,
        resamp_with_conv=False,
        attn_resolutions=(16,),
        resolution=32,
        dropout=0.0,
        act_func_name="SiLU",
        use_attention=False,
    )
    output = net(torch.randn(32, 1, 32, 32), torch.rand(32) * 999)

    def test(net):
        import numpy as np

        total_params = 0

        for name, x in filter(lambda p: p[1].requires_grad, net.named_parameters()):
            print(name + ":" + str(np.prod(x.data.numpy().shape)))
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

    test(net)
    # print(output.shape)
