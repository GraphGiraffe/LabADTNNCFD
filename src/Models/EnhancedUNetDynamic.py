import torch
import torch.nn as nn
import torch.nn.functional as F
from src.Models.EnhancedUNet import (
    create_encoder
)
from typing import List, Optional, Type


class DynamicConv2d(nn.Module):
    """
    Генерирует ядро свёртки условно от BC-вектора для каждого входа.
    """

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, fc_dim: int, heads: int = 4):
        super().__init__()
        self.in_ch, self.out_ch, self.k = in_ch, out_ch, kernel_size
        self.heads = heads
        assert fc_dim % heads == 0
        self.head_dim = fc_dim // heads
        self.weight_gen = nn.Linear(fc_dim, heads * in_ch * out_ch * kernel_size * kernel_size)
        self.bias_gen = nn.Linear(fc_dim, heads * out_ch)
        self.padding = kernel_size // 2

    def forward(self, x: torch.Tensor, bc_vec: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        w = self.weight_gen(bc_vec).view(B * self.heads, self.out_ch, self.in_ch, self.k, self.k)
        b = self.bias_gen(bc_vec).view(B * self.heads * self.out_ch)
        x_rep = x.repeat(self.heads, 1, 1, 1)
        y = F.conv2d(
            x_rep,
            weight=w.view(B * self.heads * self.out_ch, self.in_ch, self.k, self.k),
            bias=b,
            padding=self.padding,
            groups=B
        )
        y = y.view(self.heads, B, self.out_ch, H, W).mean(0)
        return y


class EnhancedUNetDynamic(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        filters: List[int] = [16, 32, 64, 128, 256],
        layers: int = 2,
        weight_norm: bool = True,
        batch_norm: bool = True,
        activation: Optional[Type[nn.Module]] = nn.ReLU,
        final_activation: Optional[nn.Module] = None,
        fc_in_channels: int = 6,
        heads: int = 4,
        device: str = 'cpu'
    ):
        super().__init__()
        assert filters, "filters не должен быть пустым"
        # Энкодер: список блоков
        self.encoder = create_encoder(
            in_channels, filters, kernel_size,
            weight_norm, batch_norm, activation, layers
        )
        # Настройки динамических блоков
        skip_ch = list(reversed(filters))
        deep_ch = filters[-1]
        self.dynamic_blocks = nn.ModuleList([
            nn.ModuleList([
                DynamicConv2d(
                    in_ch=skip_ch[lvl] + (deep_ch if lvl == 0 else filters[lvl-1]),
                    out_ch=filters[lvl],
                    kernel_size=kernel_size,
                    fc_dim=fc_in_channels,
                    heads=heads
                )
                for lvl in range(len(skip_ch))
            ])
            for _ in range(out_channels)
        ])
        self.conv_blocks = nn.ModuleList([
            nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(filters[lvl], filters[lvl], kernel_size, padding=kernel_size//2),
                    activation(inplace=True),
                    nn.Conv2d(filters[lvl], filters[lvl], kernel_size, padding=kernel_size//2),
                    activation(inplace=True)
                )
                for lvl in range(len(filters))
            ])
            for _ in range(out_channels)
        ])
        self.heads = nn.ModuleList([
            nn.Conv2d(filters[0], 1, kernel_size=1)
            for _ in range(out_channels)
        ])
        self.final_activation = final_activation
        self.device = device

    def forward(self, x: torch.Tensor, bc_vec: torch.Tensor) -> torch.Tensor:
        # Собираем фичи на каждом уровне энкодера
        feats = []
        h = x
        for block in self.encoder:
            h = block(h)
            feats.append(h)
        deep = feats[-1]
        outs = []
        for dec_idx in range(len(self.dynamic_blocks)):
            y = deep
            for lvl, dyn_block in enumerate(self.dynamic_blocks[dec_idx]):
                skip = feats[-2-lvl]
                y = F.interpolate(y, size=skip.shape[2:], mode='nearest')
                cat = torch.cat([skip, y], dim=1)
                y = dyn_block(cat, bc_vec)
                y = self.conv_blocks[dec_idx][lvl](y)
            pred = self.heads[dec_idx](y)
            pred = F.interpolate(pred, size=x.shape[2:], mode='bilinear', align_corners=False)
            outs.append(pred)
        out = torch.cat(outs, dim=1)
        return self.final_activation(out) if self.final_activation else out
