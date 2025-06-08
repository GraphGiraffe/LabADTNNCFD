import torch
import torch.nn as nn
import torch.nn.functional as F
from src.Models.AutoEncoder import (
    create_layer
)
from src.Models.EnhancedUNet import (
    create_encoder,
    create_decoder
)
from typing import List, Optional, Type


class FiLMBlock(nn.Module):
    """
    Расширенный FiLM-блок с residual, нормализацией и SE-attention.
    """

    def __init__(self, fc_dim: int, channels: int, reduction: int = 4):
        super().__init__()
        # Увеличенная ёмкость MLP: fc_dim -> 2*C -> 4*C -> 2*C
        self.mlp = nn.Sequential(
            nn.Linear(fc_dim, 2 * channels),
            nn.ReLU(inplace=True),
            nn.Linear(2 * channels, 4 * channels),
            nn.ReLU(inplace=True),
            nn.Linear(4 * channels, 2 * channels)
        )
        # Нормализация после модуляции
        self.norm = nn.InstanceNorm2d(channels, affine=True)
        # SE-attention: squeeze-and-excitation
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        # генерируем gamma и beta
        params = self.mlp(v)          # [B, 2*C]
        gamma, beta = params.chunk(2, dim=1)
        gamma = gamma.view(B, C, 1, 1)
        beta = beta.view(B, C, 1, 1)
        # FiLM-модуляция + residual
        mod = x * gamma + beta
        out = x + mod
        # нормализация
        out = self.norm(out)
        # SE-attention
        w = self.se(out)
        out = out * w
        return out


class EnhancedUNetFiLM(nn.Module):
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
        device: str = 'cpu'
    ):
        super().__init__()
        # 1) ЭНКОДЕР
        assert filters, "filters не должен быть пустым"
        self.encoder = create_encoder(
            in_channels, filters, kernel_size,
            weight_norm, batch_norm, activation, layers
        )

        # 2) Усиленные FiLM-блоки по декодерам и уровням
        skip_ch = list(reversed(filters))
        self.film_skip_blocks = nn.ModuleList([
            nn.ModuleList([
                FiLMBlock(fc_in_channels, skip_ch[lvl])
                for lvl in range(len(skip_ch))
            ])
            for _ in range(out_channels)
        ])

        # 3) ДЕКОДЕР — без изменений
        additional_fc_channels = [0] * len(filters)
        self.decoders = nn.ModuleList([
            create_decoder(
                out_channels=1,
                filters=filters,
                additional_fc_channels=additional_fc_channels,
                kernel_size=kernel_size,
                wn=weight_norm,
                bn=batch_norm,
                activation=activation,
                layers=layers
            )
            for _ in range(out_channels)
        ])

        # 4) HEADS
        self.heads = nn.ModuleList([
            nn.Conv2d(filters[0], 1, kernel_size=1)
            for _ in range(out_channels)
        ])
        self.final_activation = final_activation
        self.device = device

    def encode(self, x: torch.Tensor):
        tensors, indices, sizes = [], [], []
        for enc in self.encoder:
            x = enc(x)
            sizes.append(x.size())
            tensors.append(x)
            x, ind = F.max_pool2d(x, 2, 2, return_indices=True)
            indices.append(ind)
        return x, tensors, indices, sizes

    def decode(
        self,
        base: torch.Tensor,
        bc_vec: torch.Tensor,
        tensors: List[torch.Tensor],
        indices: List[torch.Tensor],
        sizes: List[torch.Size]
    ) -> torch.Tensor:
        outs = []
        for dec_idx, decoder in enumerate(self.decoders):
            x = base
            t_copy, i_copy, s_copy = tensors[:], indices[:], sizes[:]
            for lvl, dec_block in enumerate(decoder):
                t = t_copy.pop()
                sz = s_copy.pop()
                ind = i_copy.pop()
                # расширенный FiLM
                t_mod = self.film_skip_blocks[dec_idx][lvl](t, bc_vec)
                x = F.max_unpool2d(x, ind, 2, 2, output_size=sz)
                x = torch.cat([t_mod, x], dim=1)
                x = dec_block(x)
            outs.append(x)
        y = torch.cat(outs, dim=1)
        return y

    def forward(self, x: torch.Tensor, bc_vec: torch.Tensor) -> torch.Tensor:
        x, tensors, indices, sizes = self.encode(x)
        x = self.decode(x, bc_vec, tensors, indices, sizes)
        if self.final_activation:
            x = self.final_activation(x)
        return x
