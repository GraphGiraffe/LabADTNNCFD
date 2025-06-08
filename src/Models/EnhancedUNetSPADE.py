import torch
import torch.nn as nn
import torch.nn.functional as F
from src.Models.EnhancedUNet import create_encoder
from typing import List, Optional, Type


class SPADEBlock(nn.Module):
    """
    Spatially-Adaptive Denormalization (SPADE) для условной модуляции BC-данных.
    """

    def __init__(self, in_channels: int, bc_channels: int, hidden_channels: int = 128):
        super().__init__()
        # Нормализация без аффинных параметров
        self.norm = nn.InstanceNorm2d(in_channels, affine=False)
        # Общее пространство для BC-карт
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(bc_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        # Генераторы γ и β
        self.mlp_gamma = nn.Conv2d(hidden_channels, in_channels, kernel_size=3, padding=1)
        self.mlp_beta = nn.Conv2d(hidden_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, bc_map: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W], bc_map может быть:
        #   [B, bc_channels] или [B, bc_channels, H_bc, W_bc]
        B, C, H, W = x.size()
        # нормализуем фичу
        x_norm = self.norm(x)
        # приводим bc_map к размеру [B, bc_channels, H, W]
        if bc_map.dim() == 2:
            # вектор [B, bc_channels]
            bc_resized = bc_map.view(B, -1, 1, 1).expand(B, bc_map.size(1), H, W)
        elif bc_map.dim() == 3:
            # [B, bc_channels, 1] – расширяем до H×W
            bc_resized = bc_map.unsqueeze(-1).expand(B, bc_map.size(1), H, W)
        else:
            # [B, bc_channels, H_bc, W_bc]
            bc_resized = F.interpolate(bc_map, size=(H, W), mode='nearest')

        actv = self.mlp_shared(bc_resized)   # [B, hidden, H, W]
        gamma = self.mlp_gamma(actv)         # [B, C, H, W]
        beta = self.mlp_beta(actv)           # [B, C, H, W]
        return x_norm * (1 + gamma) + beta


class EnhancedUNetSPADE(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        fc_in_channels: int,
        kernel_size: int = 3,
        filters: List[int] = [16, 32, 64, 128, 256],
        layers: int = 2,
        activation: Optional[Type[nn.Module]] = nn.ReLU,
        final_activation: Optional[nn.Module] = None,
        device='cpu'
    ):
        super().__init__()
        self.device = device
        assert filters, "filters не должен быть пустым"

        # 1) Энкодер (создаёт список последовательных conv-блоков)
        self.encoder = create_encoder(in_channels, filters, kernel_size, True, True, activation, layers)
        num_levels = len(filters)

        # 2) SPADE-блоки: отдельно для каждого выходного канала и каждого уровня
        self.spade_blocks = nn.ModuleList([
            nn.ModuleList([
                SPADEBlock(filters[-1 - lvl], fc_in_channels, hidden_channels=32)
                for lvl in range(num_levels)
            ])
            for _ in range(out_channels)
        ])

        # 3) Декодер: разбиваем на ConvTranspose2d (upsample) и обычные conv-блоки
        # up_convs[i][lvl] ожидает ровно `C = filters[-1]` при lvl==0, иначе filters[-lvl]
        self.up_convs = nn.ModuleList([
            nn.ModuleList([
                nn.ConvTranspose2d(
                    (filters[-1] if lvl == 0 else filters[-lvl]),
                    filters[-1 - lvl],
                    kernel_size=2, stride=2
                )
                for lvl in range(num_levels)
            ])
            for _ in range(out_channels)
        ])

        # conv_blocks[i][lvl] берёт на вход конкатенацию (skip_mod, y) из 2×(filters[-1-lvl]) каналов
        self.conv_blocks = nn.ModuleList([
            nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(filters[-1 - lvl] * 2, filters[-1 - lvl], kernel_size, padding=kernel_size//2),
                    activation(inplace=True),
                    nn.Conv2d(filters[-1 - lvl], filters[-1 - lvl], kernel_size, padding=kernel_size//2),
                    activation(inplace=True)
                )
                for lvl in range(num_levels)
            ])
            for _ in range(out_channels)
        ])

        # 4) Heads: по одному 1×1 conv на каждый выходной канал
        self.heads = nn.ModuleList([
            nn.Conv2d(filters[0], 1, kernel_size=1)
            for _ in range(out_channels)
        ])

        self.final_activation = final_activation

    def forward(self, x: torch.Tensor, bc_map: torch.Tensor) -> torch.Tensor:
        # x: [B, in_channels, H, W]
        # bc_map: [B, bc_channels] или [B, bc_channels, H_bc, W_bc]
        B, _, H0, W0 = x.shape

        # --- 1. Энкодер: собираем skip-тензоры ---
        skips = []
        h = x
        for enc_block in self.encoder:
            h = enc_block(h)
            skips.append(h)
            # просто MaxPool2d для уменьшения spatial без возврата индексов
            h = F.max_pool2d(h, 2, 2)
        deep = h  # [B, filters[-1], H0/2^L, W0/2^L]

        # --- 2. Декодер для каждого выходного канала i ---
        outs = []
        for i in range(len(self.up_convs)):
            y = deep  # начинаем с “глубокого” представления

            for lvl in range(len(self.up_convs[i])):
                # 2.1 Апсемплируем y
                y = self.up_convs[i][lvl](y)

                # 2.2 Берём skip-тензор соответствующего уровня
                skip = skips[-1 - lvl]  # последовательность снизу вверх

                # 2.3 SPADE-модуляция этого skip на основе bc_map
                skip_mod = self.spade_blocks[i][lvl](skip, bc_map)

                # 2.4 Проверяем размеры (y, skip_mod) и при необходимости ресайзим y
                if y.shape[2:] != skip_mod.shape[2:]:
                    y = F.interpolate(y, size=skip_mod.shape[2:], mode='nearest')

                # 2.5 Конкатенируем по канальному измерению
                cat = torch.cat([skip_mod, y], dim=1)  # [B, 2*filters[-1-lvl], H_lvl, W_lvl]

                # 2.6 Пропускаем через последовательный conv-блок
                y = self.conv_blocks[i][lvl](cat)
                # теперь y имеет ровно filters[-1 - lvl] каналов

            # --- 2.7 Head и возврат к исходному разрешению ---
            pred = self.heads[i](y)  # [B, 1, H_L0, W_L0]
            pred = F.interpolate(pred, size=(H0, W0), mode='bilinear', align_corners=False)
            outs.append(pred)

        # --- 3. Собираем все out_channels в один тензор и применяем финальную активацию ---
        out = torch.cat(outs, dim=1)  # [B, out_channels, H0, W0]
        return self.final_activation(out) if self.final_activation else out
