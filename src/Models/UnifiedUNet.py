import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Type

# =============================================================================
#  Unified UNet with a single decoder head + LayerNorm + FiLM‑like BC injection
#    • Общий decoder → экономия параметров и более согласованные поля
#    • GroupNorm(1,C) ≈ LayerNorm для стабилизации
#    • Для каждого уровня декодера γ,β генерируются из BC‑вектора так, чтобы
#      их длина = числу каналов skip‑тензора (исправляет ошибку reshape)
# =============================================================================

try:
    from src.Models.EnhancedUNet import create_encoder, create_layer
except ModuleNotFoundError:
    # Минималистичные заглушки для автономного теста
    def create_layer(in_channels, out_channels, kernel_size, wn=True, bn=True,
                     activation=nn.ReLU, convolution=nn.Conv2d):
        assert kernel_size % 2 == 1
        layer = [convolution(in_channels, out_channels, kernel_size,
                             padding=kernel_size // 2)]
        if activation is not None:
            layer.append(activation(inplace=True))
        if bn:
            layer.append(nn.BatchNorm2d(out_channels))
        return nn.Sequential(*layer)

    def create_encoder(in_channels: int, filters: List[int], kernel_size: int,
                       wn: bool, bn: bool, activation: Type[nn.Module],
                       layers: int):
        enc = []
        for i in range(len(filters)):
            blocks = []
            for j in range(layers):
                _in = in_channels if (i == 0 and j == 0) else (
                    filters[i] if j > 0 else filters[i - 1])
                blocks.append(create_layer(_in, filters[i], kernel_size, wn, bn,
                                            activation, nn.Conv2d))
            enc.append(nn.Sequential(*blocks))
        return nn.Sequential(*enc)

class UnifiedUNet(nn.Module):
    """UNet with one shared decoder and LayerNorm; BC injected via FiLM."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        fc_in_channels: int = 6,
        kernel_size: int = 3,
        filters: List[int] = [32, 64, 128, 256, 256],
        layers_per_block: int = 3,
        activation: Optional[Type[nn.Module]] = nn.ReLU,
        device: str = "cpu"
    ):
        super().__init__()
        self.device = torch.device(device)
        self.out_channels = out_channels
        self.encoder = create_encoder(in_channels, filters, kernel_size,
                                      wn=True, bn=True,
                                      activation=activation,
                                      layers=layers_per_block)

        # Число уровней = длине filters
        self.num_levels = len(filters)
        self.level_channels = list(reversed(filters))  # [C_L‑1, …, C0]

        # Γ и Β для каждого уровня генерируем отдельными линейными слоями
        self.gamma_mlps = nn.ModuleList([
            nn.Linear(fc_in_channels, ch) for ch in self.level_channels
        ])
        self.beta_mlps = nn.ModuleList([
            nn.Linear(fc_in_channels, ch) for ch in self.level_channels
        ])

        # ------------------------------------------------------------------
        # Decoder (shared): up‑conv → concat skip → conv‑block (×2 conv)
        # ------------------------------------------------------------------
        self.upconvs = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        for lvl in range(self.num_levels):
            in_ch = filters[-1] if lvl == 0 else filters[-lvl]
            out_ch = filters[-(lvl + 1)]
            self.upconvs.append(
                nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
            )
            block = nn.Sequential(
                nn.Conv2d(out_ch * 2, out_ch, kernel_size, padding=kernel_size // 2),
                nn.GroupNorm(1, out_ch),
                activation(inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size, padding=kernel_size // 2),
                nn.GroupNorm(1, out_ch),
                activation(inplace=True)
            )
            self.dec_blocks.append(block)

        # Head 1×1 conv
        self.final_conv = nn.Conv2d(filters[0], out_channels, kernel_size=1)

    # ---------------------------------------------------------------
    def _enc_forward(self, x):
        skips = []
        h = x
        for enc in self.encoder:
            h = enc(h)
            skips.append(h)
            h = F.max_pool2d(h, 2, 2)
        return h, skips

    def forward(self, x: torch.Tensor, bc_vec: torch.Tensor):
        deep, skips = self._enc_forward(x)
        y = deep
        for lvl in range(self.num_levels):
            # 1) Upsample
            y = self.upconvs[lvl](y)
            skip = skips[-1 - lvl]
            if y.shape[-2:] != skip.shape[-2:]:
                y = F.interpolate(y, size=skip.shape[-2:], mode="nearest")

            # 2) FiLM modulation of skip
            gamma = self.gamma_mlps[lvl](bc_vec).view(bc_vec.size(0), -1, 1, 1)
            beta  = self.beta_mlps[lvl](bc_vec).view(bc_vec.size(0), -1, 1, 1)
            skip_mod = skip * (1 + gamma) + beta

            # 3) Concat and convs
            y = torch.cat([skip_mod, y], dim=1)
            y = self.dec_blocks[lvl](y)
        return self.final_conv(y)

# =============================================================================
#  Physics‑informed loss
# =============================================================================
INLET_VALUE = 3  # метка входа из Label
BODY_VALUE  = 1  # метка тела


def physics_loss(pred: torch.Tensor,
                 target: torch.Tensor,
                 label_map: torch.Tensor,
                 bc_vec: torch.Tensor,
                 lambda_div: float = 1e-2,
                 lambda_bc: float = 5e-2):
    """MSE + λ₁⋅div + λ₂⋅BC для скорости (inlet) и температуры (body)."""
    mse = F.mse_loss(pred, target)

    # Divergence(u,v)
    u, v = pred[:, 0:1], pred[:, 1:2]
    du_dx = torch.gradient(u, dim=3)[0]
    dv_dy = torch.gradient(v, dim=2)[0]
    div_loss = (du_dx + dv_dy).pow(2).mean()

    # BC: скорость на входе и температура тела
    inlet_mask = (label_map == INLET_VALUE).unsqueeze(1).float()
    body_mask  = (label_map == BODY_VALUE ).unsqueeze(1).float()
    u_in    = bc_vec[:, 0].view(-1, 1, 1, 1)
    T_body  = bc_vec[:, 1].view(-1, 1, 1, 1)

    bc_u = (((pred[:, 0:1] - u_in) * inlet_mask) ** 2).sum() / (inlet_mask.sum() + 1e-6)
    bc_T = (((pred[:, 3:4] - T_body) * body_mask) ** 2).sum() / (body_mask.sum() + 1e-6)

    return mse + lambda_div * div_loss + lambda_bc * (bc_u + bc_T)

# =============================================================================
#  Интеграция: в deepcfd_exp.py / Trainer.epoch
#     loss = physics_loss(pred, y, x[:,0], x_fc)
#  и не забудьте передавать label_map (канал Label) вместе с предсказаниями.
# =============================================================================
