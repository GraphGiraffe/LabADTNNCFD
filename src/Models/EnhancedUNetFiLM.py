import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Type

###############################################
# ------------- building blocks ------------- #
###############################################


def conv3x3(in_ch: int, out_ch: int, kernel_size: int = 3) -> nn.Conv2d:
    """3×3 conv with padding (no bias)."""
    assert kernel_size % 2 == 1
    return nn.Conv2d(in_ch, out_ch, kernel_size, padding=kernel_size // 2, bias=False)


class ResidualSEBlock(nn.Module):
    """Residual block  (conv‑BN‑ReLU)×2  +  Squeeze‑Excitation."""

    def __init__(self, in_ch: int, out_ch: int, activation: Type[nn.Module] = nn.ReLU, reduction: int = 16):
        super().__init__()
        self.conv1 = conv3x3(in_ch, out_ch)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.act = activation(inplace=True)
        self.conv2 = conv3x3(out_ch, out_ch)
        self.bn2 = nn.BatchNorm2d(out_ch)
        # Squeeze‑Excitation
        hidden = max(out_ch // reduction, 4)
        self.se_fc1 = nn.Linear(out_ch, hidden)
        self.se_fc2 = nn.Linear(hidden, out_ch)
        # projection if channels mismatch
        self.res_proj = None if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, 1, bias=False)

    def forward(self, x: torch.Tensor):
        res = x if self.res_proj is None else self.res_proj(x)
        h = self.conv1(x)
        h = self.bn1(h)
        h = self.act(h)
        h = self.conv2(h)
        h = self.bn2(h)
        # SE calibration
        se = F.adaptive_avg_pool2d(h, 1).flatten(1)
        se = self.act(self.se_fc1(se))
        se = torch.sigmoid(self.se_fc2(se)).unsqueeze(-1).unsqueeze(-1)
        h = h * se
        h = self.act(h + res)
        return h


class FiLMLayer(nn.Module):
    """Generates FiLM (γ, β) from BC vector and applies to feature map."""

    def __init__(self, bc_dim: int, feat_ch: int, hidden: int = 128):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(bc_dim, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, 2 * feat_ch)
        )

    def forward(self, feat: torch.Tensor, bc_vec: torch.Tensor):
        gamma_beta = self.fc(bc_vec)  # (B, 2C)
        gamma, beta = torch.chunk(gamma_beta, 2, dim=1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        return gamma * feat + beta


class AttentionGate(nn.Module):
    """Attention‑UNet style gate that filters skip tensor."""

    def __init__(self, F_g: int, F_l: int, F_int: int):
        super().__init__()
        self.W_g = nn.Sequential(nn.Conv2d(F_g, F_int, 1, bias=False), nn.BatchNorm2d(F_int))
        self.W_x = nn.Sequential(nn.Conv2d(F_l, F_int, 1, bias=False), nn.BatchNorm2d(F_int))
        self.psi = nn.Sequential(nn.Conv2d(F_int, 1, 1), nn.Sigmoid())
        self.act = nn.ReLU(inplace=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor):
        # ‼️ align spatial sizes BEFORE 1×1 convs (odd shapes 38/39 etc.)
        if g.shape[-2:] != x.shape[-2:]:
            g = F.interpolate(g, size=x.shape[-2:], mode='bilinear', align_corners=False)
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        # after convs dims match → safe add
        psi = self.psi(self.act(g1 + x1))
        return x * psi

###############################################
# ----------------- Main UNet --------------- #
###############################################


class EnhancedUNetFiLM(nn.Module):
    """Enhanced UNet with Attention‑gated skips **and** FiLM BC modulation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        fc_in_channels: int = 6,
        encoder_filters: List[int] = [32, 64, 128, 256, 512],
        decoder_filters: List[int] = [512, 256, 128, 64, 32],
        activation: Type[nn.Module] = nn.ReLU,
        final_activation: Optional[nn.Module] = None,
        device: str = 'cpu',
    ) -> None:
        super().__init__()
        assert len(encoder_filters) == len(decoder_filters), 'encoder/decoder depth mismatch'
        assert encoder_filters[-1] == decoder_filters[0], 'decoder must start with deepest encoder channels'
        depth = len(encoder_filters)
        self.device = torch.device(device)
        self.final_activation = final_activation

        # -------- Encoder --------
        self.enc_blocks = nn.ModuleList([
            ResidualSEBlock(in_channels if i == 0 else encoder_filters[i - 1], encoder_filters[i], activation)
            for i in range(depth)
        ])

        # -------- Decoder (components lists) --------
        self.upsamplers = nn.ModuleList([
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
            for _ in range(depth)
        ])
        self.dec_blocks = nn.ModuleList()
        self.att_gates = nn.ModuleList()
        self.film_layers = nn.ModuleList()
        for lvl in range(depth):
            in_ch = decoder_filters[lvl] + encoder_filters[-1 - lvl]
            out_ch = decoder_filters[lvl + 1] if lvl + 1 < depth else decoder_filters[-1]
            self.dec_blocks.append(ResidualSEBlock(in_ch, out_ch, activation))
            self.att_gates.append(AttentionGate(decoder_filters[lvl], encoder_filters[-1 - lvl], out_ch // 2))
            self.film_layers.append(FiLMLayer(fc_in_channels, out_ch))

        # -------- Head --------
        self.head = nn.Conv2d(decoder_filters[-1], out_channels, 1)
        self.to(self.device)

    # -------- Helper: Encoder pass --------
    def _encode(self, x: torch.Tensor):
        skips = []
        h = x
        for block in self.enc_blocks:
            h = block(h)
            skips.append(h)
            h = F.max_pool2d(h, 2)
        return h, skips

    # -------- Forward --------
    def forward(self, x: torch.Tensor, bc_vec: torch.Tensor):
        h, skips = self._encode(x)
        depth = len(self.dec_blocks)
        for lvl in range(depth):
            # 1) upsample
            h = self.upsamplers[lvl](h)
            skip = skips[-1 - lvl]
            # 2) align BEFORE gate (odd dims)
            if h.shape[-2:] != skip.shape[-2:]:
                h = F.interpolate(h, size=skip.shape[-2:], mode='bilinear', align_corners=False)
            # 3) gated skip
            skip = self.att_gates[lvl](h, skip)
            # 4) concat & residual block
            h = torch.cat([h, skip], dim=1)
            h = self.dec_blocks[lvl](h)
            # 5) FiLM condition
            h = self.film_layers[lvl](h, bc_vec)
        out = self.head(h)
        if self.final_activation is not None:
            out = self.final_activation(out)
        return out

###############################################################
# Factory for importlib lookup
###############################################################

def EnhancedUNetFiLM_factory(**kwargs):
    return EnhancedUNetFiLM(**kwargs)
