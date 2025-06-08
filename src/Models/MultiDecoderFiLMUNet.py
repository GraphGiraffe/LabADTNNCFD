import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Type

#############################################
# Utility blocks
#############################################

def conv_block(in_channels: int, out_channels: int, kernel_size: int = 3,
               activation: Type[nn.Module] = nn.ReLU, bn: bool = True) -> nn.Sequential:
    """Conv → Act → (BN) with padding."""
    assert kernel_size % 2 == 1, "kernel_size must be odd"
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2),
              activation(inplace=True)]
    if bn:
        layers.append(nn.BatchNorm2d(out_channels))
    return nn.Sequential(*layers)


def double_conv(in_ch: int, out_ch: int, activation: Type[nn.Module] = nn.ReLU) -> nn.Sequential:
    """Two conv_block's as in classical UNet."""
    mid = out_ch
    return nn.Sequential(
        conv_block(in_ch, mid, activation=activation),
        conv_block(mid, out_ch, activation=activation)
    )

#############################################
# FiLM generator
#############################################

class FiLMGenerator(nn.Module):
    """Generates γ, β for FiLM from BC vector for one decoder depth."""
    def __init__(self, bc_dim: int, feat_channels: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(bc_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 2 * feat_channels)  # γ and β
        )

    def forward(self, bc_vec: torch.Tensor):
        gamma_beta = self.net(bc_vec)  # (B, 2C)
        gamma, beta = torch.chunk(gamma_beta, 2, dim=1)
        return gamma.unsqueeze(-1).unsqueeze(-1), beta.unsqueeze(-1).unsqueeze(-1)  # broadcast

#############################################
# Main model
#############################################

class MultiDecoderFiLMUNet(nn.Module):
    """UNet encoder with **separate decoders per output channel** and FiLM conditioning.

    For each output (u, v, p, T) there is an independent decoder branch.
    Each decoder depth receives γ/β parameters computed from the BC vector via
    a small MLP (FiLMGenerator).  This allows branch‑specific BC influence.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        fc_in_channels: int = 6,
        encoder_filters: List[int] = [32, 64, 128, 256],
        decoder_filters: List[int] = [256, 128, 64, 32],
        kernel_size: int = 3,
        activation: Type[nn.Module] = nn.ReLU,
        final_activation: Optional[nn.Module] = None,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        assert len(encoder_filters) == len(decoder_filters), "Depth mismatch"
        assert encoder_filters[-1] == decoder_filters[0], "Decoder start channels must equal deepest encoder channels"

        self.device = torch.device(device)
        self.final_activation = final_activation
        depth = len(encoder_filters)

        # 1) Encoder
        self.enc_blocks = nn.ModuleList()
        enc_in = in_channels
        for f in encoder_filters:
            self.enc_blocks.append(double_conv(enc_in, f, activation))
            enc_in = f

        # 2) Decoder branches (one per field)
        self.upsamplers = nn.ModuleList([nn.ModuleList() for _ in range(out_channels)])
        self.dec_blocks = nn.ModuleList([nn.ModuleList() for _ in range(out_channels)])
        self.film_generators = nn.ModuleList([nn.ModuleList() for _ in range(out_channels)])
        for branch in range(out_channels):
            for lvl in range(depth):
                # upsample (bilinear), then concat skip => channels dec_in = dec_prev + skip
                self.upsamplers[branch].append(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False))
                in_ch = decoder_filters[lvl] + encoder_filters[-1 - lvl]
                out_ch = decoder_filters[lvl + 1] if lvl + 1 < depth else decoder_filters[-1]
                self.dec_blocks[branch].append(double_conv(in_ch, out_ch, activation))
                # FiLM for output of this conv block (out_ch channels)
                self.film_generators[branch].append(FiLMGenerator(fc_in_channels, out_ch, hidden=128))
        # 3) Output heads
        self.heads = nn.ModuleList([
            nn.Conv2d(decoder_filters[-1], 1, kernel_size=1) for _ in range(out_channels)
        ])

        self.to(self.device)

    # ---------------------------------------------
    # Helpers
    # ---------------------------------------------
    def _encode(self, x: torch.Tensor):
        skips = []
        h = x
        for block in self.enc_blocks:
            h = block(h)
            skips.append(h)
            h = F.max_pool2d(h, 2)
        return h, skips

    def forward(self, x: torch.Tensor, bc_vec: torch.Tensor) -> torch.Tensor:
        """x: [B, in_channels, H, W]; bc_vec: [B, fc_in_channels]."""
        deep, skips = self._encode(x)
        depth = len(self.enc_blocks)
        outs = []
        for b in range(len(self.heads)):
            h = deep
            for lvl in range(depth):
                h = self.upsamplers[b][lvl](h)
                skip = skips[-1 - lvl]
                # align spatial sizes if rounding occurred
                if h.shape[-2:] != skip.shape[-2:]:
                    h = F.interpolate(h, size=skip.shape[-2:], mode="bilinear", align_corners=False)
                h = torch.cat([h, skip], dim=1)
                h = self.dec_blocks[b][lvl](h)
                gamma, beta = self.film_generators[b][lvl](bc_vec)
                h = gamma * h + beta  # FiLM modulation
            pred = self.heads[b](h)
            outs.append(pred)
        out = torch.cat(outs, dim=1)
        if self.final_activation is not None:
            out = self.final_activation(out)
        return out

#############################################################
# Factory for pipeline discovery
#############################################################

def MultiDecoderFiLMUNet_factory(**kwargs):
    return MultiDecoderFiLMUNet(**kwargs)
