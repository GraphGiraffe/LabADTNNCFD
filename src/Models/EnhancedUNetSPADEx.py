import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Type

####################################################################
# ---------- helper functions & layers --------------------------- #
####################################################################

def conv3x3(in_ch: int, out_ch: int, ks: int = 3) -> nn.Conv2d:
    assert ks % 2 == 1
    return nn.Conv2d(in_ch, out_ch, ks, padding=ks // 2, bias=False)

class LayerNorm2d(nn.LayerNorm):
    """Channel-wise LayerNorm for NCHW."""
    def forward(self, x):  # type: ignore[override]
        return super().forward(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

class FiLMLayer(nn.Module):
    """Global FiLM from BC-vector."""
    def __init__(self, bc_dim: int, feat_ch: int, hid: int = 128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(bc_dim, hid), nn.ReLU(inplace=True), nn.Linear(hid, 2 * feat_ch))
    def forward(self, feat, bc_vec):
        gamma, beta = torch.chunk(self.net(bc_vec), 2, 1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        return feat * (1 + gamma) + beta

class SPADE(nn.Module):
    """SPADE conditioning for skip-tensor."""
    def __init__(self, feat_ch: int, bc_dim: int, hid: int = 64):
        super().__init__()
        self.norm = LayerNorm2d(feat_ch)
        self.shared = nn.Sequential(nn.Conv2d(bc_dim, hid, 3, padding=1), nn.ReLU(inplace=True))
        self.gamma = nn.Conv2d(hid, feat_ch, 3, padding=1)
        self.beta = nn.Conv2d(hid, feat_ch, 3, padding=1)
    def forward(self, feat, bc_map):
        B, _, H, W = feat.shape
        if bc_map.dim() == 2:
            bc_map = bc_map.view(B, -1, 1, 1).expand(B, bc_map.size(1), H, W)
        else:
            bc_map = F.interpolate(bc_map, size=(H, W), mode='nearest') if bc_map.shape[-2:] != (H, W) else bc_map
        shared = self.shared(bc_map)
        return self.norm(feat) * (1 + self.gamma(shared)) + self.beta(shared)

class CrossTalk(nn.Module):
    """Lightweight information mixing across output fields."""
    def __init__(self, ch: int):
        super().__init__()
        # depthwise separable 1×1 conv -> cheap mixing
        self.ctx = nn.Sequential(nn.Conv2d(ch, ch, 1, groups=ch//4 or 1, bias=False), LayerNorm2d(ch), nn.GELU())
    def forward(self, x):
        return self.ctx(x)

####################################################################
# --------------------- main network ----------------------------- #
####################################################################

class EnhancedUNetSPADEx(nn.Module):
    """Overhauled UNet-SPADE with:
        • Bilinear upsample + conv (anti-checkerboard)
        • Dual BC-conditioning (SPADE on skips + FiLM on decoder stream)
        • Cross-talk after each concat for u,v,p,T coherence
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        fc_in_channels: int = 6,
        filters: List[int] = [32, 64, 128, 256, 512],
        activation: Type[nn.Module] = nn.ReLU,
        final_activation: Optional[nn.Module] = None,
        device: str = 'cpu'
    ) -> None:
        super().__init__()
        self.device = torch.device(device)
        depth = len(filters)
        act = lambda: activation(inplace=True)

        # ----- encoder -----
        self.enc_blocks = nn.ModuleList()
        ch_in = in_channels
        for f in filters:
            self.enc_blocks.append(nn.Sequential(conv3x3(ch_in, f), act(), conv3x3(f, f), LayerNorm2d(f), act()))
            ch_in = f

        # ----- decoder component lists -----
        self.up_convs = nn.ModuleList()
        self.skip_spades = nn.ModuleList()
        self.skip_cross = nn.ModuleList()
        self.decoder_films = nn.ModuleList()

        dec_filters = filters[::-1]
        for lvl in range(depth):
            # upsample conv (bilinear + 3×3 conv)
            in_ch = dec_filters[lvl]
            out_ch = dec_filters[lvl + 1] if lvl + 1 < depth else dec_filters[-1]
            self.up_convs.append(nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                conv3x3(in_ch, out_ch), act()
            ))
            skip_ch = filters[-1 - lvl]
            self.skip_spades.append(SPADE(skip_ch, fc_in_channels))
            self.skip_cross.append(nn.Conv2d(out_ch + skip_ch, out_ch, 1))  # cross-talk conv
            self.decoder_films.append(FiLMLayer(fc_in_channels, out_ch))

        # ----- head -----
        self.head = nn.Conv2d(dec_filters[-1], out_channels, 1)
        self.final_activation = final_activation
        self.to(self.device)

    # ------------------ forward ------------------
    def _encode(self, x):
        skips = []
        h = x
        for blk in self.enc_blocks:
            h = blk(h)
            skips.append(h)
            h = F.max_pool2d(h, 2)
        return h, skips

    def forward(self, x: torch.Tensor, bc_vec: torch.Tensor):
        h, skips = self._encode(x)
        depth = len(self.up_convs)
        for lvl in range(depth):
            h = self.up_convs[lvl](h)
            skip = skips[-1 - lvl]
            # align spatial
            if h.shape[-2:] != skip.shape[-2:]:
                h = F.interpolate(h, size=skip.shape[-2:], mode='bilinear', align_corners=False)
            # BC to skip via SPADE
            skip = self.skip_spades[lvl](skip, bc_vec)
            # concat + cross-talk conv
            h = torch.cat([h, skip], 1)
            h = self.skip_cross[lvl](h)
            # FiLM modulation on decoder stream
            h = self.decoder_films[lvl](h, bc_vec)
        out = self.head(h)
        return self.final_activation(out) if self.final_activation else out

####################################################################
# factory ---------------------------------------------------------#
####################################################################

def EnhancedUNetSPADEx_factory(**kwargs):
    return EnhancedUNetSPADEx(**kwargs)
