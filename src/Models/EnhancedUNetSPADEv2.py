import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Type

################################################################
# --- auxiliary layers --------------------------------------- #
################################################################

def conv3x3(in_ch: int, out_ch: int, ks: int = 3) -> nn.Conv2d:
    assert ks % 2 == 1
    return nn.Conv2d(in_ch, out_ch, ks, padding=ks // 2, bias=False)


class LayerNorm2d(nn.LayerNorm):
    """LayerNorm over channel dim for 4-D tensors [B,C,H,W]."""
    def forward(self, x):  # type: ignore[override]
        return super().forward(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class SEBlock(nn.Module):
    def __init__(self, ch: int, r: int = 16):
        super().__init__()
        hidden = max(ch // r, 4)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(ch, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, ch), nn.Sigmoid()
        )
    def forward(self, x):
        w = self.fc(x).unsqueeze(-1).unsqueeze(-1)
        return x * w


class FiLMLayer(nn.Module):
    def __init__(self, bc_dim: int, feat_ch: int, hid: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(bc_dim, hid), nn.ReLU(inplace=True), nn.Linear(hid, 2 * feat_ch))
    def forward(self, feat, bc_vec):
        gamma, beta = torch.chunk(self.mlp(bc_vec), 2, 1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        return feat * (1 + gamma) + beta


class SPADE(nn.Module):
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
        elif bc_map.dim() == 3:
            bc_map = bc_map.unsqueeze(-1).expand(B, bc_map.size(1), H, W)
        elif bc_map.shape[2:] != (H, W):
            bc_map = F.interpolate(bc_map, size=(H, W), mode='nearest')
        act = self.shared(bc_map)
        return self.norm(feat) * (1 + self.gamma(act)) + self.beta(act)

################################################################
# --- main architecture -------------------------------------- #
################################################################

class EnhancedUNetSPADEv2(nn.Module):
    """UNet with bilinear-upsample, SPADE+FiLM BC, SE-skip, cross-talk & dilated RF."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        fc_in_channels: int = 6,
        filters: List[int] = [32, 64, 128, 256, 512],
        decoder_filters: Optional[List[int]] = None,
        activation: Type[nn.Module] = nn.ReLU,
        final_activation: Optional[nn.Module] = None,
        device: str = 'cpu'
    ) -> None:
        super().__init__()
        if decoder_filters is None:
            decoder_filters = filters[::-1]
        assert filters[-1] == decoder_filters[0]
        self.device = torch.device(device)
        L = len(filters)
        act_fn = activation(inplace=True)

        # 1) Encoder --------------------------------------------------
        self.enc_blocks = nn.ModuleList()
        enc_in = in_channels
        for f in filters:
            self.enc_blocks.append(
                nn.Sequential(conv3x3(enc_in, f), act_fn, conv3x3(f, f), LayerNorm2d(f), act_fn)
            )
            enc_in = f

        # 2) Bottle with dilated convs -------------------------------
        self.bottle = nn.Sequential(
            conv3x3(filters[-1], filters[-1]), act_fn,
            conv3x3(filters[-1], filters[-1]), act_fn,
            nn.Conv2d(filters[-1], filters[-1], 3, padding=2, dilation=2), act_fn,
            nn.Conv2d(filters[-1], filters[-1], 3, padding=4, dilation=4), act_fn,
        )

        # 3) Decoder components --------------------------------------
        self.upsamplers = nn.ModuleList()
        self.se_skips = nn.ModuleList()
        self.spade_blocks = nn.ModuleList()
        self.cross_talk = nn.ModuleList()
        self.film_blocks = nn.ModuleList()

        for lvl in range(L):
            dec_in = decoder_filters[lvl]
            dec_out = decoder_filters[lvl + 1] if lvl + 1 < L else decoder_filters[-1]
            skip_ch = filters[-1 - lvl]
            concat_ch = dec_out + skip_ch

            # upsample + conv3x3
            self.upsamplers.append(
                nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), conv3x3(dec_in, dec_out))
            )
            # skip branch modules
            self.se_skips.append(SEBlock(skip_ch))
            self.spade_blocks.append(SPADE(skip_ch, fc_in_channels))
            # cross-talk projects concat → dec_out
            self.cross_talk.append(nn.Conv2d(concat_ch, dec_out, 1))
            # FiLM on decoder stream
            self.film_blocks.append(FiLMLayer(fc_in_channels, dec_out))

        # 4) Output head ---------------------------------------------
        self.head = nn.Conv2d(decoder_filters[-1], out_channels, 1)
        self.final_activation = final_activation
        self.to(self.device)

    # ----------------------- forward helpers ------------------------
    def _encode(self, x):
        skips = []
        h = x
        for blk in self.enc_blocks:
            h = blk(h)
            skips.append(h)
            h = F.max_pool2d(h, 2)
        return self.bottle(h), skips

    # ----------------------------------------------------------------
    def forward(self, x: torch.Tensor, bc_vec: torch.Tensor):
        h, skips = self._encode(x)
        L = len(self.upsamplers)
        for lvl in range(L):
            h = self.upsamplers[lvl](h)
            skip = skips[-1 - lvl]
            if h.shape[-2:] != skip.shape[-2:]:
                h = F.interpolate(h, size=skip.shape[-2:], mode='bilinear', align_corners=False)
            skip = self.se_skips[lvl](skip)
            skip = self.spade_blocks[lvl](skip, bc_vec)
            h = torch.cat([h, skip], 1)
            h = self.cross_talk[lvl](h)
            h = self.film_blocks[lvl](h, bc_vec)
        out = self.head(h)
        return self.final_activation(out) if self.final_activation else out

################################################################
# factory -----------------------------------------------------
################################################################

def EnhancedUNetSPADEv2_factory(**kwargs):
    return EnhancedUNetSPADEv2(**kwargs)
