import torch
import torch.nn as nn
import torch.nn.functional as F
from segmentation_models_pytorch.encoders import get_encoder
from typing import List, Optional, Type


class CrossAttention2D(nn.Module):
    def __init__(self, fc_dim: int, spatial_ch: int, embed_dim: int, heads: int = 4):
        super().__init__()
        assert embed_dim % heads == 0
        self.heads = heads
        self.scale = (embed_dim // heads) ** -0.5
        self.to_q = nn.Linear(fc_dim, embed_dim)
        self.to_k = nn.Conv2d(spatial_ch, embed_dim, 1)
        self.to_v = nn.Conv2d(spatial_ch, embed_dim, 1)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, spatial: torch.Tensor, fc: torch.Tensor) -> torch.Tensor:
        B, _, H, W = spatial.shape
        q = self.to_q(fc).view(B, self.heads, -1)
        k = self.to_k(spatial).view(B, self.heads, -1, H * W).permute(0, 1, 3, 2)
        v = self.to_v(spatial).view(B, self.heads, -1, H * W).permute(0, 1, 3, 2)
        attn = (q.unsqueeze(2) @ k.transpose(-1, -2)) * self.scale
        attn = attn.softmax(-1)
        out = (attn @ v).squeeze(2).reshape(B, -1)
        x = self.norm1(out)
        x = self.norm2(x + self.ff(x))
        return x.view(B, -1, 1, 1).expand(B, x.shape[1], H, W)


class ImprovedAttentionUNet(nn.Module):
    def __init__(
        self,
        encoder_name: str,
        in_channels: int,
        out_channels: int,
        fc_dim: int,
        attn_heads: int = 4,
        decoder_filters: List[int] = [256, 128, 64, 32],
        kernel_size: int = 3,
        activation: Optional[Type[nn.Module]] = nn.ReLU,
        final_activation: Optional[nn.Module] = None,
        encoder_weights: Optional[str] = 'imagenet'
    ):
        super().__init__()
        self.encoder = get_encoder(
            encoder_name, in_channels=in_channels,
            depth=len(decoder_filters)+1, weights=encoder_weights
        )
        enc_ch = self.encoder.out_channels
        self.skip_ch = list(reversed(enc_ch[:-1]))[:len(decoder_filters)]
        self.deep_ch = enc_ch[-1]

        # Attention per level
        self.attns = nn.ModuleList([
            CrossAttention2D(fc_dim, self.skip_ch[i], decoder_filters[i], heads=attn_heads)
            for i in range(len(decoder_filters))
        ])

        # Decoder: separate ConvTranspose and Conv blocks
        self.up_convs = nn.ModuleList()
        self.conv_blocks = nn.ModuleList()
        for lvl, f in enumerate(decoder_filters):
            # transposed conv upsamples by 2
            in_ch = self.deep_ch if lvl == 0 else decoder_filters[lvl-1]
            self.up_convs.append(
                nn.ConvTranspose2d(in_ch, f, kernel_size=2, stride=2)
            )
            # conv block after concat(skip, y, attn)
            total_ch = f + self.skip_ch[lvl] + decoder_filters[lvl]
            self.conv_blocks.append(
                nn.Sequential(
                    nn.Conv2d(total_ch, f, kernel_size, padding=kernel_size//2),
                    activation(inplace=True),
                    nn.Conv2d(f, f, kernel_size, padding=kernel_size//2),
                    activation(inplace=True)
                )
            )

        self.heads = nn.ModuleList([nn.Conv2d(decoder_filters[-1], 1, 1)
                                    for _ in range(out_channels)])
        self.final_activation = final_activation

    def forward(self, x: torch.Tensor, x_fc: torch.Tensor) -> torch.Tensor:
        B, _, H0, W0 = x.shape
        feats = self.encoder(x)          # list of tensors
        deep = feats[-1]
        y = deep
        outs = []
        # decoder loop
        for lvl in range(len(self.up_convs)):
            # upsample y
            y = self.up_convs[lvl](y)
            # ensure match skip size
            skip = feats[-2-lvl]
            if y.shape[2:] != skip.shape[2:]:
                y = F.interpolate(y, size=skip.shape[2:], mode='nearest')
            # attention on skip
            attn = self.attns[lvl](skip, x_fc)
            # concat and conv
            y = torch.cat([skip, y, attn], dim=1)
            y = self.conv_blocks[lvl](y)
        # heads and resize
        for head in self.heads:
            pred = head(y)
            outs.append(F.interpolate(pred, size=(H0, W0),
                                      mode='bilinear', align_corners=False))
        out = torch.cat(outs, dim=1)
        return self.final_activation(out) if self.final_activation else out
