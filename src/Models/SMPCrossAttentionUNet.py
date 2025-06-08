import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from src.Models.AutoEncoder import create_layer
from typing import List, Optional, Type


def create_decoder_block(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    wn: bool = True,
    bn: bool = True,
    activation: Type[nn.Module] = nn.ReLU,
    layers: int = 2,
    final_layer: bool = False
) -> nn.Sequential:
    """
    Creates a decoder block of sequential ConvTranspose2d layers.
    """
    layers_list = []
    for i in range(layers):
        inp = in_channels if i == 0 else out_channels
        outp = out_channels
        bn_flag = bn if not (i == layers - 1 and final_layer) else False
        act = activation if not (i == layers - 1 and final_layer) else None
        layers_list.append(
            create_layer(inp, outp, kernel_size, wn, bn_flag, act, nn.ConvTranspose2d)
        )
    return nn.Sequential(*layers_list)


class CrossAttentionBC(nn.Module):
    """
    Cross-attention module with multi-head, LayerNorm and Feed-Forward.
    """
    def __init__(self,
                 fc_in_channels: int,
                 deep_channels: int,
                 embed_dim: int,
                 attn_heads: int = 4):
        super().__init__()
        assert embed_dim % attn_heads == 0, "embed_dim must be divisible by attn_heads"
        self.embed_dim = embed_dim
        self.heads = attn_heads
        self.head_dim = embed_dim // attn_heads
        self.attn_q = nn.Linear(fc_in_channels, embed_dim)
        self.attn_k = nn.Conv2d(deep_channels, embed_dim, kernel_size=1)
        self.attn_v = nn.Conv2d(deep_channels, embed_dim, kernel_size=1)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.scale = self.head_dim ** 0.5

    def forward(self, deep: torch.Tensor, x_fc: torch.Tensor) -> torch.Tensor:
        B, C, H, W = deep.shape
        q = self.attn_q(x_fc)
        k = self.attn_k(deep).view(B, self.embed_dim, -1)
        v = self.attn_v(deep).view(B, self.embed_dim, -1)
        q = q.view(B, self.heads, self.head_dim).unsqueeze(2)
        k = k.view(B, self.heads, self.head_dim, -1).permute(0,1,3,2)
        v = v.view(B, self.heads, self.head_dim, -1).permute(0,1,3,2)
        scores = torch.matmul(q, k.transpose(-1,-2)) / self.scale
        weights = torch.softmax(scores, dim=-1)
        attn = torch.matmul(weights, v).squeeze(2)
        attn_flat = attn.reshape(B, self.embed_dim)
        x1 = self.norm1(attn_flat)
        ff_out = self.ff(x1)
        x2 = self.norm2(x1 + ff_out)
        return x2.view(B, self.embed_dim, 1, 1).expand(-1, -1, H, W)


class SMPCrossAttentionUNet(nn.Module):
    def __init__(
        self,
        encoder_name: str,
        in_channels: int,
        out_channels: int,
        fc_in_channels: int,
        attn_heads: int = 4,
        decoder_filters: List[int] = [256, 128, 64, 32],
        kernel_size: int = 3,
        activation: Optional[Type[nn.Module]] = nn.ReLU,
        final_activation: Optional[nn.Module] = None,
        encoder_weights: Optional[str] = 'imagenet'
    ):
        super().__init__()
        self.encoder = smp.encoders.get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=len(decoder_filters) + 1,
            weights=encoder_weights
        )
        enc_channels = self.encoder.out_channels
        self.skip_channels = list(reversed(enc_channels[:-1]))[:len(decoder_filters)]
        self.deep_ch = enc_channels[-1]
        self.decoder_filters = decoder_filters
        # Attention modules per decoder level
        self.cross_attns = nn.ModuleList([
            CrossAttentionBC(
                fc_in_channels=fc_in_channels,
                deep_channels=self.deep_ch,
                embed_dim=decoder_filters[i],
                attn_heads=attn_heads
            ) for i in range(len(decoder_filters))
        ])
        # Build decoders
        self.decoders = nn.ModuleList()
        for _ in range(out_channels):
            blocks = []
            prev_ch = decoder_filters[0]
            for i, f in enumerate(decoder_filters):
                in_ch = prev_ch + self.skip_channels[i] + decoder_filters[i]
                blocks.append(
                    create_decoder_block(
                        in_channels=in_ch,
                        out_channels=f,
                        kernel_size=kernel_size,
                        wn=True,
                        bn=True,
                        activation=activation,
                        layers=2,
                        final_layer=(i == len(decoder_filters) - 1)
                    )
                )
                prev_ch = f
            self.decoders.append(nn.Sequential(*blocks))
        self.heads = nn.ModuleList([
            nn.Conv2d(decoder_filters[-1], 1, kernel_size=1)
            for _ in range(out_channels)
        ])
        self.final_activation = final_activation

    def forward(self, x: torch.Tensor, x_fc: torch.Tensor) -> torch.Tensor:
        B, _, H0, W0 = x.shape
        features = self.encoder(x)
        deep = features[-1]
        outs = []
        # iterate over decoders and heads
        for dec, head in zip(self.decoders, self.heads):
            # initial attention map at deepest level
            attn0 = self.cross_attns[0](deep, x_fc)
            y = F.interpolate(attn0, size=features[-2].shape[-2:], mode='nearest')
            # decode through levels with attention
            for j, block in enumerate(dec):
                skip = features[-2 - j]
                # upsample y to match current skip spatial size
                y = F.interpolate(y, size=skip.shape[-2:], mode='nearest')
                # compute attention for this level and upsample
                attn_feat = self.cross_attns[j](deep, x_fc)
                attn_feat = F.interpolate(attn_feat, size=skip.shape[-2:], mode='nearest')
                # concatenate skip, y, attention features
                y = torch.cat([skip, y, attn_feat], dim=1)
                y = block(y)
            y = head(y)
            y = F.interpolate(y, size=(H0, W0), mode='bilinear', align_corners=False)
            outs.append(y)
        out = torch.cat(outs, dim=1)
        if self.final_activation is not None:
            out = self.final_activation(out)
        return out
