import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from src.Models.AutoEncoder import create_layer
from typing import List, Optional, Type


def create_decoder_block(in_channels: int,
                         out_channels: int,
                         kernel_size: int,
                         wn: bool = True,
                         bn: bool = True,
                         activation: Type[nn.Module] = nn.ReLU,
                         layers: int = 2,
                         final_layer: bool = False) -> nn.Sequential:
    """
    Creates a decoder block of sequential ConvTranspose2d layers.
    """
    layers_list = []
    for i in range(layers):
        inp = in_channels if i == 0 else out_channels
        outp = out_channels
        bn_flag = bn if not (i == layers - 1 and final_layer) else False
        act = activation if not (i == layers - 1 and final_layer) else None
        layers_list.append(create_layer(inp, outp, kernel_size, wn, bn_flag, act, nn.ConvTranspose2d))
    return nn.Sequential(*layers_list)


class SMPCrossAttentionUNet(nn.Module):
    def __init__(self,
                 encoder_name: str,
                 in_channels: int,
                 out_channels: int,
                 fc_in_channels: int,
                 attn_heads: int = 4,
                 decoder_filters: List[int] = [256, 128, 64, 32],
                 kernel_size: int = 3,
                 activation: Optional[Type[nn.Module]] = nn.ReLU,
                 final_activation: Optional[nn.Module] = None,
                 encoder_weights: Optional[str] = 'imagenet'):
        super().__init__()
        # Encoder setup
        self.encoder = smp.encoders.get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=len(decoder_filters) + 1,
            weights=encoder_weights
        )
        enc_channels = self.encoder.out_channels
        self.skip_channels = list(reversed(enc_channels[:-1]))[:len(decoder_filters)]
        self.deep_ch = enc_channels[-1]
        self.embed_dim = decoder_filters[0]
        # Cross-attention projections
        self.attn_q = nn.Linear(fc_in_channels, self.embed_dim)
        self.attn_k = nn.Conv2d(self.deep_ch, self.embed_dim, kernel_size=1)
        self.attn_v = nn.Conv2d(self.deep_ch, self.embed_dim, kernel_size=1)
        # Build decoders
        self.decoders = nn.ModuleList()
        for _ in range(out_channels):
            blocks = []
            prev_ch = self.embed_dim
            for i, f in enumerate(decoder_filters):
                in_ch = prev_ch + self.skip_channels[i]
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
        # Heads and final activation
        self.heads = nn.ModuleList([
            nn.Conv2d(decoder_filters[-1], 1, kernel_size=1)
            for _ in range(out_channels)
        ])
        self.final_activation = final_activation

    def _cross_attention(self, deep: torch.Tensor, x_fc: torch.Tensor) -> torch.Tensor:
        """
        Compute cross-attention feature map from deep features and BC vector.
        Returns a tensor of shape (B, embed_dim, H, W).
        """
        B, _, H, W = deep.shape
        # project queries, keys, values
        q = self.attn_q(x_fc).unsqueeze(1)                       # (B,1,embed_dim)
        k = self.attn_k(deep).view(B, self.embed_dim, -1).permute(0, 2, 1)  # (B,S,embed_dim)
        v = self.attn_v(deep).view(B, self.embed_dim, -1).permute(0, 2, 1)  # (B,S,embed_dim)
        # scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-1, -2)) / (self.embed_dim ** 0.5)  # (B,1,S)
        weights = torch.softmax(scores, dim=-1)                                  # (B,1,S)
        attn = torch.matmul(weights, v).squeeze(1)                               # (B,embed_dim)
        # reshape and broadcast spatially
        return attn.view(B, self.embed_dim, 1, 1).expand(-1, -1, H, W)

    def forward(self, x: torch.Tensor, x_fc: torch.Tensor) -> torch.Tensor:
        B, _, H0, W0 = x.shape
        # encode
        features = self.encoder(x)
        deep = features[-1]
        # cross-attention feature
        feat = self._cross_attention(deep, x_fc)
        # decode and collect outputs
        outs = []
        for dec, head in zip(self.decoders, self.heads):
            y = feat
            for idx, block in enumerate(dec):
                skip = features[-2-idx]
                y = F.interpolate(y, size=skip.shape[-2:], mode='nearest')
                y = torch.cat([skip, y], dim=1)
                y = block(y)
            y = head(y)
            y = F.interpolate(y, size=(H0, W0), mode='bilinear', align_corners=False)
            outs.append(y)
        out = torch.cat(outs, dim=1)
        return self.final_activation(out) if self.final_activation is not None else out

# Example:
# model = SMPCrossAttentionUNet('resnet34', in_channels=3, out_channels=4, fc_in_channels=6)
