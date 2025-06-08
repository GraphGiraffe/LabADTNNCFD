import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Type

############################################################
# Helper blocks
############################################################

def conv_block(in_channels: int, out_channels: int, kernel_size: int = 3,
               activation: Type[nn.Module] = nn.ReLU, bn: bool = True) -> nn.Sequential:
    """A simple Conv→Act→BN block (padding keeps spatial size)."""
    assert kernel_size % 2 == 1, "kernel_size must be odd"
    layers: List[nn.Module] = [
        nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2),
        activation(inplace=True)
    ]
    if bn:
        layers.append(nn.BatchNorm2d(out_channels))
    return nn.Sequential(*layers)


def double_conv(in_channels: int, out_channels: int, mid_channels: Optional[int] = None,
                activation: Type[nn.Module] = nn.ReLU) -> nn.Sequential:
    """(U‑Net style) two 3×3 convolutions with activation & BN."""
    if mid_channels is None:
        mid_channels = out_channels
    return nn.Sequential(
        conv_block(in_channels, mid_channels, activation=activation),
        conv_block(mid_channels, out_channels, activation=activation)
    )

############################################################
# Main architecture
############################################################


class UNetTransformerBC(nn.Module):
    """UNet backbone with a Transformer bottleneck conditioned on BC vector.

    The spatial feature map at the bottleneck is flattened and passed through a
    Transformer encoder.  A learnable linear projection embeds the BC vector and
    is *prepended* as an extra token, allowing the Transformer to mix global BC
    information with spatial context.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        fc_in_channels: int = 6,
        encoder_filters: List[int] = [32, 64, 128, 256],
        decoder_filters: List[int] = [256, 128, 64, 32],
        num_transformer_layers: int = 4,
        num_heads: int = 8,
        embed_dim: int = 256,
        kernel_size: int = 3,
        activation: Type[nn.Module] = nn.ReLU,
        final_activation: Optional[nn.Module] = None,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        assert len(encoder_filters) == len(decoder_filters), "Encoder/decoder depth mismatch"
        assert encoder_filters[-1] == decoder_filters[0] == embed_dim, "embed_dim must match deepest filter size"

        self.device = torch.device(device)
        self.final_activation = final_activation

        ###########################
        # Encoder
        ###########################
        enc_blocks: List[nn.Module] = []
        in_ch = in_channels
        for f in encoder_filters:
            enc_blocks.append(double_conv(in_ch, f, activation=activation))
            in_ch = f
        self.encoder = nn.ModuleList(enc_blocks)

        ###########################
        # Bottleneck Transformer
        ###########################
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=False)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)
        self.bc_proj = nn.Linear(fc_in_channels, embed_dim)

        ###########################
        # Decoder (upsample + conv)
        ###########################
        dec_blocks: List[nn.Module] = []
        upsamplers: List[nn.Module] = []
        for idx, f in enumerate(decoder_filters):
            # nn.ConvTranspose2d could be used, but bilinear upsample avoids checkerboard artefacts.
            upsamplers.append(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False))
            in_ch = f + encoder_filters[-1 - idx]  # concat skip connection
            out_ch = decoder_filters[idx + 1] if idx + 1 < len(decoder_filters) else decoder_filters[-1]
            dec_blocks.append(double_conv(in_ch, out_ch, activation=activation))
        self.upsamplers = nn.ModuleList(upsamplers)
        self.decoder = nn.ModuleList(dec_blocks)

        ###########################
        # Output head(s)
        ###########################
        self.head = nn.Conv2d(decoder_filters[-1], out_channels, kernel_size=1)

        ###########################
        # Init & device
        ###########################
        self.to(self.device)

    # ---------------------------
    # forward helpers
    # ---------------------------
    def _encode(self, x: torch.Tensor):
        """Encode input, returning deepest feature map + list of skip tensors."""
        skips = []
        h = x
        for enc in self.encoder:
            h = enc(h)
            skips.append(h)
            h = F.max_pool2d(h, 2)  # ↓H/2, ↓W/2
        return h, skips

    def _transform(self, feats: torch.Tensor, bc_vec: torch.Tensor):
        """Apply Transformer on flattened feats, conditioned on bc_vec."""
        B, C, H, W = feats.shape
        seq = feats.flatten(2).permute(2, 0, 1)  # (S, B, C)
        bc_token = self.bc_proj(bc_vec).unsqueeze(0)  # (1, B, C)
        seq_in = torch.cat([bc_token, seq], dim=0)
        seq_out = self.transformer(seq_in)
        bc_out, spat_out = seq_out[0], seq_out[1:]
        spat_out = spat_out.permute(1, 2, 0).view(B, C, H, W)
        # broadcast bc_out over spatial dims and add (FiLM‑like bias)
        spat_out = spat_out + bc_out.unsqueeze(-1).unsqueeze(-1)
        return spat_out

    def _decode(self, feats: torch.Tensor, skips: List[torch.Tensor]):
        h = feats
        for idx in range(len(self.decoder)):
            h = self.upsamplers[idx](h)
            skip = skips[-1 - idx]  # reverse order
            # ensure spatial alignment due to potential rounding in pooling
            if h.shape[-2:] != skip.shape[-2:]:
                h = F.interpolate(h, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            h = torch.cat([h, skip], dim=1)
            h = self.decoder[idx](h)
        return h

    # ---------------------------
    # forward
    # ---------------------------
    def forward(self, x: torch.Tensor, bc_vec: torch.Tensor) -> torch.Tensor:
        """x: [B, in_channels, H, W]; bc_vec: [B, fc_in_channels]."""
        deep, skips = self._encode(x)
        deep = self._transform(deep, bc_vec)
        out = self._decode(deep, skips)
        out = self.head(out)
        if self.final_activation is not None:
            out = self.final_activation(out)
        return out

############################################################
# Convenience factory for registry use (mirrors other models)
############################################################

def UNetTransformerBC_factory(**kwargs):
    """Keeps identical signature expected by train/test pipeline (model name lookup)."""
    return UNetTransformerBC(**kwargs)
