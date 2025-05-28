import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp


class SmpUNetExFC(nn.Module):
    def __init__(
        self,
        encoder_name: str,
        in_channels: int,
        out_channels: int,
        filters: list,
        fc_in_channels: int,
        fc_filters: list,
        fc_out_channels: int,
        add_fc_blocks: list,
        batch_norm=False,
        decoder_attention_type=None,
        pretrained=True,
        device='cpu'
    ):
        super().__init__()
        self.device = device
        self.out_channels = out_channels
        self.batch_norm = batch_norm

        # Encoder
        self.encoder = smp.encoders.get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=len(filters),
            weights='imagenet' if pretrained else None
        )
        encoder_channels = self.encoder.out_channels  # list of channels at each stage

        # FC blocks for boundary conditions
        self.add_fc_blocks = add_fc_blocks
        self.fc_blocks = nn.ModuleList()
        for add in add_fc_blocks:
            blocks = nn.ModuleList()
            for _ in range(out_channels):
                if add:
                    layers_list = []
                    in_ch = fc_in_channels
                    for f in fc_filters:
                        layers_list.append(nn.Linear(in_ch, f))
                        layers_list.append(nn.ReLU(inplace=True))
                        in_ch = f
                    layers_list.append(nn.Linear(in_ch, fc_out_channels))
                    blocks.append(nn.Sequential(*layers_list).to(device))
                else:
                    blocks.append(None)
            self.fc_blocks.append(blocks)
        # compute extra channels per decoder level
        extra_channels = [int(v) * fc_out_channels for v in add_fc_blocks]
        extra_channels = list(reversed(extra_channels))

        # Decoders: one per output channel
        self.decoders = nn.ModuleList()
        for _ in range(out_channels):
            decoder = smp.unet.decoder.UnetDecoder(
                encoder_channels=encoder_channels,
                decoder_channels=filters[::-1],
                n_blocks=len(filters),
                use_batchnorm=self.batch_norm,
                attention_type=decoder_attention_type,
                center=True
            )
            self.decoders.append(decoder)

        # Final conv for each decoder
        self.heads = nn.ModuleList([
            nn.Conv2d(filters[0], 1, kernel_size=1)
            for _ in range(out_channels)
        ])

        self.to(device)

    def forward(self, x: torch.Tensor, x_fc: torch.Tensor) -> torch.Tensor:
        # Encode
        features = self.encoder(x)
        # features: list of feature maps per stage
        feats = features[1:]  # skip input stage if necessary

        outs = []
        for ch_idx, decoder in enumerate(self.decoders):
            d = feats[-1]
            # Unpool through decoder blocks
            for level_idx, block in enumerate(decoder.blocks):  # UnetDecoder stores blocks in .blocks
                # apply FC if at this level
                if self.add_fc_blocks[level_idx]:
                    fc_out = self.fc_blocks[level_idx][ch_idx](x_fc)
                    # expand spatial dims
                    b, c = fc_out.shape
                    h, w = d.shape[-2], d.shape[-1]
                    ones = torch.ones(h, w, device=self.device)
                    fc_map = torch.einsum('bc,hw->bchw', fc_out, ones)
                    d = torch.cat([d, fc_map], dim=1)
                # decode
                d = block(d, features[-(level_idx+2)])  # skip connections
            # final conv
            out = self.heads[ch_idx](d)
            outs.append(out)
        x_out = torch.cat(outs, dim=1)

        return x_out
