import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm


def create_layer(in_channels, out_channels, kernel_size, wn=True, bn=True,
                 activation=nn.ReLU, convolution=nn.Conv2d):
    assert kernel_size % 2 == 1
    layer = []
    conv = convolution(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
    if wn:
        conv = weight_norm(conv)
    layer.append(conv)
    if activation is not None:
        layer.append(activation())
    if bn:
        layer.append(nn.BatchNorm2d(out_channels))
    return nn.Sequential(*layer)


def create_encoder_block(in_channels, out_channels, kernel_size, wn=True, bn=True,
                 activation=nn.ReLU, layers=2):
    encoder = []
    for i in range(layers):
        _in = out_channels
        _out = out_channels
        if i == 0:
            _in = in_channels
        encoder.append(create_layer(_in, _out, kernel_size, wn, bn, activation, nn.Conv2d))
    return nn.Sequential(*encoder)


def create_decoder_block(in_channels, out_channels, kernel_size, wn=True, bn=True,
                 activation=nn.ReLU, layers=2, final_layer=False, additional_fc_channels=0):
    decoder = []
    for i in range(layers):
        _in = in_channels
        _out = in_channels
        _bn = bn
        _activation = activation
        if i == 0:
            _in = in_channels * 2 + additional_fc_channels
        if i == layers - 1:
            _out = out_channels
            if final_layer:
                _bn = False
                _activation = None
        decoder.append(create_layer(_in, _out, kernel_size, wn, _bn, _activation, nn.ConvTranspose2d))
    return nn.Sequential(*decoder)


def create_encoder(in_channels, filters, kernel_size, wn=True, bn=True, activation=nn.ReLU, layers=2):
    encoder = []
    for i in range(len(filters)):
        if i == 0:
            encoder_layer = create_encoder_block(in_channels, filters[i], kernel_size, wn, bn, activation, layers)
        else:
            encoder_layer = create_encoder_block(filters[i-1], filters[i], kernel_size, wn, bn, activation, layers)
        encoder = encoder + [encoder_layer]
    return nn.Sequential(*encoder)


def create_decoder(out_channels, filters, additional_fc_channels, kernel_size, wn=True, bn=True, activation=nn.ReLU, layers=2,
                   ):
    decoder = []
    for i in range(len(filters)):
        if i == 0:
            decoder_layer = create_decoder_block(filters[i], out_channels, kernel_size, wn, bn, activation, layers, final_layer=True,
                                                 additional_fc_channels=additional_fc_channels[i])
        else:
            decoder_layer = create_decoder_block(filters[i], filters[i-1], kernel_size, wn, bn, activation, layers, final_layer=False,
                                                 additional_fc_channels=additional_fc_channels[i])
        decoder = [decoder_layer] + decoder
    return nn.Sequential(*decoder)


class UNetExFC(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, filters=[16, 32, 64], layers=3,
                 weight_norm=True, batch_norm=True, activation=nn.ReLU, final_activation=None,
                 add_fc_blocks=[True, False, True],
                 fc_in_channels=5, fc_out_channels=2, fc_filters=[8, 32, 8], device='cpu'):
        super().__init__()
        assert len(filters) > 0

        self.device = device
        self.out_channels = out_channels
        self.final_activation = final_activation
        self.encoder = create_encoder(in_channels, filters, kernel_size, weight_norm, batch_norm, activation, layers)
        decoders = []

        self.add_fc_blocks = add_fc_blocks
        self.fc_blocks_decoder = nn.ModuleList()
        self.fc_in_channels = fc_in_channels
        self.fc_out_channels = fc_out_channels
        self.fc_filters = fc_filters

        for add_fc in add_fc_blocks:
            fc_block_by_out_channel = nn.ModuleList()
            for _ in range(self.out_channels):
                if add_fc:
                    fc_layer = []
                    fc_layer.append(nn.Linear(self.fc_in_channels, self.fc_filters[0]).to(device))
                    for idx in range(len(self.fc_filters) - 1):
                        fc_layer.append(nn.Linear(self.fc_filters[idx], self.fc_filters[idx + 1]).to(device))
                    fc_layer.append(nn.Linear(self.fc_filters[-1], self.fc_out_channels).to(device))
                    fc_layer = nn.Sequential(*fc_layer)
                else:
                    fc_layer = None
                fc_block_by_out_channel.append(fc_layer)
            self.fc_blocks_decoder.append(fc_block_by_out_channel)

        additional_fc_channels = [int(v)*fc_out_channels for v in add_fc_blocks]
        additional_fc_channels.reverse()
        
        for i in range(self.out_channels):
            decoders.append(create_decoder(1, filters, additional_fc_channels, kernel_size, weight_norm, batch_norm,
                                           activation, layers)
                            )
        self.decoders = nn.Sequential(*decoders)

    def encode(self, x):
        tensors = []
        indices = []
        sizes = []
        for encoder in self.encoder:
            x = encoder(x)
            sizes.append(x.size())
            tensors.append(x)
            x, ind = F.max_pool2d(x, 2, 2, return_indices=True)
            indices.append(ind)
        return x, tensors, indices, sizes

    def decode(self, _x, _x_fc, _tensors, _indices, _sizes):
        y = []
        for channel_n, _decoder in enumerate(self.decoders):
            x = _x
            tensors = _tensors[:]
            indices = _indices[:]
            sizes = _sizes[:]
            for depth_n, decoder in enumerate(_decoder):
                tensor = tensors.pop()
                size = sizes.pop()
                ind = indices.pop()
                x = F.max_unpool2d(x, ind, 2, 2, output_size=size)

                if self.fc_blocks_decoder[depth_n][channel_n] is not None:
                    x_fc = _x_fc
                    out_fc = self.fc_blocks_decoder[depth_n][channel_n](x_fc)
                    ones = torch.ones(size[-2], size[-1]).to(self.device)
                    out_fc = torch.einsum('ij,kl->ijkl', out_fc, ones)
                    x = torch.cat([tensor, out_fc, x], dim=1)
                else:
                    x = torch.cat([tensor, x], dim=1)

                x = decoder(x)

            y.append(x)
        return torch.cat(y, dim=1)

    def forward(self, x, x_fc):
        x, tensors, indices, sizes = self.encode(x)
        x = self.decode(x, x_fc, tensors, indices, sizes)
        if self.final_activation is not None:
            x = self.final_activation(x)
        return x
