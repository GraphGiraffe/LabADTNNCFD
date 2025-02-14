import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List, Optional, Type

from src.Models.AttentionUNet import (
    create_encoder,
    create_decoder,
    create_fc_block
)


###############################################
# Архитектура: MultiHeadAttentionUNet с несколькими декодерами и FC-блоками
###############################################


class MultiHeadAttentionUNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        filters: List[int] = [16, 32, 64, 128],
        enc_layers: int = 2,
        dec_layers: int = 2,
        weight_norm: bool = True,
        batch_norm: bool = True,
        activation: Optional[Type[nn.Module]] = nn.ReLU,
        final_activation: Optional[nn.Module] = None,
        add_fc_blocks: List[bool] = [True, True, True, True],
        fc_in_channels: int = 6,
        fc_filters: List[int] = [16, 32, 16],
        fc_out_channels: int = 8,
        dilation: int = 1,
        device: str = 'cpu'
    ):
        """
        UNet с механизмом внимания и несколькими декодирующими ветвями (по одной для каждого выходного поля),
        где каждая ветвь использует собственный набор FC-блоков для передачи информации о граничных условиях.

        Аргументы аналогичны AttentionUNet, за исключением того, что здесь создаются отдельные декодеры.
        """
        super(MultiHeadAttentionUNet, self).__init__()
        self.device = device
        self.out_channels = out_channels
        self.final_activation = final_activation

        # Общий кодировщик
        self.encoder = create_encoder(in_channels, filters, kernel_size, weight_norm, batch_norm, activation, enc_layers, dilation=dilation)

        # Для каждого выходного поля создаём отдельный декодер
        additional_fc_channels = [fc_out_channels if flag else 0 for flag in add_fc_blocks]
        self.decoders = nn.ModuleList([
            create_decoder(1, filters, additional_fc_channels, kernel_size, weight_norm, batch_norm, activation, dec_layers)
            for _ in range(out_channels)
        ])

        # И для каждого уровня декодера – набор FC-блоков (отдельно для каждой декодирующей ветви)
        self.fc_blocks_decoder = nn.ModuleList([
            nn.ModuleList([
                create_fc_block(fc_in_channels, fc_filters, fc_out_channels, torch.device(device), activation)
                if flag else None
                for _ in range(out_channels)
            ])
            for flag in add_fc_blocks
        ])

    def encode(self, x: torch.Tensor):
        """
        Кодировщик с сохранением skip-соединений и индексов.
        """
        tensors = []
        indices = []
        sizes = []
        for block in self.encoder:
            x = block(x)
            sizes.append(x.size())
            tensors.append(x)
            x, ind = F.max_pool2d(x, 2, 2, return_indices=True)
            indices.append(ind)
        return x, tensors, indices, sizes

    def decode(self, x: torch.Tensor, x_fc: torch.Tensor, tensors: List[torch.Tensor],
               indices: List[torch.Tensor], sizes: List[torch.Size]) -> torch.Tensor:
        """
        Для каждого выходного поля применяется свой декодер с использованием FC-блоков.
        """
        outputs = []
        # Для каждой декодирующей ветви
        for channel in range(self.out_channels):
            x_channel = x
            tensors_channel = tensors.copy()
            indices_channel = indices.copy()
            sizes_channel = sizes.copy()
            # Проходим по уровням декодера
            for depth in range(len(self.decoders[0])):
                tensor = tensors_channel.pop()
                size = sizes_channel.pop()
                ind = indices_channel.pop()
                x_channel = F.max_unpool2d(x_channel, ind, 2, 2, output_size=size)
                # Если для данного уровня FC-блок включен для этой ветви, применяем его
                fc_block = self.fc_blocks_decoder[depth][channel]
                if fc_block is not None:
                    out_fc = fc_block(x_fc)
                    ones = torch.ones(size[-2], size[-1]).to(torch.device(self.device))
                    out_fc = torch.einsum('ij,kl->ijkl', out_fc, ones)
                    x_channel = torch.cat([tensor, out_fc, x_channel], dim=1)
                else:
                    x_channel = torch.cat([tensor, x_channel], dim=1)
                x_channel = self.decoders[channel][depth](x_channel)
            outputs.append(x_channel)
        # Объединяем выходы всех ветвей по канальному измерению
        out = torch.cat(outputs, dim=1)
        if self.final_activation is not None:
            out = self.final_activation(out)
        return out

    def forward(self, x: torch.Tensor, x_fc: torch.Tensor) -> torch.Tensor:
        """
        Прямой проход через сеть: сначала кодировщик, затем для каждого поля своя ветвь декодера.
        """
        x, tensors, indices, sizes = self.encode(x)
        x = self.decode(x, x_fc, tensors, indices, sizes)
        return x
