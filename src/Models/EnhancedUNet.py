import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from typing import List, Optional, Type

###############################################
# Функции для создания блоков кодировщика и декодера
###############################################


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


def create_encoder_block(in_channels: int, out_channels: int, kernel_size: int,
                         wn: bool = True, bn: bool = True,
                         activation: Type[nn.Module] = nn.ReLU, layers: int = 2) -> nn.Sequential:
    """
    Создает блок кодировщика из нескольких сверточных слоёв.
    """
    encoder = []
    for i in range(layers):
        _in = out_channels if i > 0 else in_channels
        _out = out_channels
        encoder.append(create_layer(_in, _out, kernel_size, wn, bn, activation, nn.Conv2d))
    return nn.Sequential(*encoder)


def create_decoder_block(in_channels: int, out_channels: int, kernel_size: int,
                         wn: bool = True, bn: bool = True,
                         activation: Type[nn.Module] = nn.ReLU, layers: int = 2,
                         final_layer: bool = False, additional_fc_channels: int = 0) -> nn.Sequential:
    """
    Создает блок декодера из нескольких транспонированных сверточных слоёв.
    """
    decoder = []
    for i in range(layers):
        _in = in_channels * 2 + additional_fc_channels if i == 0 else in_channels
        _out = out_channels if (i == layers - 1) else in_channels
        _bn = bn if not (i == layers - 1 and final_layer) else False
        _activation = activation if not (i == layers - 1 and final_layer) else None
        decoder.append(create_layer(_in, _out, kernel_size, wn, _bn, _activation, nn.ConvTranspose2d))
    return nn.Sequential(*decoder)


def create_encoder(in_channels: int, filters: List[int], kernel_size: int,
                   wn: bool = True, bn: bool = True,
                   activation: Type[nn.Module] = nn.ReLU, layers: int = 2) -> nn.Sequential:
    """
    Создает последовательность блоков кодировщика.
    """
    encoder = []
    for i in range(len(filters)):
        if i == 0:
            encoder_layer = create_encoder_block(in_channels, filters[i], kernel_size, wn, bn, activation, layers)
        else:
            encoder_layer = create_encoder_block(filters[i-1], filters[i], kernel_size, wn, bn, activation, layers)
        encoder.append(encoder_layer)
    return nn.Sequential(*encoder)


def create_decoder(out_channels: int, filters: List[int], additional_fc_channels: List[int],
                   kernel_size: int, wn: bool = True, bn: bool = True,
                   activation: Type[nn.Module] = nn.ReLU, layers: int = 2) -> nn.Sequential:
    """
    Создает декодер из последовательных блоков декодера.
    """
    decoder = []
    for i in range(len(filters)):
        if i == 0:
            decoder_layer = create_decoder_block(filters[i], out_channels, kernel_size, wn, bn, activation, layers,
                                                 final_layer=True, additional_fc_channels=additional_fc_channels[i])
        else:
            decoder_layer = create_decoder_block(filters[i], filters[i-1], kernel_size, wn, bn, activation, layers,
                                                 final_layer=False, additional_fc_channels=additional_fc_channels[i])
        decoder.insert(0, decoder_layer)
    return nn.Sequential(*decoder)

###############################################
# Улучшенная функция создания FC-блока
###############################################


def create_fc_block(
    fc_in_channels: int,
    fc_filters: List[int],
    fc_out_channels: int,
    device: torch.device,
    activation: Optional[Type[nn.Module]] = nn.ReLU,
    use_activation: bool = True
) -> nn.Sequential:
    """
    Создает последовательный блок полностью-связанных (FC) слоев для декодера.

    Аргументы:
        fc_in_channels (int): Число входных нейронов для первого слоя.
        fc_filters (List[int]): Список размеров скрытых слоев. Не должен быть пустым.
        fc_out_channels (int): Число выходных нейронов (размер выхода блока).
        device (torch.device): Устройство, на которое будет перенесён блок.
        activation (Optional[Type[nn.Module]]): Класс функции активации, применяемой после каждого линейного слоя (по умолчанию nn.ReLU).
        use_activation (bool): Флаг, определяющий, следует ли добавлять активацию после каждого линейного слоя, кроме последнего.

    Возвращает:
        nn.Sequential: Последовательность линейных слоев (с опциональной активацией), перенесённая на указанное устройство.
    """
    if not fc_filters:
        raise ValueError("Список fc_filters не должен быть пустым.")

    layers: List[nn.Module] = []
    # Первый линейный слой
    layers.append(nn.Linear(fc_in_channels, fc_filters[0]))
    if use_activation and activation is not None:
        layers.append(activation(inplace=True))

    # Скрытые линейные слои с активацией
    for i in range(len(fc_filters) - 1):
        layers.append(nn.Linear(fc_filters[i], fc_filters[i+1]))
        if use_activation and activation is not None:
            layers.append(activation(inplace=True))

    # Финальный линейный слой без активации
    layers.append(nn.Linear(fc_filters[-1], fc_out_channels))

    return nn.Sequential(*layers).to(device)

###############################################
# Улучшенная архитектура: EnhancedUNet
###############################################


class EnhancedUNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        filters: List[int] = [16, 32, 64, 128, 256, 256, 128, 64, 32],
        layers: int = 3,
        weight_norm: bool = True,
        batch_norm: bool = True,
        activation: Optional[Type[nn.Module]] = nn.ReLU,
        final_activation: Optional[nn.Module] = None,
        add_fc_blocks: List[bool] = [True, False, True],
        fc_in_channels: int = 6,
        fc_out_channels: int = 8,
        fc_filters: List[int] = [16, 32, 16],
        device: str = 'cpu'
    ):
        super().__init__()
        assert len(filters) > 0, "Список filters не должен быть пустым."
        self.device = device
        self.out_channels = out_channels
        self.final_activation = final_activation

        # Создание кодировщика
        self.encoder = create_encoder(in_channels, filters, kernel_size, weight_norm, batch_norm, activation, layers)

        # Сохранение параметров для создания FC-блоков
        self.fc_in_channels = fc_in_channels
        self.fc_filters = fc_filters
        self.fc_out_channels = fc_out_channels

        # Создание FC-блоков для декодера с использованием функции create_fc_block
        self.fc_blocks_decoder = nn.ModuleList([
            nn.ModuleList([
                create_fc_block(self.fc_in_channels, self.fc_filters, self.fc_out_channels,
                                torch.device(device), activation)
                if add_fc else None
                for _ in range(out_channels)
            ])
            for add_fc in add_fc_blocks
        ])

        # Вычисление дополнительных каналов от FC-блоков для объединения в декодере
        additional_fc_channels = [int(v) * fc_out_channels for v in add_fc_blocks]
        additional_fc_channels.reverse()

        # Создание декодеров для каждого выходного канала
        decoders = []
        for i in range(out_channels):
            decoders.append(create_decoder(1, filters, additional_fc_channels, kernel_size, weight_norm, batch_norm, activation, layers))
        self.decoders = nn.Sequential(*decoders)

    def encode(self, x: torch.Tensor):
        """
        Проходит входное изображение через кодировщик.
        Возвращает выход последнего слоя, список промежуточных тензоров, индексы и размеры.
        """
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

    def decode(self, _x: torch.Tensor, _x_fc: torch.Tensor, _tensors: List[torch.Tensor],
               _indices: List[torch.Tensor], _sizes: List[torch.Size]) -> torch.Tensor:
        """
        Производит декодирование, объединяя информацию из кодировщика и выходы FC-блоков.
        """
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
                    # Применяем соответствующий FC-блок
                    out_fc = self.fc_blocks_decoder[depth_n][channel_n](_x_fc)
                    ones = torch.ones(size[-2], size[-1]).to(torch.device(self.device))
                    # Расширяем размерность выходов FC-блока до размеров тензора
                    out_fc = torch.einsum('ij,kl->ijkl', out_fc, ones)
                    x = torch.cat([tensor, out_fc, x], dim=1)
                else:
                    x = torch.cat([tensor, x], dim=1)
                x = decoder(x)
            y.append(x)
        return torch.cat(y, dim=1)

    def forward(self, x: torch.Tensor, x_fc: torch.Tensor) -> torch.Tensor:
        """
        Прямой проход через модель.
        """
        x, tensors, indices, sizes = self.encode(x)
        x = self.decode(x, x_fc, tensors, indices, sizes)
        if self.final_activation is not None:
            x = self.final_activation(x)
        return x
