import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from typing import List, Optional, Type

###############################################
# Вспомогательные функции для создания слоёв
###############################################


def create_layer(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    wn: bool,
    bn: bool,
    activation: Optional[Type[nn.Module]],
    conv_layer: Type[nn.Module] = nn.Conv2d,
    dilation: int = 1
) -> nn.Module:
    """
    Создает блок, включающий свёрточный слой (с возможной весовой нормализацией), 
    пакетную нормализацию и функцию активации.
    """
    padding = ((kernel_size - 1) * dilation) // 2
    layer = conv_layer(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
    if wn:
        layer = weight_norm(layer)
    modules = [layer]
    if bn:
        modules.append(nn.BatchNorm2d(out_channels))
    if activation is not None:
        modules.append(activation(inplace=True))
    return nn.Sequential(*modules)


def create_encoder_block(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    wn: bool = True,
    bn: bool = True,
    activation: Optional[Type[nn.Module]] = nn.ReLU,
    layers: int = 2,
    dilation: int = 1
) -> nn.Sequential:
    """
    Создает блок кодировщика из нескольких свёрточных слоёв, после которого добавляется SE-блок.
    """
    encoder_layers = []
    for i in range(layers):
        curr_in = in_channels if i == 0 else out_channels
        encoder_layers.append(create_layer(curr_in, out_channels, kernel_size, wn, bn, activation, nn.Conv2d, dilation=dilation))
    # Добавляем SE-блок для внимания (Squeeze-and-Excitation)
    encoder_layers.append(SEBlock(out_channels))
    return nn.Sequential(*encoder_layers)


def create_encoder(
    in_channels: int,
    filters: List[int],
    kernel_size: int,
    wn: bool = True,
    bn: bool = True,
    activation: Optional[Type[nn.Module]] = nn.ReLU,
    layers: int = 2,
    dilation: int = 1
) -> nn.Module:
    """
    Создает последовательность блоков кодировщика.
    """
    modules = []
    curr_in = in_channels
    for filt in filters:
        modules.append(create_encoder_block(curr_in, filt, kernel_size, wn, bn, activation, layers, dilation=dilation))
        curr_in = filt
    return nn.Sequential(*modules)


def create_decoder_block(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    wn: bool = True,
    bn: bool = True,
    activation: Optional[Type[nn.Module]] = nn.ReLU,
    layers: int = 2,
    final_layer: bool = False,
    additional_fc_channels: int = 0
) -> nn.Sequential:
    """
    Создает блок декодера из нескольких транспонированных свёрточных слоёв.
    """
    decoder_layers = []
    for i in range(layers):
        # Первый слой принимает объединённые данные (skip-связь и, если есть, FC-блок)
        curr_in = in_channels * 2 + additional_fc_channels if i == 0 else in_channels
        curr_out = out_channels if (i == layers - 1) else in_channels
        curr_bn = bn if not (i == layers - 1 and final_layer) else False
        curr_act = activation if not (i == layers - 1 and final_layer) else None
        decoder_layers.append(create_layer(curr_in, curr_out, kernel_size, wn, curr_bn, curr_act, nn.ConvTranspose2d))
    return nn.Sequential(*decoder_layers)


def create_decoder(
    out_channels: int,
    filters: List[int],
    additional_fc_channels: List[int],
    kernel_size: int,
    wn: bool = True,
    bn: bool = True,
    activation: Optional[Type[nn.Module]] = nn.ReLU,
    layers: int = 2
) -> nn.Sequential:
    """
    Создает декодер, состоящий из нескольких блоков декодера.
    """
    # Формируем блоки декодера в обратном порядке (от глубокого представления к исходному разрешению)
    decoder_blocks = []
    for i in range(len(filters)):
        if i == 0:
            block = create_decoder_block(filters[i], out_channels, kernel_size, wn, bn, activation, layers,
                                         final_layer=True, additional_fc_channels=additional_fc_channels[i])
        else:
            block = create_decoder_block(filters[i], filters[i-1], kernel_size, wn, bn, activation, layers,
                                         final_layer=False, additional_fc_channels=additional_fc_channels[i])
        # Вставляем в начало списка, чтобы порядок соответствовал этапам unpooling
        decoder_blocks.insert(0, block)
    return nn.Sequential(*decoder_blocks)

###############################################
# Модуль SEBlock для внимания
###############################################


class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        """
        Squeeze-and-Excitation блок для переоценки каналов признаков.

        Аргументы:
            channels (int): Число входных каналов.
            reduction (int): Коэффициент редукции, по умолчанию 16.
        """
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

###############################################
# FC-блок для передачи информации о граничных условиях
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
        fc_filters (List[int]): Список размеров скрытых слоёв. Не должен быть пустым.
        fc_out_channels (int): Число выходных нейронов.
        device (torch.device): Устройство, на которое будет перенесён блок.
        activation (Optional[Type[nn.Module]]): Функция активации, по умолчанию nn.ReLU.
        use_activation (bool): Если True – добавлять активацию после каждого слоя (кроме последнего).

    Возвращает:
        nn.Sequential: Последовательность линейных слоёв с активацией, перенесённая на заданное устройство.
    """
    if not fc_filters:
        raise ValueError("Список fc_filters не должен быть пустым.")
    layers = []
    # Первый линейный слой
    layers.append(nn.Linear(fc_in_channels, fc_filters[0]))
    if use_activation and activation is not None:
        layers.append(activation(inplace=True))
    # Скрытые слои
    for i in range(len(fc_filters) - 1):
        layers.append(nn.Linear(fc_filters[i], fc_filters[i+1]))
        if use_activation and activation is not None:
            layers.append(activation(inplace=True))
    # Финальный слой без активации
    layers.append(nn.Linear(fc_filters[-1], fc_out_channels))
    return nn.Sequential(*layers).to(device)

###############################################
# Архитектура: AttentionUNet с одним декодером и FC-блоками
###############################################


class AttentionUNet(nn.Module):
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
        UNet с механизмом внимания (SE-блоками) и передачей информации о граничных условиях через FC-блоки.
        Используется один общий декодер.

        Аргументы:
            in_channels (int): Число входных каналов.
            out_channels (int): Число выходных каналов (например, 4 для U, V, P, T).
            kernel_size (int): Размер свёрточного ядра.
            filters (List[int]): Список фильтров для кодировщика.
            enc_layers (int): Количество свёрточных слоёв в каждом блоке кодировщика.
            dec_layers (int): Количество слоёв в каждом блоке декодера.
            weight_norm, batch_norm, activation: Параметры для создания сверточных слоёв.
            final_activation: Финальная активация модели.
            add_fc_blocks (List[bool]): Флаги для включения FC-блока на каждом уровне декодера.
            fc_in_channels, fc_filters, fc_out_channels: Параметры для FC-блоков.
            dilation (int): Коэффициент дилатации.
            device (str): Устройство для вычислений.
        """
        super(AttentionUNet, self).__init__()
        self.device = device
        self.out_channels = out_channels
        self.final_activation = final_activation

        # Кодировщик
        self.encoder = create_encoder(in_channels, filters, kernel_size, weight_norm, batch_norm, activation, enc_layers, dilation=dilation)

        # Для хранения skip-соединений будем сохранять выходы каждого блока кодировщика
        # Декодер будет состоять из нескольких уровней, порядок которых соответствует количеству блоков в encoder
        additional_fc_channels = [fc_out_channels if flag else 0 for flag in add_fc_blocks]
        self.decoder = create_decoder(out_channels, filters, additional_fc_channels, kernel_size, weight_norm, batch_norm, activation, dec_layers)

        # FC-блоки для передачи информации о граничных условиях
        # Создаем по одному FC-блоку для каждого уровня декодера, если соответствующий флаг включен
        self.fc_blocks_decoder = nn.ModuleList([
            create_fc_block(fc_in_channels, fc_filters, fc_out_channels, torch.device(device), activation)
            if flag else None
            for flag in add_fc_blocks
        ])

    def encode(self, x: torch.Tensor):
        """
        Проходит вход через кодировщик, сохраняя skip-соединения и индексы максимального пулинга.
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
        Декодирование с использованием skip-соединений и добавлением информации через FC-блоки.
        """
        # Обратный порядок для skip-соединений
        for i in range(len(self.decoder)):
            # Извлекаем сохраненные значения для текущего уровня
            tensor = tensors.pop()
            size = sizes.pop()
            ind = indices.pop()
            # Применяем unpooling
            x = F.max_unpool2d(x, ind, 2, 2, output_size=size)
            # Если FC-блок включён на данном уровне, применяем его к x_fc
            fc_block = self.fc_blocks_decoder[i]
            if fc_block is not None:
                out_fc = fc_block(x_fc)  # Выход имеет форму (B, fc_out_channels)
                # Расширяем до размера (B, fc_out_channels, H, W)
                ones = torch.ones(size[-2], size[-1]).to(torch.device(self.device))
                out_fc = torch.einsum('ij,kl->ijkl', out_fc, ones)
                # Конкатенируем с skip-соединением и текущим тензором
                x = torch.cat([tensor, out_fc, x], dim=1)
            else:
                x = torch.cat([tensor, x], dim=1)
            # Пропускаем через соответствующий блок декодера
            x = self.decoder[i](x)
        return x

    def forward(self, x: torch.Tensor, x_fc: torch.Tensor) -> torch.Tensor:
        """
        Прямой проход: сначала кодировщик, затем декодировщик с учетом x_fc.
        """
        x, tensors, indices, sizes = self.encode(x)
        x = self.decode(x, x_fc, tensors, indices, sizes)
        if self.final_activation is not None:
            x = self.final_activation(x)
        return x


###############################################
# Пример создания сети (аналог функции exp)
###############################################

def create_network(model_type: str = 'single'):
    """
    Создает экземпляр модели AttentionUNet с общим декодером
    Параметры модели задаются через конфигурационный словарь.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config = {
        'in_channels': 3,
        'out_channels': 4,  # поля: U, V, P, T
        'kernel_size': 3,
        'filters': [16, 32, 64, 128],
        'enc_layers': 2,
        'dec_layers': 2,
        'weight_norm': True,
        'batch_norm': True,
        'activation': torch.nn.ReLU,
        'final_activation': None,
        'add_fc_blocks': [True, True, True, True],
        'fc_in_channels': 6,
        'fc_filters': [16, 32, 16],
        'fc_out_channels': 8,
        'dilation': 1,
        'device': device
    }
    model = AttentionUNet(**config)
    model = model.to(device)
    print(model)
    return model


if __name__ == '__main__':
    net = create_network()
    # Пример: случайный входной батч
    dummy_input = torch.randn(1, 3, 256, 256).to(net.device)
    dummy_x_fc = torch.randn(1, 6).to(net.device)
    output = net(dummy_input, dummy_x_fc)
    print("Output shape:", output.shape)
