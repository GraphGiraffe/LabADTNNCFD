import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """
    Две последовательные свёртки (3×3, padding=1) с ReLU,
    сохраняющие пространственный размер.
    """

    def __init__(self, in_ch: int, out_ch: int):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class BCModule(nn.Module):
    """
    МЛП для учёта граничных условий.
    На вход принимает вектор size=(B, fc_in_channels) → через скрытые слои →
    выдаёт embedding длины bottleneck_channels, который расширяется до (B, Cb, Hb, Wb)
    и добавляется к фичам «бутылочного горлышка».
    """

    def __init__(self, fc_in_channels: int, bc_hidden_dims: list[int], bottleneck_channels: int):
        super(BCModule, self).__init__()
        layers: list[nn.Module] = []
        prev = fc_in_channels
        for hid in (bc_hidden_dims or []):
            layers.append(nn.Linear(prev, hid))
            layers.append(nn.ReLU(inplace=True))
            prev = hid
        layers.append(nn.Linear(prev, bottleneck_channels))
        self.mlp = nn.Sequential(*layers)

    def forward(self, bc: torch.Tensor, spatial_shape: tuple[int, int]) -> torch.Tensor:
        # bc: (B, fc_in_channels)
        emb = self.mlp(bc)  # (B, Cb)
        B, Cb = emb.shape
        H, W = spatial_shape
        emb = emb.view(B, Cb, 1, 1).expand(-1, -1, H, W)
        return emb  # (B, Cb, H, W)


class DecoderBlock(nn.Module):
    """
    Новый DecoderBlock, где вместо ConvTranspose2d мы делаем интерполяцию
    до точного размера скипа, затем 1×1 свёртку для уменьшения каналов,
    а после — DoubleConv на конкатенации (skip + upsampled).
    """

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super(DecoderBlock, self).__init__()
        # 1×1 для снижения числа каналов после интерполяции
        self.conv_reduce = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=True)
        # После конкатенации: каналы = out_ch + skip_ch
        self.conv = DoubleConv(out_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        # Интерполируем x до тех же H×W, что у skip
        target_size = skip.shape[2], skip.shape[3]
        x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
        x = self.conv_reduce(x)              # (B, out_ch, H_skip, W_skip)
        x = torch.cat([skip, x], dim=1)      # (B, skip_ch + out_ch, H_skip, W_skip)
        x = self.conv(x)                     # (B, out_ch, H_skip, W_skip)
        return x


class UNetMultiDecoderBC(nn.Module):
    """
    U-Net с одним энкодером, MLP для BC и N=out_channels отдельных декодеров.
    Каждый декодер точно восстанавливает исходный размер (без обрезок).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        fc_in_channels: int,
        encoder_channels: list[int],
        decoder_channels: list[int],
        bc_hidden_dims: list[int] | None = None,
        device='cpu',
    ):
        super(UNetMultiDecoderBC, self).__init__()
        self.device = device
        assert len(decoder_channels) == len(encoder_channels) - 1, (
            f"len(decoder_channels) должно быть {len(encoder_channels)-1}, "
            f"получено {len(decoder_channels)}"
        )

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.encoder_channels = encoder_channels
        self.decoder_channels = decoder_channels
        bottleneck_channels = encoder_channels[-1]

        # --- Энкодер: цепочка DoubleConv + Pool (кроме последнего) ---
        enc_blocks = []
        prev_ch = in_channels
        for ch in encoder_channels:
            enc_blocks.append(DoubleConv(prev_ch, ch))
            prev_ch = ch
        self.enc_blocks = nn.ModuleList(enc_blocks)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # --- Модуль BC для бутылочного горлышка ---
        self.bc_module = BCModule(
            fc_in_channels=fc_in_channels,
            bc_hidden_dims=bc_hidden_dims or [],
            bottleneck_channels=bottleneck_channels,
        )

        # --- Многократный декодер: создаём ветку для каждого выходного канала ---
        self.decoders = nn.ModuleList()
        self.final_convs = nn.ModuleList()
        skip_channels = encoder_channels[:-1]  # каналы фич skip: [enc0, enc1, ..., enc_{L-2}]
        # Будем их брать в обратном порядке при декодировании.

        for _ in range(out_channels):
            branch_blocks = nn.ModuleList()
            in_ch_dec = bottleneck_channels
            for i, out_ch in enumerate(decoder_channels):
                skip_ch = skip_channels[::-1][i]
                branch_blocks.append(DecoderBlock(in_ch_dec, skip_ch, out_ch))
                in_ch_dec = out_ch
            self.decoders.append(branch_blocks)
            # 1×1 для получения ровно одного канала на выход ветки
            self.final_convs.append(nn.Conv2d(in_ch_dec, 1, kernel_size=1, bias=True))

    def forward(self, x: torch.Tensor, bc: torch.Tensor) -> torch.Tensor:
        """
        x: (B, in_channels, H, W)
        bc: (B, fc_in_channels)
        Возвращает (B, out_channels, H, W)
        """
        B, _, H, W = x.shape

        # Энкодер + сбор skip-фич
        skips: list[torch.Tensor] = []
        out = x
        for i, enc in enumerate(self.enc_blocks):
            out = enc(out)
            if i < len(self.enc_blocks) - 1:
                skips.append(out)      # сохраняем до пулинга
                out = self.pool(out)

        # Теперь out — бутылочное горлышко: (B, Cb, Hb, Wb)
        _, Cb, Hb, Wb = out.shape
        # Встраиваем BC
        bc_emb = self.bc_module(bc, spatial_shape=(Hb, Wb))
        out = out + bc_emb

        # Для каждой ветки-декодера: прогоняем декодирование
        outputs = []
        for branch_idx in range(self.out_channels):
            x_d = out
            dec_blocks = self.decoders[branch_idx]
            # Берём skip в обратном порядке
            for i, dec_block in enumerate(dec_blocks):
                skip_feat = skips[::-1][i]
                x_d = dec_block(x_d, skip_feat)
            # После всех DecoderBlock'ов: x_d имеет размер (B, dec_last_ch, H, W)
            out_chan = self.final_convs[branch_idx](x_d)  # (B, 1, H, W)
            outputs.append(out_chan)

        # Склеиваем по каналам → (B, out_channels, H, W)
        return torch.cat(outputs, dim=1)


# ==============================
# Пример использования и проверка
# ==============================
if __name__ == "__main__":
    # Конфигурация модели
    model_config = dict(
        in_channels=3,
        out_channels=4,
        fc_in_channels=6,
    )

    # Параметры энкодера/декодера (можно менять)
    encoder_channels = [64, 128, 256, 512]
    decoder_channels = [256, 128, 64]
    bc_hidden_dims = [32, 128]

    model = UNetMultiDecoderBC(
        in_channels=model_config["in_channels"],
        out_channels=model_config["out_channels"],
        fc_in_channels=model_config["fc_in_channels"],
        encoder_channels=encoder_channels,
        decoder_channels=decoder_channels,
        bc_hidden_dims=bc_hidden_dims,
    )

    # Проверяем на трёх размерах из датасетов:
    for H, W in [(128, 625), (256, 1250), (512, 2500)]:
        x = torch.randn(2, 3, H, W)
        bc_vec = torch.randn(2, 6)
        out = model(x, bc_vec)
        print(f"Input size:  ({H:4d}, {W:5d}) → Output size: {tuple(out.shape[2:])}")
        # Должно печатать: Output size: (H, W) точно = (H, W)
