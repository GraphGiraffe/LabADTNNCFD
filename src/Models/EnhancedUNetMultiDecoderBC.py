import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from typing import List, Optional, Type


###############################################
# Вспомогательные функции для создания блоков
###############################################

def conv_block(
    in_channels: int,
    out_channels: int,
    kernel_size: int = 3,
    layers: int = 2,
    use_weightnorm: bool = True,
    use_batchnorm: bool = True,
    activation: Type[nn.Module] = nn.ReLU,
) -> nn.Sequential:
    """
    Один сверточный блок из `layers` свёрток:
      (Conv2d → (WeightNorm) → ReLU → (BatchNorm)) × layers,
    с padding=kernel_size//2, чтобы сохранять H×W.
    """
    assert kernel_size % 2 == 1, "kernel_size должен быть нечётным."
    modules: List[nn.Module] = []
    for i in range(layers):
        in_ch = in_channels if i == 0 else out_channels
        conv = nn.Conv2d(in_ch, out_channels, kernel_size, padding=kernel_size // 2,
                         bias=not use_batchnorm)
        if use_weightnorm:
            conv = weight_norm(conv)
        modules.append(conv)
        modules.append(activation(inplace=True))
        if use_batchnorm:
            modules.append(nn.BatchNorm2d(out_channels))
    return nn.Sequential(*modules)


class BCBottle(nn.Module):
    """
    BC‐MLP для «бутылочного горлышка»: принимает (B, fc_in_channels) → MLP → (B, bottle_ch),
    разворачивает до (B, bottle_ch, Hb, Wb) и добавляет к фичам.
    """
    def __init__(
        self,
        fc_in_channels: int,
        hidden_dims: List[int],
        bottle_channels: int
    ):
        super(BCBottle, self).__init__()
        layers: List[nn.Module] = []
        prev = fc_in_channels
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU(inplace=True))
            prev = h
        layers.append(nn.Linear(prev, bottle_channels))
        self.mlp = nn.Sequential(*layers)

    def forward(self, bc: torch.Tensor, spatial: torch.Size) -> torch.Tensor:
        """
        bc: (B, fc_in_channels)
        spatial: torch.Size([*, *, Hb, Wb]) или tuple (Hb, Wb)
        возвращает (B, bottle_channels, Hb, Wb)
        """
        if isinstance(spatial, torch.Size):
            Hb, Wb = spatial[-2], spatial[-1]
        else:
            Hb, Wb = spatial
        emb = self.mlp(bc)                     # (B, bottle_ch)
        B, Cb = emb.shape
        emb = emb.view(B, Cb, 1, 1).expand(B, Cb, Hb, Wb)
        return emb


class BCDecodeBlock(nn.Module):
    """
    BC‐MLP для уровня декодера: 
    принимает (B, fc_in_channels) → MLP → (B, fc_out_channels),
    потом при нужном H×W разворачивается в (B, fc_out, H, W) для конкатенации.
    """
    def __init__(
        self,
        fc_in_channels: int,
        hidden_dims: List[int],
        fc_out_channels: int
    ):
        super(BCDecodeBlock, self).__init__()
        layers: List[nn.Module] = []
        prev = fc_in_channels
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU(inplace=True))
            prev = h
        layers.append(nn.Linear(prev, fc_out_channels))
        self.mlp = nn.Sequential(*layers)

    def forward(self, bc: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """
        bc: (B, fc_in_channels)
        H, W: целевые spatial
        возвращает (B, fc_out_channels, H, W)
        """
        emb = self.mlp(bc)               # (B, fc_out)
        B, Cout = emb.shape
        emb = emb.view(B, Cout, 1, 1).expand(B, Cout, H, W)
        return emb


###############################################
# Основная модель: Enhanced UNetMultiDecoderBC
###############################################

class EnhancedUNetMultiDecoderBC(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        fc_in_channels: int,
        encoder_filters: List[int],
        layers_per_block: int = 2,
        decoder_filters: Optional[List[int]] = None,
        bc_bottle_hidden: Optional[List[int]] = None,
        bc_decode_hidden: Optional[List[int]] = None,
        fc_out_channels: int = 8,
        add_fc_decode: Optional[List[bool]] = None,
        use_weightnorm: bool = True,
        use_batchnorm: bool = True,
        activation: Type[nn.Module] = nn.ReLU,
    ):
        """
        in_channels: число каналов входного тензора (например, 3)
        out_channels: число выходных веток (например, 4: u,v,p,T)
        fc_in_channels: размер BC‐вектора (например, 6)
        encoder_filters: список [f0, f1, ..., f_{L-1}] для encoder. Длина L>=2.
        layers_per_block: сколько conv‐слоёв в каждом «ConvBlock».
        decoder_filters: список размеров каналов для каждого decode‐уровня. Если None, взять encoder_filters[::-1][1:].
                         Должен иметь длину L-1.
        bc_bottle_hidden: список скрытых слоёв для BC на бутылке (например, [32,128]). 
                         Если None, MLP будет один линейный слой.
        bc_decode_hidden: список скрытых слоёв для BC на каждом decode‐уровне (например, [16,32]). 
                         Если None, MLP будет один линейный слой.
        fc_out_channels: число каналов, которые выдаёт каждый BCDecodeBlock.
        add_fc_decode: булев список длины L-1, указывающий, нужно ли вставлять BCIL на каждом decode‐уровне. 
                       Если None, по умолчанию вставляем только на глубочайшем decode‐уровне (уровень 0).
        use_weightnorm, use_batchnorm, activation: гиперпараметры слоёв.
        """
        super().__init__()
        assert len(encoder_filters) >= 2, "encoder_filters должен иметь длину >=2."
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.fc_in_channels = fc_in_channels

        # Количество уровней в энкодере
        self.num_levels = len(encoder_filters)       # L
        # Количество decode-уровней = L - 1
        self.num_decode = self.num_levels - 1        # L-1

        # Если decoder_filters не передали, зеркально возьмём encoder_filters без первого:
        if decoder_filters is None:
            # encoder=[f0,f1,f2,f3] → decoder=[f2,f1,f0]
            self.decoder_filters = encoder_filters[::-1][1:]
        else:
            assert len(decoder_filters) == self.num_decode, "decoder_filters должен иметь длину L-1."
            self.decoder_filters = decoder_filters

        # ========== Модуль BC на «бутылочном горлышке» ==========
        bchid = bc_bottle_hidden or []
        self.bc_bottle = BCBottle(
            fc_in_channels=fc_in_channels,
            hidden_dims=bchid,
            bottle_channels=encoder_filters[-1],
        )

        # ========== Решаем, на каких decode-уровнях встраивать BC ==========
        if add_fc_decode is None:
            # По умолчанию: только первый уровень decode (самый глубокий) получает BC
            self.add_fc_decode = [False] * self.num_decode
            self.add_fc_decode[0] = True
        else:
            assert len(add_fc_decode) == self.num_decode, "add_fc_decode должен иметь длину L-1."
            self.add_fc_decode = add_fc_decode

        # ========== Модули BC для decode-уровней ==========
        self.bc_decode_blocks = nn.ModuleList()
        for lvl in range(self.num_decode):
            lvl_blocks = nn.ModuleList()
            for _ in range(out_channels):
                if self.add_fc_decode[lvl]:
                    lvl_blocks.append(
                        BCDecodeBlock(
                            fc_in_channels=fc_in_channels,
                            hidden_dims=bc_decode_hidden or [],
                            fc_out_channels=fc_out_channels,
                        )
                    )
                else:
                    lvl_blocks.append(None)
            self.bc_decode_blocks.append(lvl_blocks)

        # ========== Построим энкодер ==========
        self.encoder_blocks = nn.ModuleList()
        prev_ch = in_channels
        for f in encoder_filters:
            block = conv_block(
                in_channels=prev_ch,
                out_channels=f,
                kernel_size=3,
                layers=layers_per_block,
                use_weightnorm=use_weightnorm,
                use_batchnorm=use_batchnorm,
                activation=activation,
            )
            self.encoder_blocks.append(block)
            prev_ch = f

        # ========== Построим проекции для Unpool ==========
        skip_channels_rev = encoder_filters[::-1][1:]  # [encoder_filters[-2], encoder_filters[-3], ...]
        self.unpool_projs = nn.ModuleList()
        for lvl in range(self.num_decode):
            if lvl == 0:
                in_ch = encoder_filters[-1]   # на самом глубоком уровне надо проецировать из bottle_channels
            else:
                in_ch = self.decoder_filters[lvl - 1]
            out_ch = skip_channels_rev[lvl]  # число каналов skip на этом уровне
            conv1x1 = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=not use_batchnorm)
            if use_weightnorm:
                conv1x1 = weight_norm(conv1x1)
            if use_batchnorm:
                conv1x1 = nn.Sequential(conv1x1, nn.BatchNorm2d(out_ch))
            self.unpool_projs.append(conv1x1)

        # ========== Построим декодер для каждой выходной ветки ==========
        self.decoders = nn.ModuleList()
        for branch in range(out_channels):
            branch_blocks = nn.ModuleList()
            for lvl in range(self.num_decode):
                skip_ch = skip_channels_rev[lvl]
                if lvl == 0:
                    # на первом decode-уровне дверь prior_ch = encoder_filters[-1]
                    pass  
                # после unpool→in_ch_proj→ skip_ch, плюс, возможно, fc_out_channels
                cat_ch = skip_ch + skip_ch + (fc_out_channels if self.add_fc_decode[lvl] else 0)
                out_ch = self.decoder_filters[lvl]
                block = conv_block(
                    in_channels=cat_ch,
                    out_channels=out_ch,
                    kernel_size=3,
                    layers=layers_per_block,
                    use_weightnorm=use_weightnorm,
                    use_batchnorm=use_batchnorm,
                    activation=activation,
                )
                branch_blocks.append(block)
            # После всех decode-уровней: финальный 1×1 conv
            final_conv = nn.Conv2d(self.decoder_filters[-1], 1, kernel_size=1)
            if use_weightnorm:
                final_conv = weight_norm(final_conv)
            branch_blocks.append(final_conv)
            self.decoders.append(branch_blocks)

        # ========== Пулинг / Анпулинг ==========
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        # Для анпулинга будем использовать F.max_unpool2d

    def encode(self, x: torch.Tensor):
        """
        Пропустить через энкодер, вернуть:
          - bottle: фича на глубине (бутылочное горлышко) после последнего conv
          - skips: список фич на каждом уровне до pool, len = L-1
          - indices: список индексов для unpool, len = L-1
          - sizes: список размеров фич до pool, len = L-1
        """
        skips: List[torch.Tensor] = []
        indices: List[torch.Tensor] = []
        sizes: List[torch.Size] = []
        out = x
        for i, block in enumerate(self.encoder_blocks):
            out = block(out)
            if i < self.num_levels - 1:
                skips.append(out)
                sizes.append(out.size())
                out, ind = self.pool(out)
                indices.append(ind)
        return out, skips, indices, sizes

    def decode_branch(
        self,
        bottle: torch.Tensor,
        bc: torch.Tensor,
        skips: List[torch.Tensor],
        indices: List[torch.Tensor],
        sizes: List[torch.Size],
        branch_idx: int
    ) -> torch.Tensor:
        """
        Декодирование одной ветки:
        - bottle: (B, bottle_ch, Hb, Wb) после BC добавления
        - bc: (B, fc_in_channels)
        - skips: [фича уровня 0, ..., L-2]  (до pool)
        - indices: [индекс уровня 0, ..., L-2]
        - sizes: [размер до pool уровня 0, ..., L-2]
        Возвращает выход (B, 1, H_in, W_in).
        """
        device = next(self.parameters()).device
        bottle = bottle.to(device)
        bc = bc.to(device)
        out = bottle

        for lvl in range(self.num_decode):
            # 1) Проекция для unpool
            proj = self.unpool_projs[lvl].to(device)
            out_proj = proj(out)  # (B, skip_ch, Hb, Wb)

            # 2) Анпулинг
            ind = indices[-1 - lvl].to(device)
            size = sizes[-1 - lvl]
            out_un = F.max_unpool2d(out_proj, ind, kernel_size=2, stride=2, output_size=size)

            # 3) Конкатенация с skip и BC-map (если нужно)
            skip_feat = skips[-1 - lvl].to(device)  # (B, skip_ch, H_lvl, W_lvl)
            H_lvl, W_lvl = skip_feat.shape[-2], skip_feat.shape[-1]

            if self.add_fc_decode[lvl]:
                bc_map = self.bc_decode_blocks[lvl][branch_idx](bc, H_lvl, W_lvl).to(device)
                cat = torch.cat([skip_feat, out_un, bc_map], dim=1)
            else:
                cat = torch.cat([skip_feat, out_un], dim=1)

            # 4) Свёртка через conv_block
            block = self.decoders[branch_idx][lvl].to(device)
            out = block(cat)

        # После всех decode-уровней: финальный 1×1 conv
        final_conv = self.decoders[branch_idx][-1].to(device)
        out = final_conv(out)
        return out

    def forward(self, x: torch.Tensor, bc: torch.Tensor) -> torch.Tensor:
        """
        x: (B, in_channels, H, W)
        bc: (B, fc_in_channels)
        Возвращает (B, out_channels, H, W)
        """
        # Переносим вход на тот же девайс, где лежат веса модели:
        device = next(self.parameters()).device
        x = x.to(device)
        bc = bc.to(device)

        # --- Кодирование ---
        bottle, skips, indices, sizes = self.encode(x)

        # --- Встраивание BC в бутылочное горлышко ---
        bc_emb = self.bc_bottle(bc, bottle.shape).to(device)
        bottle = bottle + bc_emb

        # --- Мульти‐декодирование ---
        outputs: List[torch.Tensor] = []
        for b in range(self.out_channels):
            out_b = self.decode_branch(bottle, bc, skips, indices, sizes, branch_idx=b)
            outputs.append(out_b)

        # Склеиваем по каналам → (B, out_channels, H, W)
        result = torch.cat(outputs, dim=1)
        return result


###############################################
# Пример использования и проверка
###############################################

if __name__ == "__main__":
    # Конфигурация
    model_config = dict(
        in_channels=3,
        out_channels=4,
        fc_in_channels=6,
    )

    # Пример фильтров для энкодера
    encoder_filters = [32, 64, 128, 256]   # L = 4 уровней
    # Тогда декодеру нужно L-1 = 3 фильтра:
    decoder_filters = [128, 64, 32]

    # Решаем, на каких decode-уровнях встраивать BC (от глубокого к мелкому)
    add_fc_decode = [True, False, True]

    # Скрытые слои для BC в бутылке и на decode-уровнях
    bc_bottle_hidden = [32, 128]
    bc_decode_hidden = [16, 32]
    fc_out_channels = 8

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Создаём модель и при необходимости перемещаем на GPU
    model = EnhancedUNetMultiDecoderBC(
        in_channels=model_config["in_channels"],
        out_channels=model_config["out_channels"],
        fc_in_channels=model_config["fc_in_channels"],
        encoder_filters=encoder_filters,
        layers_per_block=2,
        decoder_filters=decoder_filters,
        bc_bottle_hidden=bc_bottle_hidden,
        bc_decode_hidden=bc_decode_hidden,
        fc_out_channels=fc_out_channels,
        add_fc_decode=add_fc_decode,
        use_weightnorm=True,
        use_batchnorm=True,
        activation=nn.ReLU,
    ).to(device) # пример, если у вас есть GPU

    # Проверим на трёх размерах датасетов
    for H, W in [(128, 625), (256, 1250), (512, 2500)]:
        x = torch.randn(2, 3, H, W)
        bc_vec = torch.randn(2, 6)
        # Переносим входы на тот же девайс, что и модель:
        x = x.to(device)
        bc_vec = bc_vec.to(device)
        out = model(x, bc_vec)
        print(f"Input ({H:4d}×{W:4d}) → Output {tuple(out.shape[2:])}")
        # Должно быть точно (H, W)
