from pathlib import Path, PosixPath
import importlib
import copy
import json
from types import SimpleNamespace
from collections import defaultdict

import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LinearLR, SequentialLR

from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary
from tqdm import tqdm
# import pandas as pd
# import cv2
# import seaborn as sns
# import matplotlib.pyplot as plt
# from mpl_toolkits.axes_grid1 import make_axes_locatable

from clearml import Task, OutputModel
from pprint import pprint

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from src.deepcfd_utils import (
    prepare_datasets,
    dump_norm_data,
    load_norm_data,
    normalize_sample,
    denormalize_sample
)
torch.manual_seed(0)


class Trainer():
    def __init__(self, p, criterion, optimizer=None, scheduler=None, metrics=[], device='cuda'):
        self.p = p
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.metrics = metrics
        self.device = device

    def calc_metrics_on_epoch(self):
        pass

    def calc_metrics_on_batch(self):
        pass

    def epoch(self, model, loader, train=True):
        loss_list = []
        metrics_dict = defaultdict(list)

        for batch in loader:
            x, x_fc, y = [v.to(self.device) for v in batch]

            pred = model(x, x_fc)
            loss = self.criterion(y, pred)
            loss_list.append(loss.item())
            for name, fn in self.metrics.items():
                metrics_dict[name].append(fn(y, pred))

            if train:
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                self.optimizer.step()
        if train:
            if self.scheduler is not None:
                self.scheduler.step()

        s = loader.dataset.out_list.shape
        values_number = s[0] * s[2] * s[3]  # samples_num * num_x * num_y

        total_metrics = dict(loss=sum(loss_list) / len(loader))
        for k, v in metrics_dict.items():
            total_metrics[k] = sum(v) / values_number

        return total_metrics

    def train_epoch(self, model, loader):
        model.train()
        metrics = self.epoch(model, loader, train=True)
        if self.scheduler is not None:
            lr = self.scheduler.get_last_lr()[0]
        else:
            lr = self.optimizer.param_groups[0]['lr']
        return lr, metrics

    def valid_epoch(self, model, loader):
        model.eval()
        with torch.no_grad():
            metrics = self.epoch(model, loader, train=False)
        return metrics

    def test(self, model, loader):
        metrics = self.valid_epoch(model, loader)

        return metrics


def MSE(target, pred):
    return float(torch.sum((pred - target) ** 2))


def gen_loss_func(channels_weights):
    def loss_MSE(target, pred):
        return (pred - target) ** 2

    def loss_MAE(target, pred):
        return torch.abs(pred - target)

    def loss_func(target, pred):
        loss_list = list()
        for idx in range(pred.shape[1]):
            loss_cur = loss_MSE(target[:, idx, :, :], pred[:, idx, :, :]).reshape(
                (pred.shape[0], 1, pred.shape[2], pred.shape[3]))
            loss_list.append(loss_cur*channels_weights[idx])

        loss = torch.stack(loss_list, dim=0).sum(dim=0)

        return torch.sum(loss)/target.shape[0]/target.shape[2]/target.shape[3]
    return loss_func


def params_to_device(model, device):
    for vv in model.fc_blocks_decoder:
        for v in vv:
            if v is not None:
                v = v.to(device)
    model.device = device


def exp(p_in, run_clear_ml=False, exp_dir_path=None):
    p = copy.copy(p_in)

    if exp_dir_path is None:
        exp_dir_path = Path('tmp')
    exp_dir_path.mkdir(parents=True, exist_ok=True)

    p_dump = copy.copy(p)
    for k, v in p_dump.items():
        for kk, vv in v.items():
            if isinstance(vv, PosixPath):
                p_dump[k][kk] = str(vv)
    with open(exp_dir_path / 'params.json', 'w') as f:
        json.dump(p_dump, f, indent=4)

    # with open(log_dir / 'params.json', 'r') as f:
    #     loaded_params = **json.load(f)

    for k, v in p.items():
        p[k] = SimpleNamespace(**v)
    p = SimpleNamespace(**p)

    if 'modes' not in vars(p.model).keys():
        p.model.modes = []
    pprint(p_dump)

    train_dataset_fn, val_dataset_fn, test_dataset_fn = prepare_datasets(p.dataset, model_modes=p.model.modes)
    train_loader = DataLoader(train_dataset_fn(None), shuffle=True, **vars(p.dataloader))
    val_loader = DataLoader(val_dataset_fn(train_loader.dataset.norm_data), shuffle=False, **vars(p.dataloader))
    # test_loader = DataLoader(test_dataset_fn(train_loader.dataset.norm_data), shuffle=False, **vars(p.dataloader))
    # channels_weights = torch.FloatTensor(train_loader.dataset.out_mean_norm)
    channels_weights = torch.FloatTensor([1, 1, 1, 1])

    dump_norm_data(train_loader.dataset.norm_data, exp_dir_path / 'norm_data.json')

    if 'added_fc' in p.model.modes:
        p.model.add_fc_blocks = []
        for idx in range(len(p.model.filters)):
            ans = p.model.add_fc_blocks_every_N > 0 and idx % p.model.add_fc_blocks_every_N == 0
            p.model.add_fc_blocks.append(ans)

    if 'bc_in_x' in p.model.modes:
        p.model.in_channels += p.model.BCinX_channels
        # p.model.add_fc_blocks = [False] * len(p.model.filters)

    model_fn = getattr(importlib.import_module(
        f'src.Models.{p.model.name}'), p.model.name)
    del p.model.name
    del p.model.modes
    if 'BCinX_channels' in vars(p.model).keys():
        del p.model.BCinX_channels
    if 'add_fc_blocks_every_N' in vars(p.model).keys():
        del p.model.add_fc_blocks_every_N
    device = torch.device('cuda')
    model = model_fn(**vars(p.model))
    model = model.to(device)
    # params_to_device(model, p.model.device)
    summary(model)

    optimizer_fn = getattr(importlib.import_module(
        'torch.optim'), p.optimizer.name)
    del p.optimizer.name
    optimizer = optimizer_fn(model.parameters(), **vars(p.optimizer))

    if p.scheduler.name is not None:
        scheduler_fn = getattr(importlib.import_module(
            'torch.optim.lr_scheduler'), p.scheduler.name)
        del p.scheduler.name
        if 'warmup_epochs' in vars(p.scheduler).keys() and p.scheduler.warmup_epochs not in [None, 0]:
            warmup_epochs = p.scheduler.warmup_epochs
            del p.scheduler.warmup_epochs
            main_scheduler = scheduler_fn(optimizer, **vars(p.scheduler))
            
            # Сохраняем базовые скорости
            base_lr = [group["lr"] for group in optimizer.param_groups][0]
            # # Делаем «глубокую» копию и на ней прогоняем warmup_epochs раз
            # tmp_sched = copy.deepcopy(main_scheduler)
            # for _ in range(warmup_epochs):
            #     tmp_sched.step()
            # # Получаем обученные LR после симуляции
            # end_lr = tmp_sched.get_last_lr()[0]
            end_lr = base_lr

            warmup_start_lr = 1e-6
            start_factor = warmup_start_lr / base_lr
            end_factor   = end_lr  / base_lr

            warmup_scheduler = LinearLR(
                optimizer,
                start_factor=start_factor,
                end_factor=end_factor,
                total_iters=warmup_epochs
            )
            
            # scheduler_kwargs = vars(p.scheduler)
            # scheduler_kwargs
            # main_scheduler = scheduler_fn(optimizer, **scheduler_kwargs)

            scheduler = SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, main_scheduler],
                milestones=[warmup_epochs]
            )

        else:
            if 'warmup_epochs' in vars(p.scheduler).keys():
                del p.scheduler.warmup_epochs
            scheduler = scheduler_fn(optimizer, **vars(p.scheduler))
    else:
        scheduler = None

    if run_clear_ml:
        task = Task.init(project_name="DeepCFD_FC",
                         task_name=str(exp_dir_path),
                         output_uri=False)

        model_p_dump = p_dump.get('model')
        task.connect(p_dump)
        output_model = OutputModel(task=task)
        output_model.update_design(config_dict=model_p_dump)
    else:
        task = None
    writer = SummaryWriter(log_dir=exp_dir_path)

    criterion = gen_loss_func(channels_weights=channels_weights)

    metrics = dict(
        mse=lambda target, pred: MSE(target, pred),
        u=lambda target, pred: MSE(target[:, 0], pred[:, 0]),
        v=lambda target, pred: MSE(target[:, 1], pred[:, 1]),
        p=lambda target, pred: MSE(target[:, 2], pred[:, 2]),
        T=lambda target, pred: MSE(target[:, 3], pred[:, 3]),
    )
    trainer = Trainer(p,
                      criterion,
                      optimizer=optimizer,
                      scheduler=scheduler,
                      metrics=metrics)

    best_score = torch.inf
    with tqdm(total=p.train.epochs, desc="Epochs", unit="epoch") as pbar:
        for epoch in range(0, p.train.epochs):
            writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)

            lr, train_metrics = trainer.train_epoch(model, train_loader)
            for k, v in train_metrics.items():
                writer.add_scalar(f'{k}/train', v, epoch)
                # print(f"{' '.join(k.split('_')).title()}: {v:.2f}")

            valid_metrics = trainer.valid_epoch(model, val_loader)
            for k, v in valid_metrics.items():
                writer.add_scalar(f'{k}/val', v, epoch)
                # print(f"{' '.join(k.split('_')).title()}: {v:.2f}")

            # do something (save model, change lr, etc.)
            if best_score > valid_metrics[p.train.score_metric]:
                best_epoch = epoch
                best_score = valid_metrics[p.train.score_metric]
                torch.save(model, exp_dir_path / 'best_model.pth')

            # if scheduler is not None:
            #     scheduler.step()

            pbar.set_postfix({
                'best': f'{best_epoch+1:04d}',
                'LR': f'{lr:7.1e}',
                'Train': '|'.join([f'{k} {v:7.1e}' for k, v in train_metrics.items()]),
                'Val': '|'.join([f'{k} {v:7.1e}' for k, v in valid_metrics.items()]),
            })
            pbar.update(1)  # Обновляем прогресс на одну эпоху

    # trainer.test(model, test_loader)

    writer.close()
    if run_clear_ml:
        task.close()


def create_result_image(flow, pred_flow):
    error_flow = np.abs(pred_flow - flow)

    # Настройка фигуры и GridSpec
    fontsize = 30
    fontsize_cb = 20
    figsize = (60, 25)
    nrows = flow.shape[0]
    ncols = 5
    width_ratios = [1, 1, 0.05, 1, 0.05]  # gt, pred, cax, error, cax_err
    height_ratios = [1] * nrows
    cmap = 'viridis'

    # Заголовки столбцов и названия параметров
    col_titles = ['Ground Truth', 'Predicted', 'Abs Error']
    param_names = ['u', 'v', 'p', 'T']

    fig = plt.figure(figsize=figsize)
    gs = GridSpec(nrows=nrows, ncols=ncols,
                  figure=fig,
                  width_ratios=width_ratios,
                  height_ratios=height_ratios,
                  hspace=0.15,
                  wspace=0.15)

    for row_idx in range(nrows):
        # Данные для текущего параметра
        gt_data = flow[row_idx]
        pred_data = pred_flow[row_idx]
        err_data = error_flow[row_idx]

        # Определение общего масштаба для Flow
        combined = np.concatenate([gt_data, pred_data])
        vmin, vmax = np.min(combined), np.max(combined)

        # Определение общего масштаба Error
        err_vmin, err_vmax = np.min(err_data), np.max(err_data)

        # Создание осей для текущей строки
        ax_gt = fig.add_subplot(gs[row_idx, 0])
        ax_pred = fig.add_subplot(gs[row_idx, 1])
        cax = fig.add_subplot(gs[row_idx, 2])  # Ось для цветбара Flow
        ax_err = fig.add_subplot(gs[row_idx, 3])
        cax_err = fig.add_subplot(gs[row_idx, 4])  # Ось для цветбара Error

        # Визуализация данных
        im = ax_gt.imshow(gt_data, cmap=cmap, vmin=vmin, vmax=vmax)
        ax_pred.imshow(pred_data, cmap=cmap, vmin=vmin, vmax=vmax)
        im_err = ax_err.imshow(err_data, cmap=cmap, vmin=err_vmin, vmax=err_vmax)

        # Удаление осей
        for ax in [ax_gt, ax_pred, ax_err]:
            ax.set_xticks([])
            ax.set_yticks([])

        def format_cbar_ticks(x, _): return f"{x:0.3f}"

        # Добавление цветбара Flow
        plt.colorbar(im, cax=cax, format=format_cbar_ticks)
        cax.tick_params(labelsize=fontsize_cb)

        # Добавление цветбара Error
        plt.colorbar(im_err, cax=cax_err, format=format_cbar_ticks)
        cax_err.tick_params(labelsize=fontsize_cb)

        # Подписи слева (параметры)
        ax_gt.set_ylabel(
            param_names[row_idx], rotation=0, fontsize=fontsize, labelpad=20, va='center')

        # Заголовки столбцов (только для первой строки)
        if row_idx == 0:
            ax_gt.set_title(col_titles[0], fontsize=fontsize)
            ax_pred.set_title(col_titles[1], fontsize=fontsize)
            ax_err.set_title(col_titles[2], fontsize=fontsize)


def test_exp(exp_dir_path, out_dir_path,
             total_samples=None,
             datasets_dir=None,
             batch_size=None,
             num_samples_to_draw=None,
             test_sample_num=None):
    """Тестирование модели и сохранение результатов."""
    exp_dir_path = Path(exp_dir_path)
    out_dir_path = Path(out_dir_path)

    # Загрузка конфигурации эксперимента
    with open(exp_dir_path / 'params.json', 'r') as f:
        p = json.load(f)

    for k, v in p.items():
        p[k] = SimpleNamespace(**v)
    p = SimpleNamespace(**p)

    if total_samples is not None:
        p.dataset.total_samples = total_samples
    if datasets_dir is not None:
        p.dataset.datasets_dir = datasets_dir
    if batch_size is not None:
        p.dataloader.batch_size = batch_size

    train_loader, val_loader, test_dataset_fn = prepare_datasets(
        p.dataset, model_modes=p.model.modes)

    norm_data_fp = exp_dir_path / 'norm_data.json'
    if not norm_data_fp.exists():
        norm_data_fp = Path(p.datasets_dir) / p.dataset_name / 'norm_data.json'
    norm_data = load_norm_data(norm_data_fp)

    # train_loader = DataLoader(train_dataset_fn(norm_data), shuffle=True, **vars(p.dataloader))
    # val_loader = DataLoader(val_dataset_fn(norm_data), shuffle=False, **vars(p.dataloader))
    test_loader = DataLoader(test_dataset_fn(norm_data, sample_num=test_sample_num), shuffle=False, **vars(p.dataloader))
    channels_weights = torch.FloatTensor(test_loader.dataset.out_mean_norm)

    in_min = test_loader.dataset.in_min
    in_max = test_loader.dataset.in_max
    out_min = test_loader.dataset.out_min
    out_max = test_loader.dataset.out_max

    if 'added_fc' in p.model.modes:
        p.model.add_fc_blocks = []
        for idx in range(len(p.model.filters)):
            ans = p.model.add_fc_blocks_every_N > 0 and idx % p.model.add_fc_blocks_every_N == 0
            p.model.add_fc_blocks.append(ans)

    if 'bc_in_x' in p.model.modes:
        p.model.in_channels += p.model.BCinX_channels
        # p.model.add_fc_blocks = [False] * len(p.model.filters)

    # model_fn = getattr(importlib.import_module(f'src.Models.{p.model.name}'), p.model.name)
    # del p.model.name
    # del p.model.modes
    # del p.model.BCinX_channels
    # del p.model.add_fc_blocks_every_N
    # p.model.device = torch.device(p.model.device)
    # model = model_fn(**vars(p.model))

    # # Загрузка весов лучшей модели
    model = torch.load(exp_dir_path / 'best_model.pth', weights_only=False)
    # model.load_state_dict(state_dict)
    # model = model.to(p.model.device)
    # model.eval()  # Режим инференса

    device = torch.device('cuda')
    # params_to_device(model, device)
    model = model.to(device)
    summary(model)

    criterion = gen_loss_func(channels_weights=channels_weights)

    metrics = dict(
        mse=lambda target, pred: MSE(target, pred),
        u=lambda target, pred: MSE(target[:, 0], pred[:, 0]),
        v=lambda target, pred: MSE(target[:, 1], pred[:, 1]),
        p=lambda target, pred: MSE(target[:, 2], pred[:, 2]),
        T=lambda target, pred: MSE(target[:, 3], pred[:, 3]),
    )
    trainer = Trainer(p,
                      criterion,
                      optimizer=None,
                      scheduler=None,
                      metrics=metrics)

    # label_list = []
    # flow_list = []
    # pred_flow_list = []
    # error_flow_list = []
    loss_list = []
    metrics_dict = defaultdict(list)
    sample_counter = 0

    out_dir_path.mkdir(exist_ok=True)
    out_images_dir = out_dir_path / 'images'
    out_images_dir.mkdir(exist_ok=True)

    for batch in tqdm(test_loader):
        x, x_fc, y = [v.to(device) for v in batch]

        pred = model(x, x_fc)
        loss = trainer.criterion(y, pred)
        loss_list.append(loss.item())
        for name, fn in trainer.metrics.items():
            metrics_dict[name].append(fn(y, pred))

        x = x.detach().cpu().numpy()
        # x_fc = x_fc.detach().cpu().numpy()
        y = y.detach().cpu().numpy()
        pred = pred.detach().cpu().numpy()

        # denormalization
        # denorm_x = denormalize_sample(x, in_min, in_max)
        # denorm_y = denormalize_sample(y, out_min, out_max)
        # denorm_pred = denormalize_sample(pred, out_min, out_max)

        for sample_idx in range(x.shape[0]):
            sample_y = y[sample_idx, ...]
            sample_pred = pred[sample_idx, ...]

            # denormalization
            # sample_y = denorm_y[sample_idx, ...]
            # sample_pred = denorm_pred[sample_idx, ...]
            # sample_out_min, sample_out_max = sample_y.min(axis=(1, 2)), sample_y.max(axis=(1, 2))
            # sample_y = normalize_sample(sample_y[np.newaxis, ...], sample_out_min, sample_out_max)[0]
            # sample_pred = normalize_sample(sample_pred[np.newaxis, ...], sample_out_min, sample_out_max)[0]

            if num_samples_to_draw is not None and sample_counter < num_samples_to_draw:
                create_result_image(sample_y, sample_pred)
                out_image_fp = out_images_dir / f'{sample_counter:06d}.png'
                plt.savefig(out_image_fp, bbox_inches='tight', pad_inches=1)
                plt.close()
            sample_counter += 1

    s = test_loader.dataset.out_list.shape
    values_number = s[0] * s[2] * s[3]  # samples_num * num_x * num_y

    total_metrics = dict(loss=sum(loss_list) / len(test_loader))
    for k, v in metrics_dict.items():
        total_metrics[k] = sum(v) / values_number

    for k, v in total_metrics.items():
        print(f"{' '.join(k.split('_')).title()}: {v:.2e}")
