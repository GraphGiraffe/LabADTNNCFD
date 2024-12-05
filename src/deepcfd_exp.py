from pathlib import Path, PosixPath
import importlib
import copy
import json
from types import SimpleNamespace
from collections import defaultdict

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm
# import pandas as pd
# import cv2
# import seaborn as sns
# import matplotlib.pyplot as plt
# from mpl_toolkits.axes_grid1 import make_axes_locatable

from clearml import Task, OutputModel
from pprint import pprint

from src.deepcfd_utils import (
    prepare_datasets
)
torch.manual_seed(0)

class Trainer():
    def __init__(self, p, criterion, optimizer=None, scheduler=None, metrics=[]):
        self.p = p
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.metrics = metrics

    def calc_metrics_on_epoch(self):
        pass
    
    def calc_metrics_on_batch(self):
        pass
        
    def epoch(self, model, loader, train=True):
        loss_list = []
        metrics_dict = defaultdict(list)
        
        for batch in loader:
            x, x_fc, y = [v.to(model.device) for v in batch]
            
            pred = model(x, x_fc)
            loss = self.criterion(y, pred)
            loss_list.append(loss.item())
            for name, fn in self.metrics.items():
                metrics_dict[name].append(fn(y, pred))

            if train:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
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
            loss_cur = loss_MSE(target[:,idx,:,:], pred[:,idx,:,:]).reshape((pred.shape[0], 1, pred.shape[2], pred.shape[3])) 
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

def experiment(p_in, run_clear_ml=False, log_dir=None):
    p = copy.copy(p_in)
    
    if log_dir is None:
        log_dir = Path('tmp')
    log_dir.mkdir(parents=True, exist_ok=True)
    
    p_dump = copy.copy(p)
    for k, v in p_dump.items():
        for kk, vv in v.items():
            if isinstance(vv, PosixPath):
                p_dump[k][kk] = str(vv)
    with open(log_dir / 'params.json', 'w') as f:
        json.dump(p_dump, f, indent=4)
        
    # with open(log_dir / 'params.json', 'r') as f:
    #     loaded_params = **json.load(f)
    
    for k, v in p.items():
        p[k] = SimpleNamespace(**v)
    p = SimpleNamespace(**p)
    pprint(p_dump)
    
    train_dataset, val_dataset, test_dataset = prepare_datasets(p.dataset, model_modes=p.model.modes)
    train_loader = DataLoader(train_dataset, shuffle=True, **vars(p.dataloader))
    val_loader = DataLoader(val_dataset, shuffle=False, **vars(p.dataloader))
    test_loader = DataLoader(test_dataset, shuffle=False, **vars(p.dataloader))
    channels_weights = torch.FloatTensor(train_dataset.out_mean_norm)
    

    if 'added_fc' in p.model.modes:
        p.model.add_fc_blocks = []
        for idx in range(len(p.model.filters)):
            ans = p.model.add_fc_blocks_every_N > 0 and idx % p.model.add_fc_blocks_every_N == 0
            p.model.add_fc_blocks.append(ans)
    
    if 'bc_in_x' in p.model.modes:
        p.model.in_channels += p.model.BCinX_channels
        p.model.add_fc_blocks = [False] * len(p.model.filters)
    
    model_fn = getattr(importlib.import_module(f'src.Models.{p.model.name}'), p.model.name)
    del p.model.name
    del p.model.modes
    del p.model.BCinX_channels
    del p.model.add_fc_blocks_every_N
    p.model.device = torch.device(p.model.device)
    model = model_fn(**vars(p.model))
    model = model.to(p.model.device)
    params_to_device(model, p.model.device)
    
    optimizer_fn = getattr(importlib.import_module('torch.optim'), p.optimizer.name)
    del p.optimizer.name
    optimizer = optimizer_fn(model.parameters(), **vars(p.optimizer))
    
    if p.scheduler.name is not None:
        scheduler_fn = getattr(importlib.import_module('torch.optim.lr_scheduler'), p.scheduler.name)
        del p.scheduler.name
        scheduler = scheduler_fn(optimizer, **vars(p.scheduler))
    else:
        scheduler = None

    if run_clear_ml:
        task = Task.init(project_name="DeepCFD_FC",
                         task_name=str(log_dir),
                         output_uri=False)
        
        model_p_dump = p_dump.get('model')
        task.connect(p_dump)
        output_model = OutputModel(task=task)
        output_model.update_design(config_dict=model_p_dump)
    else:
        task = None
    writer = SummaryWriter(log_dir=log_dir)

    criterion = gen_loss_func(channels_weights=channels_weights)
    
    metrics = dict(
        mse=lambda target, pred: MSE(target, pred),
        u=lambda target, pred: MSE(target[:,0], pred[:,0]),
        v=lambda target, pred: MSE(target[:,1], pred[:,1]),
        p=lambda target, pred: MSE(target[:,2], pred[:,2]),
        T=lambda target, pred: MSE(target[:,3], pred[:,3]),
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
                torch.save(model, log_dir / 'best_model.pth')

            if scheduler is not None:
                scheduler.step()
                
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
