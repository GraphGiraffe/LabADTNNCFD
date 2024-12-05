from pathlib import Path

import torch

from src.deepcfd_exp import experiment
from src.deepcfd_utils import get_str_timestamp

# TORCH_HUB_DIR = '/storage0/pia/python/hub/'

# torch.hub.set_dir(TORCH_HUB_DIR)

if __name__ == '__main__':
    test = False

    # Set config
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_clear_ml = True
    out_dir = Path('out')

    # Dataset config
    dataset_config = dict(
        datasets_dir='datasets',
        dataset_name='dataset_with_T_rand_BC',
        obj_types=['spline'],  # ['pol']  # ['pol', 'spline'],
        total_samples=1000,
        train_ratio=0.6,
        val_ratio=0.2
    )

    # Dataloader config
    dataloader_config = dict(
        batch_size=8,
        num_workers=1
    )

    # Model config
    model_config = dict(
        name=None,
        # ['added_fc']  ['bc_in_x']  ['added_fc', 'bc_in_x']
        modes=['added_fc'],
        add_fc_blocks_every_N=1,
        BCinX_channels=2,
        in_channels=3,
        out_channels=4,
        filters=[8, 16, 32, 64, 128, 64, 32, 16],
        kernel_size=3,
        batch_norm=False,
        weight_norm=False,
        fc_in_channels=6,
        fc_out_channels=32,
        fc_filters=[8, 16, 32, 64, 128, 64, 32],
        device=device,
    )

    # Optimizer config
    optimizer_config = dict(
        name='AdamW',
        lr=1e-4,
        weight_decay=0.005
    )

    # Optimizer config
    scheduler_config = dict(
        name=None
    )

    # Train config
    train_config = dict(
        epochs=300,
        patience=-1,
        score_metric='mse',
    )

    params = dict(
        dataset=dataset_config,
        dataloader=dataloader_config,
        model=model_config,
        optimizer=optimizer_config,
        scheduler=scheduler_config,
        train=train_config
    )

    if test:
        run_clear_ml = False
        dataset_config['total_samples'] = 10
        train_config['epochs'] = 5
        out_dir = Path('out_test')

    models = ['UNetExFC']
    for model_name in models:
        for add_fc_blocks_every_N in [3, 2, 1, 0]:
            for obj_types in [['spline'], ['pol'], ['pol', 'spline']]:
                ts = get_str_timestamp()
                params['model']['name'] = model_name
                params['model']['add_fc_blocks_every_N'] = add_fc_blocks_every_N
                params['dataset']['obj_types'] = obj_types
                log_dir = out_dir / f"{params['model']['name']}" / f"{ts}"
                experiment(params, run_clear_ml=run_clear_ml, log_dir=log_dir)
                torch.cuda.empty_cache()
