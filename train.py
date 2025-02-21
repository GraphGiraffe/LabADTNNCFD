from pathlib import Path
import copy

import torch

from src.deepcfd_exp import exp
from src.deepcfd_utils import get_str_timestamp


CASCADE = True
run_clear_ml = True


if __name__ == '__main__':
    debug_run = False

    if CASCADE:
        TORCH_HUB_DIR = '/storage2/pia/python/hub/'
        torch.hub.set_dir(TORCH_HUB_DIR)
        root_dir = '/storage2/pia/python/deepcfd/'
        run_clear_ml = False
    else:
        root_dir = '.'

    root_dir = Path(root_dir)

    # Set config
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path('out')

    # Dataset config
    dataset_config = dict(
        datasets_dir=root_dir / 'datasets',
        dataset_name='.',
        child_dir_samples_num={'dataset_rndshap_Randombc_step_1to256_clean': 500,
                             'dataset_rndshap_Randombc_move_body_step_1to256_clean': 1000,
                             'dataset_rndshap_Randombc_second_body_step_1to256_clean': 500},
        obj_types=['pol'],  # ['pol']  # ['pol', 'spline'],
        total_samples=None,
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
        add_fc_blocks_every_N=None,
        BCinX_channels=2,
        in_channels=3,
        out_channels=4,
        filters=[16, 32, 64, 128, 256, 256, 128, 64, 32],
        layers=3,
        kernel_size=3,
        batch_norm=False,
        weight_norm=False,
        fc_in_channels=6,
        fc_out_channels=8,
        fc_filters=[16, 32, 16],
        device=device,
    )

    # Optimizer config
    optimizer_config = dict(
        name='AdamW',
        lr=1e-3,
        weight_decay=0.005
    )

    # Scheduler config
    epochs = 1000
    gamma = pow(1e-3, 1 / (epochs)) if epochs != 0 else 0
    scheduler_config = dict(
        name='StepLR',
        step_size=1,
        gamma=gamma,
        last_epoch=-1
    )

    # Train config
    train_config = dict(
        epochs=epochs,
        patience=-1,
        score_metric='mse',
    )

    def_params = dict(
        dataset=dataset_config,
        dataloader=dataloader_config,
        model=model_config,
        optimizer=optimizer_config,
        scheduler=scheduler_config,
        train=train_config
    )

    if debug_run:
        run_clear_ml = False
        dataset_config['total_samples'] = None
        dataset_config['child_dir_samples_num'] = {'dataset_rndshap_Randombc_step_1to256_clean': 6,
                                                   'dataset_rndshap_Randombc_move_body_step_1to256_clean': 12,
                                                   'dataset_rndshap_Randombc_second_body_step_1to256_clean': 6}
        dataloader_config['batch_size'] = 1
        train_config['epochs'] = 5
        out_dir = Path('out_test')

    models = ['UNetExFC', 'EnhancedUNet', 'AttentionUNet', 'MultiHeadAttentionUNet']
    add_fc_blocks_every_N_list = [1]
    obj_types_list = [['pol'],
                      #   ['spline']
                      ]
    for model_name in models:
        for add_fc_blocks_every_N in add_fc_blocks_every_N_list:
            for obj_types in obj_types_list:
                params = copy.deepcopy(def_params)

                if model_name in ['AttentionUNet', 'MultiHeadAttentionUNet']:
                    params['model']['dilation'] = 2
                    params['model']['enc_layers'] = params['model']['layers']
                    params['model']['dec_layers'] = params['model']['layers']
                    del params['model']['layers']

                ts = get_str_timestamp()
                params['model']['name'] = model_name
                params['model']['add_fc_blocks_every_N'] = add_fc_blocks_every_N
                params['dataset']['obj_types'] = obj_types
                exp_dir_path = out_dir / f"{params['model']['name']}" / f"{ts}"
                exp(params, run_clear_ml=run_clear_ml, exp_dir_path=exp_dir_path)
                torch.cuda.empty_cache()
