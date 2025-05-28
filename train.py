import sys
from pathlib import Path
import copy
import itertools

import torch

from src.deepcfd_exp import exp
from src.deepcfd_utils import get_str_timestamp


CASCADE = True
run_clear_ml = True
debug_run = False


def prepare_pramas(def_params, obj_types, dataset_type, stepSize, max_y, model_name, filters):
    params = copy.deepcopy(def_params)

    params['dataset']['obj_types'] = obj_types
    if dataset_type == 'all':
        child_dir_samples_num = {f'dataset_rndshap_Randombc_step_1to{stepSize}_clean': 1000,
                                 f'dataset_rndshap_Randombc_move_body_step_1to{stepSize}_clean': 1000,
                                 f'dataset_rndshap_Randombc_second_body_step_1to{stepSize}_clean': 500}
    elif dataset_type == 'fix':
        child_dir_samples_num = {f'dataset_rndshap_Randombc_step_1to{stepSize}_clean': 1000}
    params['dataset']['child_dir_samples_num'] = child_dir_samples_num
    params['dataset']['max_y'] = max_y

    if model_name in ['AttentionUNet', 'MultiHeadAttentionUNet']:
        params['model']['dilation'] = 2
        params['model']['enc_layers'] = params['model']['layers']
        params['model']['dec_layers'] = params['model']['layers']
        del params['model']['layers']
    params['model']['name'] = model_name
    params['model']['filters'] = filters

    if debug_run:
        params['dataset']['total_samples'] = None
        for k, v in params['dataset']['child_dir_samples_num'].items():
            params['dataset']['child_dir_samples_num'][k] = 10
        params['dataloader']['batch_size'] = 1
        params['train']['epochs'] = 5

    return params


if __name__ == '__main__':
    argv = sys.argv
    cfg1 = argv[1] if len(argv) > 1 else None
    cfg2 = int(argv[2]) if len(argv) > 2 else None

    print(argv)
    print(cfg1)
    print(cfg2)

    out_dir = Path('out_0525_5')

    if CASCADE:
        TORCH_HUB_DIR = '/storage2/pia/python/hub/'
        torch.hub.set_dir(TORCH_HUB_DIR)
        root_dir = '/storage2/pia/python/deepcfd/'
        run_clear_ml = False
    else:
        root_dir = '.'
    root_dir = Path(root_dir)

    if debug_run:
        run_clear_ml = False
        out_dir = Path('out_test')

# region main_config
    # Set config
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Dataset config
    dataset_config = dict(
        datasets_dir=root_dir / 'datasets',
        dataset_name='.',
        max_y=None,
        child_dir_samples_num=None,
        obj_types=['pol'],  # ['pol']  # ['pol', 'spline'],
        total_samples=None,
        train_ratio=0.6,
        val_ratio=0.2
    )

    # Dataloader config
    dataloader_config = dict(
        batch_size=16,
        num_workers=1
    )

    # # Model config
    # model_config = dict(
    #     name=None,
    #     # ['added_fc']  ['bc_in_x']  ['added_fc', 'bc_in_x']
    #     modes=['added_fc'],
    #     add_fc_blocks_every_N=1,
    #     BCinX_channels=2,
    #     in_channels=3,
    #     out_channels=4,
    #     filters=None,
    #     layers=None,
    #     kernel_size=3,
    #     batch_norm=False,
    #     weight_norm=False,
    #     fc_in_channels=6,
    #     fc_out_channels=8,
    #     fc_filters=[8, 16, 32, 64, 32, 16],
    #     device=device,
    # )

    # Model config
    model_config = dict(
        encoder_name='resnet34',
        in_channels=3,
        out_channels=4,
        filters=[256, 128, 64, 32, 16],
        fc_in_channels=6,
        fc_filters=[8, 16, 32, 64, 32, 16],
        fc_out_channels=8,
        add_fc_blocks=[True, False, True, False, True],
        batch_norm=False
        decoder_attention_type=None,
        pretrained=True,
        device=device
    )

    # Optimizer config
    optimizer_config = dict(
        name='AdamW',
        lr=1e-3,
        weight_decay=1e-2
    )
    
    # optimizer_config = dict(
    #     name='RAdam',
    #     lr=1e-4,
    #     betas=(0.9, 0.99),
    #     eps=1e-8,
    #     weight_decay=1e-4
    # )

    # Scheduler config
    epochs = 600
    # gamma_epochs = epochs
    gamma_epochs = epochs
    gamma = pow(1e-2, 1 / gamma_epochs) if gamma_epochs != 0 else 0
    scheduler_config = dict(
        name='StepLR',
        warmup_epochs=0,
        step_size=1,
        gamma=gamma,
        last_epoch=-1
    )
    epochs += scheduler_config.get('warmup_epochs', 0)

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
# endregion main_config

# region exp_configs

    dataset_type_list = [
        'fix',
        'all',
    ]

    obj_types_list = [['pol']]

    stepSize_maxy_list = [
        [64, 625],
        [128, 1250],
        [256, 2500],
    ]

    model_list = [
        # 'UNetExFC',
        # 'EnhancedUNet',
        'SmpUNetExFC',
        # 'AttentionUNet',
        # 'MultiHeadAttentionUNet',
    ]

    # add_fc_blocks_every_N_list = [1]

    # filter_list = [[16, 32, 64, 128, 256, 256, 128, 64, 32]]  # best results for stepSize == 256; min height == 512

    # filters_list = [
    #     [16, 32, 64, 128, 256, 512],
    #     [16, 32, 64, 128, 256],
    #     [16, 32, 64, 128],
    # ]

    # filters_list = [
    #     [8, 16, 32, 64, 128, 256, 256],
    #     [8, 16, 32, 64, 128, 256],
    #     [8, 16, 32, 64, 128],
    # ]

    filters_list = [
        [16, 32, 64, 128, 256, 256, 256],
    ]

    # layers_list = [3]

    if cfg1 != None:
        dataset_type_list = [cfg1]
    if cfg2 != None:
        stepSize_maxy_list = stepSize_maxy_list[cfg2:cfg2+1]

# endregion exp_configs

    exp_list = list()
    iter_list = itertools.product(obj_types_list, dataset_type_list, stepSize_maxy_list, model_list, filters_list)
    for (obj_types, dataset_type, (stepSize, max_y), model_name, filters) in iter_list:
        params = prepare_pramas(def_params, obj_types, dataset_type, stepSize, max_y, model_name, filters)
        tag = f'{dataset_type}_{stepSize}s_{model_name}'

        exp_list.append((params, tag))

    for params, tag in exp_list:
        exp_dir_path = out_dir / tag / get_str_timestamp()
        exp(params, run_clear_ml=run_clear_ml, exp_dir_path=exp_dir_path)
        torch.cuda.empty_cache()
