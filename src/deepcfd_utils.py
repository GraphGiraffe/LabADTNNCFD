import time
import datetime
from pathlib import Path
from src.deepcfd_datasets import (
    DatasetCFD,
    DatasetCFD_BCinX
)


def get_str_timestamp(timestamp=None):
    if timestamp is None:
        timestamp = time.time()
    date_time = datetime.datetime.fromtimestamp(timestamp)
    str_date_time = date_time.strftime("%Y%m%d_%H%M%S")
    return str_date_time


def get_fps(idx_start, idx_stop, obj_types, input_dir):
    in_fps = list()
    in_bcs_fps = list()
    out_fps = list()
    
    for idx in range(idx_start, idx_stop):
        for obj_type in obj_types:
            in_fps.append(input_dir / 'np' / f'{idx:06d}_{obj_type}_In.npy')
            in_bcs_fps.append(input_dir / 'np' / f'{idx:06d}_{obj_type}_In_BCs.npy')
            out_fps.append(input_dir / 'np' / f'{idx:06d}_{obj_type}_Out.npy')

    return in_fps, in_bcs_fps, out_fps


def prepare_datasets(params, model_modes=[]):
    test_ratio = 1 - params.train_ratio - params.val_ratio
    train_idx_start = 0
    train_idx_stop = train_idx_start + int(params.total_samples * params.train_ratio)
    val_idx_start = train_idx_stop
    val_idx_stop = val_idx_start + int(params.total_samples * params.val_ratio)
    test_idx_start = val_idx_stop
    test_idx_stop = test_idx_start + int(params.total_samples * test_ratio)

    if 'bc_in_x' in model_modes:
        dataset_cls = DatasetCFD_BCinX
    else:
        dataset_cls = DatasetCFD

    input_dir = Path(params.datasets_dir) / params.dataset_name
    train_dataset = dataset_cls(*get_fps(train_idx_start, train_idx_stop, params.obj_types, input_dir), norm_data=None)
    val_dataset = dataset_cls(*get_fps(val_idx_start, val_idx_stop, params.obj_types, input_dir), norm_data=train_dataset.norm_data)  # transfer normalization data
    test_dataset = dataset_cls(*get_fps(test_idx_start, test_idx_stop, params.obj_types, input_dir), norm_data=train_dataset.norm_data)  # transfer normalization data

    return train_dataset, val_dataset, test_dataset