import time
import datetime
from pathlib import Path
import json
import numpy as np
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


def get_fps(obj_types, input_dir, csv_suffix='.csv.gz'):
    label_fp_list = list(input_dir.glob(f'*_Label{csv_suffix}'))
    label_fp_list.sort()

    sample_fps_list = list()
    for label_fp in label_fp_list:
        split = label_fp.name.split('_')
        idx = split[0]

        for obj_type in obj_types:
            in_fps = [
                input_dir / f'{idx}_{obj_type}_{f}{csv_suffix}' for f in ['Label', 'SDF1', 'SDF2']]
            in_bcs_fps = [input_dir /
                          f'{idx}_{obj_type}_{f}{csv_suffix}' for f in ['BCs']]
            out_fps = [
                input_dir / f'{idx}_{obj_type}_{f}{csv_suffix}' for f in ['UVel', 'VVel', 'Pres', 'Temp']]
            if all([v.exists() for v in in_fps + in_bcs_fps + out_fps]):
                sample_fps_list.append([in_fps, in_bcs_fps, out_fps])

    return sample_fps_list


def prepare_datasets(params, model_modes=[], skip_train=False, skip_val=False, skip_test=False):
    if 'bc_in_x' in model_modes:
        dataset_cls = DatasetCFD_BCinX
    else:
        dataset_cls = DatasetCFD

    input_dir = Path(params.datasets_dir) / params.dataset_name

    sample_fps_list = get_fps(params.obj_types, input_dir)
    samples_num = len(sample_fps_list)
    if params.total_samples is not None:
        samples_num = min(params.total_samples, samples_num)
    train_idx_start = 0
    train_idx_stop = train_idx_start + int(samples_num * params.train_ratio)
    val_idx_start = train_idx_stop
    val_idx_stop = val_idx_start + int(samples_num * params.val_ratio)
    test_idx_start = val_idx_stop
    test_idx_stop = samples_num

    def train_dataset_fn(norm_data): return dataset_cls(
        sample_fps_list[train_idx_start:train_idx_stop], norm_data=norm_data)

    def val_dataset_fn(norm_data): return dataset_cls(
        # transfer normalization data
        sample_fps_list[val_idx_start:val_idx_stop], norm_data=norm_data)

    def test_dataset_fn(norm_data): return dataset_cls(
        # transfer normalization data
        sample_fps_list[test_idx_start:test_idx_stop], norm_data=norm_data)

    return train_dataset_fn, val_dataset_fn, test_dataset_fn


def dump_norm_data(norm_data, fp):
    key_order = ['in_max', 'in_min', 'in_mean',
                 'out_max', 'out_min', 'out_mean']
    
    norm_data_dict = dict()
    for idx, k in enumerate(key_order):
        norm_data_dict[k] = norm_data[idx].tolist()
    with open(fp, 'w') as f:
        json.dump(norm_data_dict, f, indent=4)


def load_norm_data(fp):
    with open(fp, 'r') as f:
        norm_data_dict = json.load(f)
    key_order = ['in_max', 'in_min', 'in_mean',
                 'out_max', 'out_min', 'out_mean']
    norm_data = [np.array(norm_data_dict[k], dtype=np.float32) for k in key_order]
    return norm_data