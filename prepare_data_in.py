import shutil
import pandas as pd
import numpy as np
from scipy.ndimage import distance_transform_edt

from multiprocessing import Pool

import matplotlib.pyplot as plt
from matplotlib import cm

from pathlib import Path
from PIL import Image

import ast
import tqdm
from tqdm.contrib.concurrent import process_map


def show_m(m):
    Image.fromarray(((m - m.min())/(m.max() - m.min())
                    * 255).astype(np.uint8)).show()


def read_and_clean(fp):
    df = pd.read_csv(fp, low_memory=False, header=None, dtype=str)
    df = df.replace('Indeterminate', '0')
    df = df.map(ast.literal_eval)

    return df


def calculate_sdf(matrix, step_size=1, body_value=1):
    """
    Calculate the Signed Distance Field (SDF) for a given binary matrix.

    Args:
        matrix (np.ndarray): Binary matrix of size NxM, where 1 represents the geometry body, and 0 is the background.

    Returns:
        np.ndarray: Matrix of size NxM containing the Signed Distance Field.
    """
    # Ensure matrix is binary
    binary_matrix = (matrix == body_value).astype(np.uint8)

    # Distance transform for the outside (background)
    outside_distance = distance_transform_edt(1 - binary_matrix)

    # Distance transform for the inside (geometry)
    inside_distance = distance_transform_edt(binary_matrix)

    # Combine distances: inside distances are negative
    sdf = (outside_distance - inside_distance) * step_size

    return sdf


if __name__ == '__main__':
    root_dir = '.'

    WALL_WIDTH = 10
    INLET_WIDTH = 10
    OUTLET_WIDTH = 10

    FLAW_VALUE = 0
    OBJECT_VALUE = 1
    WALL_VALUE = 2
    INLET_VALUE = 3
    OUTLET_VALUE = 4

    root_dir = Path(root_dir)
    datasets_dir = root_dir / 'datasets'
    # dataset_name = 'dataset_rndshap_Randombc_step_1to256'
    # dataset_name = 'dataset_rndshap_Randombc_move_body_step_1to256'
    dataset_name = 'dataset_rndshap_Randombc_second_body_step_1to256'

    input_dir = datasets_dir / dataset_name
    output_dir = datasets_dir / (dataset_name+'_clean')

    # show = True
    # save = False
    # num_samples = 8

    show = False
    save = True
    num_samples = None

    label_fp_list = list(input_dir.glob('*_Label.csv'))
    label_fp_list.sort()
    label_fp_list = label_fp_list[:num_samples]

    def process(label_fp):
        split = label_fp.name.split('_')
        idx, obj_type = split[0], split[1]

        field_fps = list()
        out_field_fps = list()
        for field_type in ['UVel', 'VVel', 'Pres', 'Temp']:
            field_fps.append(input_dir / f'{idx}_{obj_type}_{field_type}.csv')
            out_field_fps.append(
                output_dir / f'{idx}_{obj_type}_{field_type}.csv.gz')
        bcs_fp = input_dir / f'{idx}_{obj_type}_BCs.csv'
        out_label_fp = output_dir / f'{idx}_{obj_type}_Label.csv.gz'
        out_bcs_fp = output_dir / f'{idx}_{obj_type}_BCs.csv.gz'
        out_sdf1_fp = output_dir / f'{idx}_{obj_type}_SDF1.csv.gz'
        out_sdf2_fp = output_dir / f'{idx}_{obj_type}_SDF2.csv.gz'

        if all([v.exists() for v in field_fps]):
            # if all([v.exists() for v in out_field_fps]) and all(v.exists() for v in [out_label_fp, out_sdf1_fp, out_sdf2_fp]):
            #     return 1

            if save:
                output_dir.mkdir(exist_ok=True)

            for field_fp, out_field_fp in zip(field_fps, out_field_fps):
                try:
                    field = read_and_clean(field_fp)
                    if save:
                        field.astype(np.float32)[::-1].to_csv(out_field_fp, header=False,
                                                              index=False, compression='gzip')
                    if show:
                        show_m(field.to_numpy())
                except Exception as e:
                    print(str(e))
                    return 2
            try:
                label = pd.read_csv(label_fp, low_memory=False,
                                    header=None, dtype=np.int8).to_numpy()
            except Exception as e:
                print(str(e))
                return 3

            label[:WALL_WIDTH, :] = WALL_VALUE
            label[-WALL_WIDTH:, :] = WALL_VALUE
            label[:, :INLET_WIDTH] = INLET_VALUE
            label[:, -OUTLET_WIDTH:] = OUTLET_VALUE
            # label[label == 1] = OBJECT_VALUE
            if save:
                pd.DataFrame(label.astype(np.uint8)[::-1]).to_csv(
                    out_label_fp, header=False, index=False, compression='gzip')
            if show:
                show_m(label)

            try:
                bcs = pd.read_csv(bcs_fp, low_memory=False,
                                  header=None, index_col=None).to_numpy()
            except Exception as e:
                print(str(e))
                return 3
            if save:
                pd.DataFrame(bcs.astype(np.float32)).to_csv(
                    out_bcs_fp, header=False, index=False, compression='gzip')

            sdf1 = calculate_sdf(label, step_size=1/256, body_value=WALL_VALUE)
            if save:
                pd.DataFrame(sdf1.astype(np.float32)[::-1]).to_csv(
                    out_sdf1_fp,  header=False, index=False, compression='gzip')
            if show:
                show_m(sdf1)

            sdf2 = calculate_sdf(label, step_size=1/256,
                                 body_value=OBJECT_VALUE)
            if save:
                pd.DataFrame(sdf2.astype(np.float32)[::-1]).to_csv(
                    out_sdf2_fp,  header=False, index=False, compression='gzip')
            if show:
                show_m(sdf2)

            return 0

    # with Pool(8) as pool:
    #     pool.map(process, label_fp_list[:num_samples])
    #     res = list(tqdm(pool.imap(process, label_fp_list), total=len(label_fp_list), desc="Processing files"))

    # process(label_fp_list[0])
    results = process_map(process, label_fp_list, max_workers=8, chunksize=1)
    for idx, res in enumerate(results):
        if res != 0:
            label_fp = label_fp_list[idx]
            split = label_fp.name.split('_')
            idx, obj_type = split[0], split[1]

            field_fps = list()
            out_field_fps = list()
            for field_type in ['UVel', 'VVel', 'Pres', 'Temp']:
                field_fps.append(input_dir / f'{idx}_{obj_type}_{field_type}.csv')
                out_field_fps.append(
                    output_dir / f'{idx}_{obj_type}_{field_type}.csv.gz')
            bcs_fp = input_dir / f'{idx}_{obj_type}_BCs.csv'

            print(res, out_field_fps + [bcs_fp])
            # for fp in out_field_fps + [bcs_fp]:
            #     shutil.rm(fp)
    print("DONE!")
