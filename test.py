# %% imports and main parameters
import os
import os.path as osp
import torch

from src.deepcfd_exp import test_exp

CASCADE = False

if __name__ == '__main__':

    if CASCADE:
        server_name = 'CASCADE'
        TORCH_HUB_DIR = '/storage0/pia/python/hub/'
        torch.hub.set_dir(TORCH_HUB_DIR)
        root_dir = '/storage0/pia/python/heatnet/'
    else:
        server_name = 'seth'
        root_dir = '.'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # exp_dir_path = 'out_CASCADE/out/UNetExFC/20250123_115005'  # [best] static body
    # exp_dir_path = 'out_CASCADE/out/UNetExFC/full_mb_sb_20250221_112641'  # [receptive field problem in "p"] move body and second body
    # exp_dir_path = 'out_CASCADE/out/UNetExFC/full_mb_sb_20250221_163041'  # [?] move body and second body
    exp_dir_path = 'out_CASCADE/out/EnhancedUNet/full_mb_sb_20250222_174822'  # [best] move body and second body

    out_dir_path = osp.join(exp_dir_path, 'results')
    total_samples = None
    datasets_dir = './datasets'
    batch_size = 1

    test_sample_num = 100
    num_samples_to_draw = 100
    test_exp(exp_dir_path, out_dir_path,
             total_samples=total_samples,
             datasets_dir=datasets_dir,
             batch_size=batch_size,
             num_samples_to_draw=num_samples_to_draw,
             test_sample_num=test_sample_num)

# %%
