# %% imports and main parameters
import os
import os.path as osp
import torch

from src.deepcfd_exp import test_exp

CASCADE = False

if CASCADE:
    server_name = 'CASCADE'
    TORCH_HUB_DIR = '/storage0/pia/python/hub/'
    torch.hub.set_dir(TORCH_HUB_DIR)
    root_dir = '/storage0/pia/python/heatnet/'
else:
    server_name = 'seth'
    root_dir = '.'

device = 'cuda' if torch.cuda.is_available() else 'cpu'

exp_dir_path = 'out_CASCADE/out/UNetExFC/20250123_115005'
out_dir_path = osp.join(exp_dir_path, 'results')
total_samples = None
datasets_dir = './datasets'
batch_size = 2

num_samples_to_draw=10
test_exp(exp_dir_path, out_dir_path,
         total_samples=total_samples,
         datasets_dir=datasets_dir,
         batch_size=batch_size,
         num_samples_to_draw=num_samples_to_draw)