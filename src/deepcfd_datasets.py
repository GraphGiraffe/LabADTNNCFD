from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
import numpy as np
import pandas as pd
from torch.utils.data import Dataset as BaseDataset


MAX_Y = None
MAX_WORKERS = 8

def read_csv_to_numpy(fp):
    df = pd.read_csv(fp, low_memory=False, header=None, dtype=np.float32)

    return df.to_numpy().astype(np.float32)


def process(args):
    i_fp, ib_fp, o_fp = args
    in_np = np.stack([read_csv_to_numpy(fp)[..., :MAX_Y] for fp in i_fp]).astype(np.float32)
    in_BCs_np = np.stack([read_csv_to_numpy(fp)[..., 0] for fp in ib_fp]).astype(np.float32)[0]
    out_np = np.stack([read_csv_to_numpy(fp)[..., :MAX_Y] for fp in o_fp]).astype(np.float32)
    
    return in_np, in_BCs_np, out_np
        
class DatasetCFD(BaseDataset):
    def __init__(
            self,
            samples_fps,
            norm_data=None
    ):
        self.samples_fps = samples_fps  # [[in_fps, in_bcs_fps, out_fps], ...]
        self.norm_data = norm_data

        self.in_list, self.in_bcs_list, self.out_list = None, None, None
        self._process_data()

        if self.norm_data is None:
            self.norm_data = self._calc_norm()
        self.in_max, self.in_min, self.in_mean, self.out_max, self.out_min, self.out_mean = self.norm_data
        self.in_mean_norm, self.out_mean_norm = None, None

        self._normalize()

    def _process_data(self):
        self.in_list = list()
        self.in_bcs_list = list()
        self.out_list = list()
        
        res = process_map(process, self.samples_fps, max_workers=MAX_WORKERS, chunksize=1)
        self.in_list = np.stack([r[0] for r in res]).astype(np.float32)
        self.in_bcs_list = np.stack([r[1] for r in res]).astype(np.float32)
        self.out_list = np.stack([r[2] for r in res]).astype(np.float32)
        del res

        # for i_fp, ib_fp, o_fp in tqdm(self.samples_fps):

        #     in_np = np.stack([rea d_csv_to_numpy(fp)[..., :MAX_Y] for fp in i_fp])
        #     in_BCs_np = np.stack([read_csv_to_numpy(fp)[..., 0] for fp in ib_fp])[0]
        #     out_np = np.stack([read_csv_to_numpy(fp)[..., :MAX_Y] for fp in o_fp])

        #     self.in_list.append(in_np)
        #     self.in_bcs_list.append(in_BCs_np)
        #     self.out_list.append(out_np)

        # self.in_list = np.stack(self.in_list)
        # self.in_bcs_list = np.stack(self.in_bcs_list)
        # self.out_list = np.stack(self.out_list)

    def _calc_norm(self):
        in_max = self.in_list.max(axis=(0, 2, 3))
        in_min = self.in_list.min(axis=(0, 2, 3))
        in_mean = self.in_list.mean(axis=(0, 2, 3))
        out_max = self.out_list.max(axis=(0, 2, 3))
        out_min = self.out_list.min(axis=(0, 2, 3))
        out_mean = self.out_list.mean(axis=(0, 2, 3))

        return in_max, in_min, in_mean, out_max, out_min, out_mean

    def _normalize(self):
        for idx in range(self.in_list.shape[1]):
            self.in_list[:, idx, :, :] = (
                self.in_list[:, idx, :, :] - self.in_min[idx]) / (self.in_max[idx] - self.in_min[idx])
        for idx in range(self.out_list.shape[1]):
            self.out_list[:, idx, :, :] = (
                self.out_list[:, idx, :, :] - self.out_min[idx]) / (self.out_max[idx] - self.out_min[idx])

        self.in_mean_norm = self.in_list.mean(axis=(0, 2, 3))
        self.out_mean_norm = self.out_list.mean(axis=(0, 2, 3))

    def __getitem__(self, i):

        return self.in_list[i], self.in_bcs_list[i], self.out_list[i]

    def __len__(self):
        return self.in_list.shape[0]


class DatasetCFD_BCinX(BaseDataset):
    def __init__(
            self,
            in_fps,
            in_bcs_fps,
            out_fps,
            norm_data=None
    ):
        self.in_fps = in_fps
        self.in_bcs_fps = in_bcs_fps
        self.out_fps = out_fps
        self.norm_data = norm_data

        self.in_list, self.in_bcs_list, self.out_list = None, None, None
        self._process_data()

        if self.norm_data is None:
            self.norm_data = self._calc_norm()
        self.in_max, self.in_min, self.in_mean, self.out_max, self.out_min, self.out_mean = self.norm_data
        self.in_mean_norm, self.out_mean_norm = None, None

        self._normalize()
        self._add_bc_in_x()

    def _process_data(self):
        self.in_list = list()
        self.in_bcs_list = list()
        self.out_list = list()
        for i_fp, ib_fp, o_fp in tqdm(zip(self.in_fps, self.in_bcs_fps, self.out_fps)):
            in_np = np.load(i_fp)[:, :, :MAX_Y]
            in_BCs_np = np.load(ib_fp)
            out_np = np.load(o_fp)[:4, :, :MAX_Y]

            self.in_list.append(in_np)
            self.in_bcs_list.append(in_BCs_np[..., 0])
            self.out_list.append(out_np)

        self.in_list = np.stack(self.in_list)
        self.in_bcs_list = np.stack(self.in_bcs_list)
        self.out_list = np.stack(self.out_list)

    def _calc_norm(self):
        in_max = self.in_list.max(axis=(0, 2, 3))
        in_min = self.in_list.min(axis=(0, 2, 3))
        in_mean = self.in_list.mean(axis=(0, 2, 3))
        out_max = self.out_list.max(axis=(0, 2, 3))
        out_min = self.out_list.min(axis=(0, 2, 3))
        out_mean = self.out_list.mean(axis=(0, 2, 3))

        return in_max, in_min, in_mean, out_max, out_min, out_mean

    def _normalize(self):
        for idx in range(self.in_list.shape[1]):
            self.in_list[:, idx, :, :] = (
                self.in_list[:, idx, :, :] - self.in_min[idx]) / (self.in_max[idx] - self.in_min[idx])
        for idx in range(self.out_list.shape[1]):
            self.out_list[:, idx, :, :] = (
                self.out_list[:, idx, :, :] - self.out_min[idx]) / (self.out_max[idx] - self.out_min[idx])

        self.in_mean_norm = self.in_list.mean(axis=(0, 2, 3))
        self.out_mean_norm = self.out_list.mean(axis=(0, 2, 3))

    def _add_bc_in_x(self):
        bcs_len = 2
        zeros = np.zeros(
            (self.in_list.shape[0], bcs_len, self.in_list.shape[2], self.in_list.shape[3]), dtype=np.float32)
        self.in_list = np.hstack([self.in_list, zeros])

        for idx, (x, bcs, y) in enumerate(zip(self.in_list, self.in_bcs_list, self.out_list)):
            u, _, _, t = y

            u_bc = np.zeros(u.shape)
            for k in range(10):
                u_bc[:, k] = u[:, 0]

            t_bc = np.zeros(t.shape, dtype=np.float32)
            t_bc[t == 0] = bcs[1]
            for k in range(10):
                t_bc[:, k] = t[:, 0]

            bcs_list = [u_bc, t_bc]
            self.in_list[idx] = np.vstack([x[:-bcs_len, ...], bcs_list])

        # self.in_bcs_list = np.zeros_like(self.in_bcs_list)

    def __getitem__(self, i):

        return self.in_list[i], self.in_bcs_list[i], self.out_list[i]

    def __len__(self):
        return self.in_list.shape[0]
