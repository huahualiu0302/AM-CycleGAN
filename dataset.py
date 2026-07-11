import numpy as np
from torch.utils.data import Dataset, DataLoader
from config import batchsize

N_TRAIN, N_VAL, N_TEST = 250, 100, 150


class LBMdataset(Dataset):
    def __init__(self, xdata, xdata_, ydata, ydata_):
        self.xdata = xdata
        self.xdata_ = xdata_
        self.ydata = ydata
        self.ydata_ = ydata_

    def __len__(self):
        return len(self.ydata)

    def __getitem__(self, i):
        mask = np.array(self.xdata[i], dtype=float)
        mask_ = np.array(self.xdata_[i], dtype=float)
        ydata = self.ydata[i]
        ydata_ = self.ydata_[i]
        return mask, mask_, ydata, ydata_


def _split_indices(n):
    rng = np.random.default_rng()
    idx = rng.permutation(n)
    tr = idx[:N_TRAIN]
    va = idx[N_TRAIN:N_TRAIN + N_VAL]
    te = idx[N_TRAIN + N_VAL:N_TRAIN + N_VAL + N_TEST]
    return tr, va, te


def fit_normalization(y_train):
    avg = np.mean(y_train, axis=0)
    std = np.std(y_train, axis=0)
    z = np.nan_to_num((y_train - avg) / std)
    return {'avg': avg, 'std': std, 'zmin': z.min(), 'zmax': z.max()}


def apply_normalization(y, stats):
    z = np.nan_to_num((y - stats['avg']) / stats['std'])
    return 2 * (z - stats['zmin']) / (stats['zmax'] - stats['zmin']) - 1


def invert_normalization(z, stats):
    y = (z + 1) / 2 * (stats['zmax'] - stats['zmin']) + stats['zmin']
    return y * stats['std'] + stats['avg']


def get_dataloaders():
    ResultA = np.load('Banderagray_size200_xdrive_num500.npy', allow_pickle=True)
    ResultB = np.load('Banderagray_size200_ydrive_num500.npy', allow_pickle=True)

    xA = ResultA[:, 0]
    yA = np.sqrt(ResultA[:, 2] ** 2 + ResultA[:, 3] ** 2)   # 域A 速度模
    xB = ResultB[:, 0]
    yB = np.sqrt(ResultB[:, 2] ** 2 + ResultB[:, 3] ** 2)   # 域B 速度模

    # --- 5:2:3 划分---
    trA, vaA, teA = _split_indices(len(yA))
    trB, vaB, teB = _split_indices(len(yB))

    # --- 归一化 ---
    statsA = fit_normalization(yA[trA])
    statsB = fit_normalization(yB[trB])

    def make_loader(iA, iB, shuffle):
        ds = LBMdataset(
            np.expand_dims(xA[iA], axis=1),
            np.expand_dims(xB[iB], axis=1),
            apply_normalization(yA[iA], statsA),
            apply_normalization(yB[iB], statsB),
        )
        return DataLoader(ds, batch_size=batchsize, shuffle=shuffle)

    loaders = {
        'train': make_loader(trA, trB, shuffle=True),
        'val':   make_loader(vaA, vaB, shuffle=False),
        'test':  make_loader(teA, teB, shuffle=False),
    }
    return loaders, statsA, statsB
