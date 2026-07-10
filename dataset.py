import numpy as np
from torch.utils.data import Dataset, DataLoader
from config import batchsize

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


def get_dataloader():
    # 域A：500张切片，沿x方向的LBM结果
    ResultA = np.load('Banderagray_size200_xdrive_num500.npy', allow_pickle=True)
    # 域B：另外500张不同切片，沿y方向的LBM结果
    ResultB = np.load('Banderagray_size200_ydrive_num500.npy', allow_pickle=True)

    xtrain = ResultA[:, 0]
    ytrain = np.sqrt(ResultA[:, 2] ** 2 + ResultA[:, 3] ** 2)  # 域A速度模
    xtrain_ = ResultB[:, 0]
    ytrain_ = np.sqrt(ResultB[:, 2] ** 2 + ResultB[:, 3] ** 2)  # 域B速度模

    xdata_norm_l2 = np.expand_dims(xtrain, axis=1)
    xdata_norm_l2_ = np.expand_dims(xtrain_, axis=1)

    avg = np.mean(ytrain, axis=0)
    std = np.std(ytrain, axis=0)
    ydata_norm_l2 = (ytrain - avg) / std
    ydata_norm_l2 = np.nan_to_num(ydata_norm_l2)
    # 归一化到 [-1, 1]
    ydata_norm_l2 = 2 * (ydata_norm_l2 - ydata_norm_l2.min()) / (ydata_norm_l2.max() - ydata_norm_l2.min()) - 1

    # 标准化 ytrain 数据
    avg_ = np.mean(ytrain_, axis=0)
    std_ = np.std(ytrain_, axis=0)
    ydata_norm_l2_ = (ytrain_ - avg_) / std_
    ydata_norm_l2_ = np.nan_to_num(ydata_norm_l2_)
    # 将标签数据归一化到 -1 到 1 之间
    ydata_norm_l2_ = 2 * (ydata_norm_l2_ - np.min(ydata_norm_l2_)) / (
                np.max(ydata_norm_l2_) - np.min(ydata_norm_l2_)) - 1

    inputdata = LBMdataset(xdata_norm_l2, xdata_norm_l2_, ydata_norm_l2, ydata_norm_l2_)
    dataloader = DataLoader(inputdata, batch_size=batchsize, shuffle=True)

    return dataloader