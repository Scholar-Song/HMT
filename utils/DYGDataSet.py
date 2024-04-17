import json

import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch import random
from torch.utils.data import Dataset, DataLoader
with open('../ConfigFile/config.json', 'r') as f:
    config = json.load(f)
model_para = config ['model_params']
t0 = config['time_seqlength']
lr = config['optimizer']['lr']
predict_t= config["predict_t"]

class ExcelDataset(Dataset):
    def __init__(self,N,t0,file_path,transform=None):
        # 读取Excel文件
        df = pd.read_excel(file_path,engine='openpyxl',header=0)
        # 将数据转换为PyTorch Tensor
        # df = df.drop(df.index['dl'])
        self.fx = torch.tensor(df.values, dtype=torch.float32).reshape(N,t0+predict_t)
        self.x = torch.cat(N*[torch.arange(0,t0+predict_t).type(torch.float).unsqueeze(0)])
        self.masks = self._generate_square_subsequent_mask(t0)
        self.N = N
        self.t0 = t0
        self.transform = transform
    def __getitem__(self, idx):
        # 获取数据和标签
        # 获取数据和标签
        idx = idx.tolist()

        sample = (self.x[idx, :],
                  self.fx[idx, :],
                  self.masks)

        if self.transform:
            sample_feature_2d = sample[1].reshape(-1, 1)
            scaler = self.transform
            sample_feature_2d_norm = scaler.fit_transform(sample_feature_2d)
            sample_feature_1d_norm = sample_feature_2d_norm.squeeze(1)
            sample_feature_1d_norm = torch.Tensor(sample_feature_1d_norm)
            sample = (sample[0], sample_feature_1d_norm, sample[2])
            # sample = self.transform(sample)

        return sample

    def __len__(self):
        # 返回数据集大小
        return len(self.fx)

    def _generate_square_subsequent_mask(self,t0):
        mask = torch.zeros(t0+predict_t,t0+predict_t)
        for i in range(0,t0):
            mask[i,t0:] = 1
        for i in range(t0,t0+predict_t):
            mask[i,i+1:] = 1
        mask = mask.float().masked_fill(mask == 1, float('-inf'))#.masked_fill(mask == 1, float(0.0))
        return mask
# min_max_scaler = MinMaxScaler()
# test_data = ExcelDataset(N=158*3,t0=8,file_path="../DongYGdata/onefeature_test.xlsx",transform=None)
# test_data_dl = DataLoader(test_data,batch_size=32)
# # print(test_data.x.shape)
# print(test_data.fx.shape)
# print(test_data.masks.shape)