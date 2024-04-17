from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from utils.DataReading import *
from utils.DYGDataSet import ExcelDataset
import warnings
import numpy as np
from utils.Train_test import *
import json
import torch
from torch.utils.data import DataLoader
from pytorch_forecasting import TimeSeriesDataSet

with open('../ConfigFile/config.json', 'r') as f:
    config = json.load(f)
batch_size = config["batch_size"]
lr = config["learning_rate"]
epochs = config["num_epochs"]
datadir = config["data_path"]
t0 = config["time_seqlength"]



criterion = torch.nn.MSELoss()
min_max_scaler = MinMaxScaler(feature_range=(-1,1))
Z_score_scaler = StandardScaler()

full_data = ExcelDataset(N=181*3,t0=t0,file_path=datadir+'hxnn_train_and_test.xlsx',transform=Z_score_scaler)
train_data,test_data = train_test_split(full_data,test_size=0.2,random_state = 2)

train_data_dl = DataLoader(train_data,batch_size=batch_size,shuffle=False)
test_data_dl = DataLoader(test_data,batch_size=batch_size)

training = TimeSeriesDataSet.from_dataloader(
    train_loader,
    time_idx=1,  # 时间步长 ID 在第 1 个维度上
    target_idx=2,  # 目标序列在第 2 个维度上
    group_ids=None,  # 没有组 ID
)