import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

def open_excel(filename):
    """
    打开数据集，进行数据处理
    :param filename:文件名
    :return:特征集数据、标签集数据
    """
    readbook = pd.read_excel(f'{filename}.xlsx', engine='openpyxl',header=0,index_col=[0])
    nplist = readbook.T.to_numpy()
    data = nplist[0:-1].T
    data = np.float64(data)
    target = nplist[-1]
    return data, target

def open_csv(filename):
    """
    打开数据集，进行数据处理
    :param filename:文件名
    :return:特征集数据、标签集数据
    """
    readbook = pd.read_csv(f'{filename}.csv')
    nplist = readbook.T.to_numpy()
    data = nplist[0:-1].T
    data = np.float64(data)
    target = nplist[-1]
    return data, target


def random_number(data_size, key):
    """
   使用shuffle()打乱
    """
    number_set = []
    for i in range(data_size):
        number_set.append(i)

    if key == 1:
        random.shuffle(number_set)

    return number_set


def split_data_set(data_set, target_set, rate, ifsuf):
    """
    说明：分割数据集，默认数据集的rate是测试集
    :param data_set: 数据集
    :param target_set: 标签集
    :param rate: 测试集所占的比率
    :return: 返回训练集数据、测试集数据、训练集标签、测试集标签
    """
    # 计算训练集的数据个数
    train_size = int((1 - rate) * len(data_set))
    # 随机获得数据的下标
    data_index = random_number(len(data_set), ifsuf)
    # 分割数据集（X表示数据，y表示标签），以返回的index为下标
    # 训练集数据
    x_train = data_set[data_index[:train_size]]
    # 测试集数据
    x_test = data_set[data_index[train_size:]]
    # 训练集标签
    y_train = target_set[data_index[:train_size]]
    # 测试集标签
    y_test = target_set[data_index[train_size:]]

    return x_train, x_test, y_train, y_test


def inputtotensor(inputtensor, labeltensor):
    """
    将数据集的输入和标签转为tensor格式
    :param inputtensor: 数据集输入
    :param labeltensor: 数据集标签
    :return: 输入tensor，标签tensor
    """
    inputtensor = np.array(inputtensor)
    inputtensor = torch.FloatTensor(inputtensor)

    labeltensor = np.array(labeltensor)
    labeltensor = labeltensor.astype(float)
    labeltensor = torch.LongTensor(labeltensor)

    return inputtensor, labeltensor

