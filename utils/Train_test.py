import numpy as np
import torch
import matplotlib.pyplot as plt
import casual_conv_layer
import Dataloader
import json
import torch.nn.functional as F
# In[3]:

from Model.Transformer import TransformerTimeSeries

with open('../ConfigFile/config.json', 'r') as f:
    config = json.load(f)
model_para = config ['model_params']
t0 = config['time_seqlength']
lr = config['optimizer']['lr']
predict_t= config["predict_t"]
model = TransformerTimeSeries().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = torch.nn.MSELoss()

def train_epoch(model, train_dl, t0=t0):
    model.train()
    train_loss = 0
    n = 0
    for step, (x, y, attention_masks) in enumerate(train_dl):
        optimizer.zero_grad()
        output = model(x.cuda(), y.cuda(), attention_masks[0].cuda())
        loss = criterion(output.squeeze()[:, (t0 - 1):(t0 + predict_t - 1)], y.cuda()[:, t0:])  # not missing data
        # loss = criterion(output.squeeze()[:,(t0-1-10):(t0+24-1-10)],y.cuda()[:,(t0-10):]) # missing data
        loss.backward()
        optimizer.step()

        train_loss += (loss.detach().cpu().item() * x.shape[0])
        n += x.shape[0]
    return train_loss / n



def eval_epoch(model, validation_dl, t0=t0):
    model.eval()
    eval_loss = 0
    n = 0
    with torch.no_grad():
        for step, (x, y, attention_masks) in enumerate(validation_dl):
            output = model(x.cuda(), y.cuda(), attention_masks[0].cuda())
            loss = criterion(output.squeeze()[:, (t0 - 1):(t0 + predict_t- 1)], y.cuda()[:, t0:])  # not missing data
            # loss = criterion(output.squeeze()[:,(t0-1-10):(t0+24-1-10)],y.cuda()[:,(t0-10):]) # missing data

            eval_loss += (loss.detach().cpu().item() * x.shape[0])
            n += x.shape[0]

    return eval_loss / n


# In[36]:
def Dp(y_pred, y_true, q):
    return max([q * (y_pred - y_true), (q - 1) * (y_pred - y_true)])

def Rp_num_den(y_preds, y_trues, q):
    numerator = np.sum([Dp(y_pred, y_true, q) for y_pred, y_true in zip(y_preds, y_trues)])
    denominator = np.sum([np.abs(y_true) for y_true in y_trues])
    return numerator, denominator
def mse_loss(output, target):
    return F.mse_loss(output, target)
def mae_loss(output, target):
    return torch.mean(torch.abs(output - target))
def mape_loss(y_true, y_pred):
    """
    计算MAPE指标
    :param y_true: 真实值，形状为(N,)
    :param y_pred: 预测值，形状为(N,)
    :return: MAPE指标，标量
    """
    mape = None  # 定义变量mape并初始化为None
    if len(y_true) > 0:
        mape = torch.mean(torch.abs((y_true - y_pred) / y_true)) * 100.0
    return mape
def TIC_loss(y_true, y_pred):
    """
    计算 Theil 不等系数
    :param y_true: 真实值张量，形状为 [batch_size, seq_len]
    :param y_pred: 预测值张量，形状为 [batch_size, seq_len]
    :return: Theil 不等系数张量，形状为 [batch_size]
    """
    # 计算真实值和预测值的平均值
    y_true_mean = torch.mean(y_true, dim=1, keepdim=True)
    y_pred_mean = torch.mean(y_pred, dim=1, keepdim=True)

    # 计算分子和分母
    numerator = torch.mean((y_true - y_pred) ** 2, dim=1)
    denominator = torch.mean(y_true ** 2, dim=1) + torch.mean(y_pred ** 2, dim=1)

    # 计算 Theil 不等系数
    coefficient = numerator / denominator

    return coefficient

def test_epoch(model, test_dl, t0=t0):
    with torch.no_grad():
        predictions = []
        observations = []
        maelossValue = 0
        mselossValue = 0
        mapelossValue = 0
        TicLossValue = 0
        n = 0
        

        model.eval()
        for step, (x, y, attention_masks) in enumerate(test_dl):
            output = model(x.cuda(), y.cuda(), attention_masks[0].cuda())
            mse = mse_loss(output.squeeze()[:, (t0 - 1):(t0 + predict_t - 1)], y.cuda()[:, t0:])
            mae = mae_loss(output.squeeze()[:, (t0 - 1):(t0 + predict_t - 1)], y.cuda()[:, t0:])
            mape = mape_loss(output.squeeze()[:, (t0 - 1):(t0 + predict_t - 1)], y.cuda()[:, t0:])
            tic = TIC_loss(output.squeeze()[:, (t0 - 1):(t0 + predict_t - 1)], y.cuda()[:, t0:])
            tic = torch.mean(tic,dim=0,keepdim=True)

            maelossValue += (mae.detach().cpu().item() * x.shape[0])
            mselossValue += (mse.detach().cpu().item() * x.shape[0])
            mapelossValue+= (mape.detach().cpu().item()*x.shape[0])
            TicLossValue += (tic.detach().cpu().item()*x.shape[0])
            n += x.shape[0] # n 是batch—size


            for p, o in zip(output.squeeze()[:, (t0 - 1):(t0 + predict_t - 1)].cpu().numpy().tolist(),
                            y.cuda()[:, t0:].cpu().numpy().tolist()):  # not missing data
                # for p,o in zip(output.squeeze()[:,(t0-1-10):(t0+24-1-10)].cpu().numpy().tolist(),y.cuda()[:,(t0-10):].cpu().numpy().tolist()): # missing data

                predictions.append(p) # p 代表预测值output，o代表真实值target
                observations.append(o)


        num = 0
        den = 0
        for y_preds, y_trues in zip(predictions, observations):
            num_i, den_i = Rp_num_den(y_preds, y_trues, .5)
            num += num_i
            den += den_i
        Rp = (2 * num) / den

    return Rp,maelossValue/n,mselossValue/n,mapelossValue/n,TicLossValue/n
