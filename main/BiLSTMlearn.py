from datetime import date

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from Model.BiLSTM import BiLSTM_Time_Series
from utils.DataReading import *
from utils.DYGDataSet import ExcelDataset
import warnings
import numpy as np
from utils.Train_test import *
import json

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

# In[8]:


# In[ ]:





# In[ ]:


criterion_LSTM = torch.nn.MSELoss()

# In[ ]:


model = BiLSTM_Time_Series().cuda()

# In[ ]:


optimizer = torch.optim.Adam(model.parameters(), lr=lr)


# In[ ]:


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
# In[ ]:


def train_epoch(model, train_target_dl, t0=t0):
    model.train()
    train_loss = 0
    n = 0

    # 输入到lstm的数据包括x：时间协变量 y：等待预测的序列 _ :mask掩码
    for step, (x, y, _) in enumerate(train_data_dl):
        x = x.cuda()
        y = y.cuda()

        optimizer.zero_grad()
        output = model(x, y)
        loss = criterion(output.squeeze()[:, (t0 - 1):(t0 + predict_t - 1)], y.cuda()[:, t0:])
        loss.backward()
        optimizer.step()

        train_loss += (loss.detach().cpu().item() * x.shape[0])
        n += x.shape[0]
    return train_loss / n


# In[ ]:


def eval_epoch(model, validation_dl, t0=t0):
    model.eval()
    eval_loss = 0
    n = 0
    with torch.no_grad():
        for step, (x, y, _) in enumerate(train_data_dl):
            x = x.cuda()
            y = y.cuda()

            output = model(x, y)
            loss = criterion(output.squeeze()[:, (t0 - 1):(t0 + predict_t - 1)], y.cuda()[:, t0:])

            eval_loss += (loss.detach().cpu().item() * x.shape[0])
            n += x.shape[0]

    return eval_loss / n


# In[ ]:


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
        for step, (x, y, _) in enumerate(train_data_dl):
            x = x.cuda()
            y = y.cuda()

            output = model(x, y)
            mse = mse_loss(output.squeeze()[:, (t0 - 1):(t0 + predict_t - 1)], y.cuda()[:, t0:])
            mae = mae_loss(output.squeeze()[:, (t0 - 1):(t0 + predict_t - 1)], y.cuda()[:, t0:])
            mape = mape_loss(output.squeeze()[:, (t0 - 1):(t0 + predict_t - 1)], y.cuda()[:, t0:])
            tic = TIC_loss(output.squeeze()[:, (t0 - 1):(t0 + predict_t - 1)], y.cuda()[:, t0:])
            tic = torch.mean(tic, dim=0, keepdim=True)

            maelossValue += (mae.detach().cpu().item() * x.shape[0])
            mselossValue += (mse.detach().cpu().item() * x.shape[0])
            mapelossValue += (mape.detach().cpu().item() * x.shape[0])
            TicLossValue += (tic.detach().cpu().item() * x.shape[0])
            n += x.shape[0]  # n 是batch—size


            for p, o in zip(output.squeeze()[:, (t0 - 1):(t0 + predict_t - 1)].cpu().numpy().tolist(),
                            y.cuda()[:, t0:].cpu().numpy().tolist()):
                predictions.append(p)
                observations.append(o)

        num = 0
        den = 0
        for y_preds, y_trues in zip(predictions, observations):
            num_i, den_i = Rp_num_den(y_preds, y_trues, .5)
            num += num_i
            den += den_i
        Rp = (2 * num) / den

    return Rp,maelossValue/n,mselossValue/n,mapelossValue/n,TicLossValue/n


# In[ ]:


train_epoch_loss = []
eval_epoch_loss = []
Rp_best = 10
Rp_loss_list = []
mae_loss_list = []
mape_loss_list = []
mse_loss_list = []
tic_loss_list = []
for e, epoch in enumerate(range(epochs)):
    train_loss = []
    eval_loss = []
    test_loss = []

    l_train = train_epoch(model, train_data_dl)
    train_loss.append(l_train)

    # l_eval = eval_epoch(LSTM,validation_dl)
    # eval_loss.append(l_eval)

    Rp,mae,mse,mape,tic= test_epoch(model, test_data_dl)

    if Rp_best > Rp:
        Rp_best = Rp

    with torch.no_grad():
        print("Epoch {}: Train loss: {} \t Validation loss: {} \t R_p={}\t MAE={}\t MSE={}\t MAPE={}\t TIC={}"
              .format(e, np.mean(train_loss), np.mean(eval_loss), Rp, mae, mse, mape, tic))
        Rp_loss_list.append(Rp)
        mae_loss_list.append(mae)
        mse_loss_list.append(mse)
        mape_loss_list.append(mape)
        tic_loss_list.append(tic)

        train_epoch_loss.append(np.mean(train_loss))
        eval_epoch_loss.append(np.mean(eval_loss))
        test_loss.append(np.mean(test_loss))

# In[ ]:


print("Best Rp={}".format(Rp_best))
mse_min = min (mse_loss_list)
mse_min_index= min(enumerate(mse_loss_list),key=lambda x:x[1])[0]
print("mse_min_index",mse_min_index)

Rp_result = Rp_loss_list[mse_min_index]
mae_result = mae_loss_list[mse_min_index]
mape_result = mape_loss_list[mse_min_index]
tic_result = tic_loss_list[mse_min_index]

predict_t_str = str(predict_t)
t0_str = str(t0)
exercise_name = f"predict_time_{predict_t_str}_t0_{t0_str}"
bilstm_df = pd.DataFrame({"exercise_name":exercise_name,"model":"Bilstm","Epoch":mse_min_index,"Rp_result":Rp_result,"mae_result":mae_result,\
                       "mse_min":mse_min,"mape_result":mape_result,"tic_result":tic_result},index=[0])

today = date.today().strftime("%Y-%m-%d")
file_name = f"{today}_bio_ai_result"
file_name ="../DongYGdata/"+file_name
bilstm_df.to_csv(file_name+".txt",sep="\t",mode="a",header=False)
#bilstm_df.to_excel(file_name+".xlsx",mode="a")

# In[ ]:
n_plots = 5
with torch.no_grad():
    model.eval()
    for step, (x, y, _) in enumerate(test_data_dl):
        x = x.cuda()
        y = y.cuda()

        output = model(x, y)

        if step > n_plots:
            break

        with torch.no_grad():
            plt.figure(figsize=(10, 10))
            plt.plot(x[0].cpu().detach().squeeze().numpy(), y[0].cpu().detach().squeeze().numpy(), 'g--', linewidth=3)
            plt.plot(x[0, t0:].cpu().detach().squeeze().numpy(),
                     output[0, (t0 - 1):(t0 + predict_t - 1)].cpu().detach().squeeze().numpy(), 'b--',
                     linewidth=3)  # not missing data
            # plt.plot(x[0,(t0-10):].cpu().detach().squeeze().numpy(),output[0,(t0-1-10):(t0+24-1-10)].cpu().detach().squeeze().numpy(),'b--',linewidth=3) # missing data
            plt.xlabel("time_step", fontsize=20)
            plt.legend(["$[0,t_0+13)_{obs}$", "$[t_0,t_0+13)_{predicted}$"])
            plt.show()


plt.figure(figsize=(10, 10))
plt.plot(train_epoch_loss)
plt.plot(eval_epoch_loss)
plt.legend(['Train Loss', 'Eval Loss'], fontsize=25)
plt.xlabel("Epoch", fontsize=25)
plt.ylabel("MSE Loss", fontsize=25)
plt.show()

