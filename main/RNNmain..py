from datetime import date

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
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


class RNN_Time_Series(torch.nn.Module):
    def __init__(self, input_size=2, embedding_size=2, kernel_width=1, hidden_size=512):
        super(RNN_Time_Series, self).__init__()

        self.input_embedding = casual_conv_layer.context_embedding(input_size, embedding_size, kernel_width)

        self.rnn = torch.nn.RNN(embedding_size, hidden_size, batch_first=True)

        self.fc1 = torch.nn.Linear(hidden_size, 1)

    def forward(self, x, y):
        """
        x: the time covariate
        y: the observed target
        """
        # concatenate observed points and time covariate
        # (B,input size + covariate size,sequence length)
        z_obs = torch.cat((y.unsqueeze(1), x.unsqueeze(1)), 1)

        # input_embedding returns shape (B,embedding size,sequence length)
        z_obs_embedding = self.input_embedding(z_obs)

        # permute axes (B,sequence length, embedding size)
        z_obs_embedding = self.input_embedding(z_obs).permute(0, 2, 1)

        # all hidden states from lstm
        # (B,sequence length,num_directions * hidden size)
        rnn_out, _ = self.rnn(z_obs_embedding)

        # input to nn.Linear: (N,*,Hin)
        # output (N,*,Hout)
        return self.fc1(rnn_out)


# In[ ]:


criterion_LSTM = torch.nn.MSELoss()

# In[ ]:


RNN = RNN_Time_Series().cuda()

# In[ ]:


optimizer = torch.optim.Adam(RNN.parameters(), lr=lr)


# In[ ]:


def Dp(y_pred, y_true, q):
    return max([q * (y_pred - y_true), (q - 1) * (y_pred - y_true)])


def Rp_num_den(y_preds, y_trues, q):
    numerator = np.sum([Dp(y_pred, y_true, q) for y_pred, y_true in zip(y_preds, y_trues)])
    denominator = np.sum([np.abs(y_true) for y_true in y_trues])
    return numerator, denominator


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
        for step, (x, y, _) in enumerate(test_dl):
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

    l_train = train_epoch(RNN, train_data_dl)
    train_loss.append(l_train)

    # l_eval = eval_epoch(LSTM,validation_dl)
    # eval_loss.append(l_eval)

    Rp,mae,mse,mape,tic= test_epoch(RNN, test_data_dl)

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
rnn_df = pd.DataFrame({"exercise_name":exercise_name,"model":"RNN","Epoch":mse_min_index,"Rp_result":Rp_result,"mae_result":mae_result,\
                       "mse_min":mse_min,"mape_result":mape_result,"tic_result":tic_result},index=[0])

today = date.today().strftime("%Y-%m-%d")
file_name = f"{today}_bio_ai_result"
file_name ="../DongYGdata/"+file_name
rnn_df.to_csv(file_name+".txt",sep="\t",mode="a",header=False)
#rnn_df.to_excel(file_name+".xlsx",mode="a")

# In[ ]:
n_plots = 5
with torch.no_grad():
    RNN.eval()
    for step, (x, y, _) in enumerate(test_data_dl):
        x = x.cuda()
        y = y.cuda()

        output = RNN(x, y)

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

