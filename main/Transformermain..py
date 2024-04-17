from datetime import date

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import SubsetRandomSampler

from Model.Transformer import TransformerTimeSeries
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
# dataset,target = open_excel("./DongYGdata/testdata")
# train_data,test_data,train_target,test_target = split_data_set(dataset,target,rate=0.2,ifsuf=False)
# train_data,train_target = inputtotensor(train_data,train_target)
min_max_scaler = MinMaxScaler(feature_range=(-1,1))
Z_score_scaler = StandardScaler()
# train_data = ExcelDataset(N=162*3,t0=t0,file_path=datadir+"onefeature_train.xlsx",transform=Z_score_scaler)
# test_data = ExcelDataset(N=158*3,t0=t0,file_path=datadir+'onefeature_test.xlsx',transform=Z_score_scaler)

full_data = ExcelDataset(N=181*3,t0=t0,file_path=datadir+'hxnn_train_and_test.xlsx',transform=Z_score_scaler)
train_data,test_data = train_test_split(full_data,test_size=0.2,random_state = 2)
# 计算训练集和测试集的大小
# train_size = int(0.8 * len(full_data))
# test_size = len(full_data) - train_size
#
# # 创建训练集和测试集的随机采样器
# train_sampler = SubsetRandomSampler(range(train_size))
# test_sampler = SubsetRandomSampler(range(test_size, len(full_data)))
# train_data_dl = DataLoader(full_data,batch_size=batch_size,sampler=train_sampler)
# test_data_dl = DataLoader(full_data,batch_size=batch_size,sampler=test_sampler)

train_data_dl = DataLoader(train_data,batch_size=batch_size,shuffle=False)
test_data_dl = DataLoader(test_data,batch_size=batch_size)



train_epoch_loss = []
eval_epoch_loss = []
Rp_best = 10
Rp_loss_list = []
mae_loss_list = []
mape_loss_list = []
mse_loss_list = []
tic_loss_list = []
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
for e, epoch in enumerate(range(epochs)):
    train_loss = []
    eval_loss = []

    l_t = train_epoch(model, train_data_dl, t0)
    train_loss.append(l_t)

    # l_e = eval_epoch(model, train_data_dl, t0)
    # eval_loss.append(l_e)

    Rp,mae,mse,mape,tic = test_epoch(model, test_data_dl, t0)

    if Rp_best > Rp:
        Rp_best = Rp

    train_epoch_loss.append(np.mean(train_loss))
    eval_epoch_loss.append(np.mean(eval_loss))

    print("Epoch {}: Train loss: {} \t Validation loss: {} \t R_p={}\t MAE={}\t MSE={}\t MAPE={}\t TIC={}"
          .format(e,np.mean(train_loss),np.mean(eval_loss), Rp,mae,mse,mape,tic))
    Rp_loss_list.append(Rp)
    mae_loss_list.append(mae)
    mse_loss_list.append(mse)
    mape_loss_list.append(mape)
    tic_loss_list.append(tic)

# In[ ]:


print("Rp best={}".format(Rp_best))
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
trans_df = pd.DataFrame({"exercise_name":exercise_name,"model":"ConvTrans","Epoch":mse_min_index,"Rp_result":Rp_result,"mae_result":mae_result,\
                       "mse_min":mse_min,"mape_result":mape_result,"tic_result":tic_result},index=[0])

today = date.today().strftime("%Y-%m-%d")
file_name = f"{today}_bio_ai_result"
file_name ="../DongYGdata/"+file_name
trans_df.to_csv(file_name+".txt",sep="\t",mode="a",header=False)
#trans_df.to_excel(file_name+".xlsx",mode="a")

plt.figure(figsize=(10, 10))
plt.plot(train_epoch_loss)
plt.plot(eval_epoch_loss)
plt.legend(['Train Loss', 'Eval Loss'], fontsize=25)
plt.xlabel("Epoch", fontsize=25)
plt.ylabel("MSE Loss", fontsize=25)
plt.show()

# In[39]:


n_plots = 5
with torch.no_grad():
    model.eval()
    for step, (x, y, attention_masks) in enumerate(test_data_dl):
        output = model(x.cuda(), y.cuda(), attention_masks[0].cuda())

        if step > n_plots:
            break

        with torch.no_grad():
            plt.figure(figsize=(10, 10))
            plt.plot(x[0].cpu().detach().squeeze().numpy(), y[0].cpu().detach().squeeze().numpy(), 'g--', linewidth=3)
            plt.plot(x[0, t0:].cpu().detach().squeeze().numpy(),
                     output[0, (t0 - 1):(t0 + predict_t - 1)].cpu().detach().squeeze().numpy(), 'b--',
                     linewidth=3)  # not missing data
            # plt.plot(x[0,(t0-10):].cpu().detach().squeeze().numpy(),output[0,(t0-1-10):(t0+24-1-10)].cpu().detach().squeeze().numpy(),'b--',linewidth=3) # missing data
            plt.xlabel("x", fontsize=20)
            plt.legend(["$[0,t_0+13)_{obs}$", "$[t_0,t_0+13)_{predicted}$"])
            plt.show()


# In[40]:


def get_attn(model, x, y, attention_masks):
    model.eval()
    with torch.no_grad():
        x = x.cuda()
        y = y.cuda()
        attention_masks = attention_masks.cuda()
        z = torch.cat((y.unsqueeze(1), x.unsqueeze(1)), 1)
        z_embedding = model.input_embedding(z).permute(2, 0, 1)
        positional_embeddings = model.positional_embedding(x.type(torch.long)).permute(1, 0, 2)
        input_embedding = z_embedding + positional_embeddings

        attn_layer_i = []
        for layer in model.transformer_decoder.layers:
            attn_layer_i.append(
                layer.self_attn(input_embedding, input_embedding, input_embedding, attn_mask=attention_masks)[
                    -1].squeeze().cpu().detach().numpy())
            input_embedding = layer.forward(input_embedding, attention_masks)

        return attn_layer_i


# In[41]:


idx_example = 5
attn_layers = get_attn(model, test_data[idx_example][0].unsqueeze(0), test_data[idx_example][1].unsqueeze(0),
                       test_data[idx_example][2])

# In[42]:


plt.figure(figsize=(10, 5))
plt.plot(test_data[idx_example][0].numpy(), train_data[10][1].numpy())
plt.plot([t0 + 4 - 1, t0 + predict_t - 1], [20, 120], 'g--')  # not missing data
# plt.plot([t0+24-1,t0+24-1],[20,120],'g--') # missing data
plt.figure(figsize=(10, 10))
plt.plot(attn_layers[0][t0 + predict_t - 1])  # not missing data
plt.plot(attn_layers[1][t0 + predict_t - 1])  # not missing data
plt.plot(attn_layers[2][t0 + predict_t - 1])  # not missing data

# plt.plot(train_dataset[idx_example][0].numpy(),attn_layers[0][119-10]) # missing data
# plt.plot(train_dataset[idx_example][0].numpy(),attn_layers[1][119-10]) # missing data
# plt.plot(train_dataset[idx_example][0].numpy(),attn_layers[2][119-10]) # missing data


plt.legend(["attn score in layer 1", "attn score in layer 2", "attn score in layer 3"])
plt.title("Attn for t = 26")  # not missing data

plt.show()
