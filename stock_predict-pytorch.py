import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torch import nn
import torch.utils.data as Data

f=open('dataset_3.csv')
df=pd.read_csv(f)     #读入股票数据
data=np.array(df['high'])   #获取最高价序列
data=data[::-1]      #反转，使数据按照日期先后顺序排列
#以折线图展示dataQYYBAC9D3J-eyJsaWNlbnNlSWQiOiJRWVlCQUM5RDNKIiwibGljZW5zZWVOYW1lIjoi6LaF57qnIOeoi+W6j+WRmCIsImFzc2lnbmVlTmFtZSI6IiIsImFzc2lnbmVlRW1haWwiOiIiLCJsaWNlbnNlUmVzdHJpY3Rpb24iOiIiLCJjaGVja0NvbmN1cnJlbnRVc2UiOmZhbHNlLCJwcm9kdWN0cyI6W3siY29kZSI6IklJIiwiZmFsbGJhY2tEYXRlIjoiMjAyMC0wMS0wNCIsInBhaWRVcFRvIjoiMjAyMS0wMS0wMyJ9LHsiY29kZSI6IkFDIiwiZmFsbGJhY2tEYXRlIjoiMjAyMC0wMS0wNCIsInBhaWRVcFRvIjoiMjAyMS0wMS0wMyJ9LHsiY29kZSI6IkRQTiIsImZhbGxiYWNrRGF0ZSI6IjIwMjAtMDEtMDQiLCJwYWlkVXBUbyI6IjIwMjEtMDEtMDMifSx7ImNvZGUiOiJQUyIsImZhbGxiYWNrRGF0ZSI6IjIwMjAtMDEtMDQiLCJwYWlkVXBUbyI6IjIwMjEtMDEtMDMifSx7ImNvZGUiOiJHTyIsImZhbGxiYWNrRGF0ZSI6IjIwMjAtMDEtMDQiLCJwYWlkVXBUbyI6IjIwMjEtMDEtMDMifSx7ImNvZGUiOiJETSIsImZhbGxiYWNrRGF0ZSI6IjIwMjAtMDEtMDQiLCJwYWlkVXBUbyI6IjIwMjEtMDEtMDMifSx7ImNvZGUiOiJDTCIsImZhbGxiYWNrRGF0ZSI6IjIwMjAtMDEtMDQiLCJwYWlkVXBUbyI6IjIwMjEtMDEtMDMifSx7ImNvZGUiOiJSUzAiLCJmYWxsYmFja0RhdGUiOiIyMDIwLTAxLTA0IiwicGFpZFVwVG8iOiIyMDIxLTAxLTAzIn0seyJjb2RlIjoiUkMiLCJmYWxsYmFja0RhdGUiOiIyMDIwLTAxLTA0IiwicGFpZFVwVG8iOiIyMDIxLTAxLTAzIn0seyJjb2RlIjoiUkQiLCJmYWxsYmFja0RhdGUiOiIyMDIwLTAxLTA0IiwicGFpZFVwVG8iOiIyMDIxLTAxLTAzIn0seyJjb2RlIjoiUEMiLCJmYWxsYmFja0RhdGUiOiIyMDIwLTAxLTA0IiwicGFpZFVwVG8iOiIyMDIxLTAxLTAzIn0seyJjb2RlIjoiUk0iLCJmYWxsYmFja0RhdGUiOiIyMDIwLTAxLTA0IiwicGFpZFVwVG8iOiIyMDIxLTAxLTAzIn0seyJjb2RlIjoiV1MiLCJmYWxsYmFja0RhdGUiOiIyMDIwLTAxLTA0IiwicGFpZFVwVG8iOiIyMDIxLTAxLTAzIn0seyJjb2RlIjoiREIiLCJmYWxsYmFja0RhdGUiOiIyMDIwLTAxLTA0IiwicGFpZFVwVG8iOiIyMDIxLTAxLTAzIn0seyJjb2RlIjoiREMiLCJmYWxsYmFja0RhdGUiOiIyMDIwLTAxLTA0IiwicGFpZFVwVG8iOiIyMDIxLTAxLTAzIn0seyJjb2RlIjoiUlNVIiwiZmFsbGJhY2tEYXRlIjoiMjAyMC0wMS0wNCIsInBhaWRVcFRvIjoiMjAyMS0wMS0wMyJ9XSwiaGFzaCI6IjE2MDgwOTA5LzAiLCJncmFjZVBlcmlvZERheXMiOjcsImF1dG9Qcm9sb25nYXRlZCI6ZmFsc2UsImlzQXV0b1Byb2xvbmdhdGVkIjpmYWxzZX0=-I7c5mu4hUCMxcldrwZEJMaT+qkrzrF1bjJi0i5QHcrRxk2LO0jqzUe2fBOUR4L+x+7n6kCwAoBBODm9wXst8dWLXdq179EtjU3rfJENr1wXGgtef//FNow+Id5iRufJ4W+p+3s5959GSFibl35YtbELELuCUH2IbCRly0PUBjitgA0r2y+9jV5YD/dmrd/p4C87MccC74NxtQfRdeUEGx87vnhsqTFH/sP4C2VljSo/F/Ft9JqsSlGfwSKjzU8BreYt1QleosdMnMK7a+fkfxh7n5zg4DskdVlNbfe6jvYgMVE16DMXd6F1Zhwq+lrmewJA2jPToc+H5304rcJfa9w==-MIIElTCCAn2gAwIBAgIBCTANBgkqhkiG9w0BAQsFADAYMRYwFAYDVQQDDA1KZXRQcm9maWxlIENBMB4XDTE4MTEwMTEyMjk0NloXDTIwMTEwMjEyMjk0NlowaDELMAkGA1UEBhMCQ1oxDjAMBgNVBAgMBU51c2xlMQ8wDQYDVQQHDAZQcmFndWUxGTAXBgNVBAoMEEpldEJyYWlucyBzLnIuby4xHTAbBgNVBAMMFHByb2QzeS1mcm9tLTIwMTgxMTAxMIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEAxcQkq+zdxlR2mmRYBPzGbUNdMN6OaXiXzxIWtMEkrJMO/5oUfQJbLLuMSMK0QHFmaI37WShyxZcfRCidwXjot4zmNBKnlyHodDij/78TmVqFl8nOeD5+07B8VEaIu7c3E1N+e1doC6wht4I4+IEmtsPAdoaj5WCQVQbrI8KeT8M9VcBIWX7fD0fhexfg3ZRt0xqwMcXGNp3DdJHiO0rCdU+Itv7EmtnSVq9jBG1usMSFvMowR25mju2JcPFp1+I4ZI+FqgR8gyG8oiNDyNEoAbsR3lOpI7grUYSvkB/xVy/VoklPCK2h0f0GJxFjnye8NT1PAywoyl7RmiAVRE/EKwIDAQABo4GZMIGWMAkGA1UdEwQCMAAwHQYDVR0OBBYEFGEpG9oZGcfLMGNBkY7SgHiMGgTcMEgGA1UdIwRBMD+AFKOetkhnQhI2Qb1t4Lm0oFKLl/GzoRykGjAYMRYwFAYDVQQDDA1KZXRQcm9maWxlIENBggkA0myxg7KDeeEwEwYDVR0lBAwwCgYIKwYBBQUHAwEwCwYDVR0PBAQDAgWgMA0GCSqGSIb3DQEBCwUAA4ICAQAF8uc+YJOHHwOFcPzmbjcxNDuGoOUIP+2h1R75Lecswb7ru2LWWSUMtXVKQzChLNPn/72W0k+oI056tgiwuG7M49LXp4zQVlQnFmWU1wwGvVhq5R63Rpjx1zjGUhcXgayu7+9zMUW596Lbomsg8qVve6euqsrFicYkIIuUu4zYPndJwfe0YkS5nY72SHnNdbPhEnN8wcB2Kz+OIG0lih3yz5EqFhld03bGp222ZQCIghCTVL6QBNadGsiN/lWLl4JdR3lJkZzlpFdiHijoVRdWeSWqM4y0t23c92HXKrgppoSV18XMxrWVdoSM3nuMHwxGhFyde05OdDtLpCv+jlWf5REAHHA201pAU6bJSZINyHDUTB+Beo28rRXSwSh3OUIvYwKNVeoBY+KwOJ7WnuTCUq1meE6GkKc4D/cXmgpOyW/1SmBz3XjVIi/zprZ0zf3qH5mkphtg6ksjKgKjmx1cXfZAAX6wcDBNaCL+Ortep1Dh8xDUbqbBVNBL4jbiL3i3xsfNiyJgaZ5sX7i8tmStEpLbPwvHcByuf59qJhV/bZOl8KqJBETCDJcY6O2aqhTUy+9x93ThKs1GKrRPePrWPluud7ttlgtRveit/pcBrnQcXOl1rHq7ByB8CFAxNotRUYL9IF5n3wJOgkPojMy6jetQA5Ogc8Sm7RG6vg1yow==

plt.figure()
plt.plot(data)
#plt.show()

time_step=60      #时间步
num_layers=1      #层数
hidden_size=60       #hidden layer units
batch_size=20     #每一批次训练多少个样例
input_size=1      #输入层维度
output_size=1     #输出层维度
lr=0.0006         #学习率
train_x,train_y=[],[]   #训练集

normalize_data=(data-np.mean(data))/np.std(data)  #标准化
mean_data=np.mean(data)
std_data=np.std(data)
#print('mean_data=',mean_data,'std_data=',std_data)

normalize_data=normalize_data[:,np.newaxis]       #增加维度x


for i in range(len(normalize_data)-time_step-1):
    x=normalize_data[i:i+time_step]
    y=normalize_data[i+1:i+time_step+1]
    train_x.append(x.tolist())
    train_y.append(y.tolist())

train_x=torch.Tensor(train_x)
train_y=torch.Tensor(train_y)
train_x=Variable(train_x)
var_x = Variable(train_x)
var_y = Variable(train_y)
"""torch_dataset=Data.TensorDataset(train_x,train_y)
loader=Data.DataLoader(
    dataset=torch_dataset,
    batch_size=batch_size,
    shuffle=False,
)"""

#————————搭建模型——————————
class lstm_reg(nn.Module):
    def __init__(self,input_size,hidden_size,output_size,num_layers):
        super(lstm_reg,self).__init__()
        self.rnn=nn.LSTM(input_size,hidden_size,num_layers)   #隐藏层
        self.reg=nn.Linear(hidden_size,output_size)    #输出层

    def forward(self,x):
        x, (h_n, h_c) = self.rnn(x, None)  # None 表示 hidden state 会用全0的 state
        x = self.reg(x)
        return x


net1 = lstm_reg(input_size,hidden_size,output_size,num_layers)
#———————————训练模型——————————
def train_lstm(distance_test):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(net1.parameters(),lr=lr)
    for i in range(2):
        start = 0
        end = start + batch_size
        while (end < len(train_x) - distance_test):
            out = net1(var_x[start:end]).view(-1)
            p=var_y[start:end].view(-1)
            loss = criterion(out ,p)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            start += batch_size
            end = start + batch_size  # 每次用一批，即batch_size进行训练
        real_distance_test = end - batch_size  # 循环跳出来之前，end已经加了一个batch_size，这里要减回去
        print(i)
    return real_distance_test


#————————————预测模型————————————
def prediction(real_distance_test, offset, predict_number):
        #net1 = lstm_reg(input_size, hidden_size, output_size, num_layers)
        end2 = real_distance_test - offset  # 扣除offset的起始点位置
        prev_seq = train_x[end2]  # 预测起始点的输入
        label = []  # 测试标签
        pre_predict = []  # 记录用训练数据预测的结果，数据无意义，仅用于隐含层记忆历史数据
        predict = []  # 有效的预测结果
            # 得到之后100个预测结果
        for i in range(offset + predict_number):
            prev_seq=torch.Tensor(prev_seq)
            prev_seq=torch.unsqueeze(prev_seq,1)
            next_seq = net1(prev_seq)
            label.append(train_y[end2 + i][-1])
            if i < offset:  # 用训练集输入用于预测，预测结果无意义
                pre_predict.append(next_seq[-1])
                prev_seq = train_x[end2 + i + 1]
                    # print('old=',prev_seq,'\n')
            else:  # 用上步预测结果作为当前步的输入，进行连续有效预测
                predict.append(next_seq[-1].detach().numpy())
                    # 每次得到最后一个时间步的预测结果，与之前的数据加在一起，形成新的测试样本
                prev_seq = torch.squeeze(prev_seq,dim=2)
                prev_seq=torch.cat((prev_seq[1:], next_seq[-1]))
                prev_seq = prev_seq.tolist()
                    # print('new=',prev_seq,'\n')
        label=np.array(label).reshape(-1,1)
        #label=label[0]
        predict=np.array(predict)
        predict=predict.reshape(-1,1)
        label=label*std_data+mean_data
        predict=predict*std_data+mean_data
        print('label=\n',label,'\n predict=\n',predict)

        np.savez('./index.npz', label, pre_predict, predict)  # 保存数据，用于画图。可运行draw.py作图


if __name__ == "__main__":
    distance_test = 450  # 训练数据的截止点离最新数据的距离
    predict_number = 10  # 连续预测天数
    # 已经训练过的输入数据作为预测时的输入。由于LSTM隐含层和历史输入数据相关，
    # 当用于预测时，需要用一段训练数据作为预测输入，但该段数据
    # 的预测结果没有意义，仅仅是让模型隐含层记忆历史数据
    offset = 0
    # 训练数据的截止点离最近数据的真实距离，因为训练是以batch_size为单位进行训练的。
    # 因此real_distance_test大于等于distance_test.
    real_distance_test = train_lstm(distance_test)
    prediction(real_distance_test, offset, predict_number)


    
    D = np.load('./index.npz')
    label = D['arr_0']
    pre_predict = D['arr_1']
    predict = D['arr_2']

    plt.figure()
    plt.plot(list(range(len(label))), label, color='b')
    plt.plot(list(range(len(pre_predict))), pre_predict, color='r')
    plt.plot(list(range(len(pre_predict), len(pre_predict) + len(predict))), predict, color='y')
    plt.show()
    