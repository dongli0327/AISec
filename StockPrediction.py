# --基于RNN实现的股市预测--
# Python 3.9.13
# pytorch 1.8.0+cpu
# matplotlib 3.3.0
# sklearn 1.2.2
# pandas 2.0.3

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import torch.nn as nn
import matplotlib.pyplot as plt
from torch import optim
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 数据处理
def prepare_data(dir, N):
    aapl = pd.read_csv(dir)
    X = []
    y = []
    date = []
    for i in range(30, len(aapl) - 1):
        y.append(aapl.iloc[i + 1, 6])
        date.append(aapl.iloc[i, 0])

    aapl['Close'] = aapl['Close'].apply(lambda x: x[1:])
    aapl['Open'] = aapl['Open'].apply(lambda x: x[1:])
    aapl['High'] = aapl['Low'].apply(lambda x: x[1:])
    aapl['Low'] = aapl['Low'].apply(lambda x: x[1:])

    mms = MinMaxScaler()
    # 训练集归一化
    mms.fit(aapl.iloc[:N][['Close', 'Volume', 'Open', 'High', 'Low']])
    aapl[['Close', 'Volume', 'Open', 'High', 'Low']] = mms.transform(aapl[['Close', 'Volume', 'Open', 'High', 'Low']])
    # print(aapl.head())
    for i in range(31, len(aapl)):
        list = aapl.iloc[i - 30:i, 1:7].values.tolist()
        X.append(list)
    X = torch.tensor(X, dtype=torch.float)
    y = torch.tensor(y, dtype=torch.float).view(-1, 1)
    X_train, X_test = X[:N], X
    y_train, y_test = y[:N], y
    trainloader = DataLoader(TensorDataset(X_train, y_train), 64, True)
    return trainloader, date, X_test, y_test


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=1, batch_first=True)
        self.rnn2 = nn.RNN(input_size=hidden_size, hidden_size=hidden_size, num_layers=1, batch_first=True)
        self.output = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # ----rnn----
        x, _ = self.rnn(x)
        # x=[batch_size,days,hidden_size]
        # print(x.size())
        out = F.avg_pool2d(x, (x.shape[1], 1)).squeeze()
        out = self.rnn2(out)
        return self.output(out)


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=1, batch_first=True)
        self.output = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # ----lstm----
        x, (_, _) = self.lstm(x, None)
        # x=[batch_size,days,hidden_size]
        out = F.avg_pool2d(x, (x.shape[1], 1)).squeeze()
        return self.output(out)


class GRU(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.activation = nn.ReLU()
        self.output = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # ----gru----
        _, hidden = self.gru(x)
        # hidden=[n layers, batch_size, hidden_size]
        x = F.avg_pool2d(hidden, (hidden.shape[0], 1)).squeeze()
        hidden = self.activation(x)
        out = self.output(hidden)
        return out


def train(epochs, model, optimizer, lossf, trainloader):
    for epoch in range(epochs):
        model.train()
        losses = []
        for X, y in trainloader:
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            loss = lossf(y_pred, y)
            losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if (epoch + 1) % 100 == 0:
            avg_loss = sum(losses) / len(losses)
            print(f"[Epoch{epoch + 1}] Loss:{avg_loss:.5f}")


def predict_and_plot(model, X_test, y_test, path, date):
    with torch.no_grad():
        pred = model(X_test)
    pred = pred.tolist()
    y_pred = []
    for i in range(len(pred)):
        y_pred.append(pred[i])
    # print(y_pred)
    # x = list(range(1, len(X_test) + 1))

    plt.figure(dpi=128, figsize=(20, 15))
    plt.plot(date, y_test, "ob:", label='y-test')
    plt.plot(date, y_pred, "rs-", label='y-pred')
    # 显示图例
    plt.legend()
    plt.xlabel('date')
    plt.ylabel('p_change')
    plt.xticks(range(0, len(date), 20))

    plt.xticks(rotation=45)
    # 保存图片到本地
    plt.savefig(path)
    # 显示图片
    plt.show()


if __name__ == "__main__":
    dir = './Stock/AAPL_300.csv'
    N = -50
    trainloader, date, X_test, y_test = prepare_data(dir, N)

    model = RNN(6, 32)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    # optimizer = optim.RMSprop(model.parameters(), lr=0.001, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0,
    #                         centered=False)
    lossf = nn.MSELoss()
    train(500, model, optimizer, lossf, trainloader)
    path = 'Stock/gru_32.png'
    predict_and_plot(model, X_test, y_test, path, date)
