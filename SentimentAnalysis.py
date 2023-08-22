# --基于RNN实现的情感分类--
# Python 3.9.13
# pytorch 1.8.0+cpu
# keras 2.13.1

import torch
import torch.nn as nn
from torch import optim
from keras.preprocessing.sequence import pad_sequences
from keras.datasets import imdb
from torch.utils.data import RandomSampler, DataLoader, TensorDataset, SequentialSampler
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_WORDS = 10000
MAX_LEN = 200
BATCH_SIZE = 256


def prepare_data():
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=MAX_WORDS)
    x_train = pad_sequences(x_train, maxlen=MAX_LEN, padding='post', truncating='post')
    x_test = pad_sequences(x_test, maxlen=MAX_LEN, padding='post', truncating='post')
    # 将数据转为tensorDataset
    train_data = TensorDataset(torch.LongTensor(x_train), torch.LongTensor(y_train))
    test_data = TensorDataset(torch.LongTensor(x_test), torch.LongTensor(y_test))
    # 将数据放入dataloader
    train_sampler = RandomSampler(train_data)
    trainloader = DataLoader(train_data, sampler=train_sampler, batch_size=BATCH_SIZE)
    test_sampler = SequentialSampler(test_data)
    testloader = DataLoader(test_data, sampler=test_sampler, batch_size=BATCH_SIZE)
    return trainloader, testloader


class RNN(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.RNN(input_size=embedding_size, hidden_size=hidden_size, num_layers=2, batch_first=True)
        self.output = nn.Linear(hidden_size, 2)

    def forward(self, x):
        embedded = self.embedding(x)
        x, hidden = self.rnn(embedded)
        x = F.avg_pool2d(x, (x.shape[1], 1)).squeeze()
        return self.output(x)


class LSTM(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bidirectional = False
        self.embedding_size = embedding_size
        self.bi_num = 2 if self.bidirectional else 1
        self.num_layer = 2

        self.embedding = nn.Embedding(input_size, embedding_size)

        self.lstm = nn.LSTM(self.embedding_size, self.hidden_size,
                            self.num_layer, bidirectional=self.bidirectional, batch_first=True)
        self.output = nn.Linear(self.hidden_size * self.bi_num, 2)

    def forward(self, x):
        embedded = self.embedding(x)
        x, (_, _) = self.lstm(embedded)
        # x=[batch_size,sen_size(MAX_LEN),embedding_size]
        out = F.avg_pool2d(x, (x.shape[1], 1)).squeeze()
        # out=[batch_size,embedding_size]
        out = self.output(out)
        return out


def check_accuracy(model, loader):
    num_correct = 0
    num_samples = 0
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

    model.train()
    return (num_correct / num_samples).item()


def train(epochs, model, optimizer, lossf, trainloader, testloader):
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

        if (epoch + 1) % 1 == 0:
            # avg_acc = check_accuracy(model, trainloader)
            avg_test = check_accuracy(model, testloader)
            avg_loss = sum(losses) / len(losses)
            print(f"[Epoch{epoch + 1}] Loss:{avg_loss:.5f} Test acc:{avg_test:.3f}")


if __name__ == "__main__":
    path = './Sentiment/aclImdb'
    trainloader, testloader = prepare_data()

    model = LSTM(MAX_WORDS, 200, 100).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.01)
    lossf = nn.CrossEntropyLoss()
    train(2, model, optimizer, lossf, trainloader, testloader)
    torch.save(model, "LSTM1.pth")
