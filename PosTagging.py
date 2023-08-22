# --基于RNN实现的词性标注--
# Python 3.9.13
# pytorch 1.8.0+cpu
# torchtext 0.9.0
# numpy 1.24.3
import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.legacy import data
from torchtext.legacy import datasets
import numpy as np
import random


SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 128


def prepare_data():
    TEXT = data.Field(lower=True)
    UD_TAGS = data.Field(unk_token=None)
    fields = (("text", TEXT), ("udtags", UD_TAGS), (None, None))
    # 加载UDPOS数据集，只加载UD标签
    train_data, valid_data, test_data = datasets.UDPOS.splits(fields)
    # 构建词汇表，统计并排序词频、为每个单词分配标识符
    TEXT.build_vocab(train_data, min_freq=2, vectors="glove.6B.100d", unk_init=torch.Tensor.normal_)
    UD_TAGS.build_vocab(train_data)
    trainloader, validloader, testloader \
        = data.BucketIterator.splits((train_data, valid_data, test_data),
                                     batch_size=BATCH_SIZE,
                                     device=device)

    return trainloader, validloader, testloader, \
           len(TEXT.vocab), len(UD_TAGS.vocab), \
           TEXT.vocab.stoi[TEXT.pad_token], UD_TAGS.vocab.stoi[UD_TAGS.pad_token]


class LSTM(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, output_size, pad_idx):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bidirectional = True
        self.embedding_size = embedding_size
        self.bi_num = 2 if self.bidirectional else 1
        self.num_layer = 1
        self.output_size = output_size

        self.embedding = nn.Embedding(input_size, embedding_size, padding_idx=pad_idx)

        self.lstm = nn.LSTM(self.embedding_size, self.hidden_size,
                            self.num_layer, bidirectional=self.bidirectional, batch_first=True)
        self.output = nn.Linear(self.hidden_size * self.bi_num, self.output_size)

    def forward(self, x):
        # x=[sen_len,batch_size]
        embedded = self.embedding(x)
        out, (h_n, _) = self.lstm(embedded)
        out = self.output(out)
        # out=[sen_size,batch_size,tag_size]
        return out


class RNN(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, output_size, pad_idx):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, embedding_size, padding_idx=pad_idx)
        self.rnn = nn.RNN(input_size=embedding_size, hidden_size=hidden_size, num_layers=1, batch_first=True)
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        embedded = self.embedding(x)
        x, hidden = self.rnn(embedded)
        return self.output(x)


class GRU(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, output_size, pad_idx):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, embedding_size, padding_idx=pad_idx)
        self.gru = nn.GRU(input_size=embedding_size, hidden_size=hidden_size, batch_first=True)
        self.activation = nn.ReLU()
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # ----gru----
        embedded = self.embedding(x)
        x, hidden = self.gru(embedded)
        # hidden=[n layers, batch_size, hidden_size]
        out = self.output(x)
        return out


# 计算除了'<pad>'标记的准确率
def check_accuracy(preds, y, tag_pad_idx):
    max_preds = preds.argmax(dim=1, keepdim=True)  # get the index of the max probability
    non_pad_elements = (y != tag_pad_idx).nonzero()
    correct = max_preds[non_pad_elements].squeeze(1).eq(y[non_pad_elements])
    return correct.sum() / torch.FloatTensor([y[non_pad_elements].shape[0]])


def evaluate(model, loader, tag_pad_idx):
    model.eval()
    acces = []
    with torch.no_grad():
        for batch in loader:
            x = batch.text
            y = batch.udtags
            x, y = x.to(device), y.to(device)
            y_pred = model(x)
            y_pred = y_pred.view(-1, y_pred.shape[-1])
            y = y.view(-1)
            acc = check_accuracy(y_pred, y, tag_pad_idx)
            acces.append(acc.item())
    avg_acc = sum(acces) / len(acces)
    return avg_acc


def train(epochs, model, optimizer, lossf, trainloader, validloader, tag_pad_idx):
    for epoch in range(epochs):
        model.train()
        acces = []
        losses = []
        for batch in trainloader:
            X = batch.text
            y = batch.udtags
            optimizer.zero_grad()
            # predictions = [sent len, batch size, output dim]
            # tags = [sent len, batch size]
            y_pred = model(X)
            y_pred = y_pred.view(-1, y_pred.shape[-1])
            y = y.view(-1)
            loss = lossf(y_pred, y)
            losses.append(loss.item())
            acc = check_accuracy(y_pred, y, tag_pad_idx)
            acces.append(acc.item())
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 1 == 0:
            avg_loss = sum(losses) / len(losses)
            avg_acc = sum(acces) / len(acces)
            acc_val = evaluate(model, validloader, tag_pad_idx)

            print(f"[Epoch{epoch + 1}] Loss:{avg_loss:.5f} | Train acc:{avg_acc:.3f} | Val acc:{acc_val:.3f}")


if __name__ == "__main__":
    trainloader, testloader, validloader, \
    input_size, output_size, \
    pad_idx, TAG_PAD_IDX = prepare_data()

    # model = RNN(input_size, 100, 128, output_size, pad_idx).to(device)
    # print(model)
    model = torch.load("LSTM_pt_2.pth")
    # optimizer = optim.AdamW(model.parameters(), lr=0.01)
    # optimizer = optim.Adam(model.parameters(), lr=0.01)
    optimizer = optim.RMSprop(model.parameters(), lr=0.001, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0,
                              centered=False)
    lossf = nn.CrossEntropyLoss(ignore_index=TAG_PAD_IDX)
    train(5, model, optimizer, lossf, trainloader, validloader, TAG_PAD_IDX)
    test_acc = evaluate(model, testloader, TAG_PAD_IDX)
    print(f"Test acc: {test_acc:.3f}")
    torch.save(model, "LSTM_pt_2.pth")
