import math
import torch
import torch.nn as nn
import pandas as pd
import csv
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.x2h = nn.Linear(input_size, 4 * hidden_size, bias=bias)
        self.h2h = nn.Linear(hidden_size, 4 * hidden_size, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, x, hidden):
        hx, cx = hidden
        x = x.view(-1, x.size(1))

        gates = self.x2h(x) + self.h2h(hx)
        gates = gates.squeeze()
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        ingate = torch.sigmoid(ingate) # 입력 게이트에 시그모이드 적용
        forgetgate = torch.sigmoid(forgetgate) # 망각 게이트에 시그모이드 적용
        cellgate = torch.tanh(cellgate) # 셀 게이트에 탄젠트 적용
        outgate = torch.sigmoid(outgate) # 출력 게이트에 시그모이드 적용

        cy = torch.mul(cx, forgetgate) + torch.mul(ingate, cellgate)
        hy = torch.mul(outgate, torch.tanh(cy))

        return(hy, cy)


class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, bias=True):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.lstm = nn.ModuleList([LSTMCell(input_dim, hidden_dim, bias) for _ in range(layer_dim)])
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)

        outs = []
        cn = c0[0,:,:]
        hn = h0[0,:,:]

        for seq_model in range(x.size(1)):
            hn, cn = self.lstm[0](x[:, seq_model, :], (h0[0], c0[0]))  # 첫 번째 LSTMCell 사용, hn과 cn 초기화
            for layer in range(1, self.layer_dim):  # 두 번째 이후 LSTMCell 사용
                hn, cn = self.lstm[layer](hn, (h0[layer], cn))  # 이전 층의 hn과 현재 층의 cn을 사용
            outs.append(hn)

        out = outs[-1].squeeze()
        out = self.fc(out)
        return out


def call_df(path):
    df = pd.read_csv(path)
    df = df.sort_values(by='Time')

    test = df[df['IBI'].isnull()]
    test_input = test[['EDA', 'TEMP']].values

    df = df.dropna(subset=['IBI'])
    train_size = int(len(df) * 0.8)

    train = df.iloc[:train_size]
    train_input = train[['EDA', 'TEMP']].values
    train_output = train['IBI'].values


    val = df.iloc[train_size:]
    val_input = val[['EDA', 'TEMP']].values
    val_output = val['IBI'].values

    return train_input, train_output, val_input, val_output, test_input

input_dim = 2
hidden_dim = 32
layer_dim = 3
output_dim = 1
learning_rate = 0.001
num_epochs = 100

path = './norm/Session01/Sess01_script01_User001F.csv'
train_input, train_output, val_input, val_output, test_input = call_df(path)
seq = 1

batch_train = len(train_input)
batch_val = len(val_input)
batch_test = len(test_input)

train_input = train_input.reshape(batch_train, seq, input_dim)
val_input = val_input.reshape(batch_val, seq, input_dim)
test_input = test_input.reshape(batch_test, seq, input_dim)

model = LSTMModel(input_dim, hidden_dim, layer_dim, output_dim)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

if torch.cuda.is_available():
    model.cuda()

for epoch in range(num_epochs):
        model.train()
        outputs = model(train_input)

        # Compute the loss
        loss = criterion(outputs, train_output)

        # Zero gradients, backward pass, and update weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print the progress
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")
            # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(val_input)

            # Compute the loss for validation data
            val_loss = criterion(val_outputs, val_output)

            # Print the progress for validation
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch + 1}/{num_epochs}], Validation Loss: {val_loss.item():.4f}")

pred = model(test_input)
plt.plot(pred)
plt.show()

