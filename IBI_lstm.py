import torch
import torch.nn as nn
import pandas as pd
import os
import matplotlib.pyplot as plt


class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, bias=True):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h_0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(x.device)
        c_0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.lstm(x, (h_0, c_0))
        out = self.fc(out)
        return out


def call_df(path):
    df = pd.read_csv(path)
    df = df.sort_values(by='Time')

    all_input = df[['EDA', 'TEMP']].values

    df = df.dropna(subset=['IBI'])
    train_size = int(len(df) * 0.8)

    train = df.iloc[:train_size]
    train_input = train[['EDA', 'TEMP']].values
    train_output = train['IBI'].values

    val = df.iloc[train_size:]
    val_input = val[['EDA', 'TEMP']].values
    val_output = val['IBI'].values

    return train_input, train_output, val_input, val_output, all_input


def getfile(path):
    file_list = []
    for name in os.listdir(path):
        file_list.append(name)
    return file_list


def gettime(path):
    df = pd.read_csv(path)
    pred_size = len(df[df['IBI'].isnull()])
    all_size = len(df)
    dftime = df['Time'].iloc[all_size - pred_size : all_size]
    len_pred = all_size - pred_size

    return dftime, len_pred


def train_model(path, pred_path):
    train_input, train_output, val_input, val_output, all_input = call_df(path)

    train_input = torch.tensor(train_input, dtype=torch.float)
    train_output = torch.tensor(train_output, dtype=torch.float)
    val_input = torch.tensor(val_input, dtype=torch.float)
    val_output = torch.tensor(val_output, dtype=torch.float)
    all_input = torch.tensor(all_input, dtype=torch.float)

    input_dim = 2
    hidden_dim = 16
    layer_dim = 1
    output_dim = 1
    learning_rate = 0.001
    num_epochs = 100

    batch_size = 1

    seq_train = len(train_input)
    seq_val = len(val_input)
    seq_all = len(all_input)

    train_input = train_input.reshape(batch_size, seq_train, input_dim)
    val_input = val_input.reshape(batch_size, seq_val, input_dim)
    all_input = all_input.reshape(batch_size, seq_all, input_dim)

    pred_time, pred_len = gettime(path)

    model = LSTMModel(input_dim, hidden_dim, layer_dim, output_dim)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


    for epoch in range(num_epochs):
        model.train()
        output = model(train_input)


        # Compute the loss
        loss = criterion(output, train_output.view(-1, 1))

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
            val_res = model(val_input)

            # Compute the loss for validation data
            val_loss = criterion(val_res, val_output.view(-1, 1))

            # Print the progress for validation
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch + 1}/{num_epochs}], Validation Loss: {val_loss.item():.4f}")

    model.eval()
    print(len(all_input))
    with torch.no_grad():
        pred = model(all_input)
    pred_values = pred[0, :, 0].detach().numpy()
    df = pd.read_csv(path)
    df_pred = pd.DataFrame({'Time': pred_time.values, 'IBI': pred_values[pred_len:]})
    final_df = pd.merge(df, df_pred, on='Time', how='left')
    final_df['IBI_x'] = final_df['IBI_x'].combine_first(final_df['IBI_y'])
    final_df.drop(columns=['IBI_y'], inplace=True)
    final_df.rename(columns={'IBI_x': 'IBI'}, inplace=True)
    final_df.to_csv(pred_path)

fir_path = './norm'
folder_paths = getfile('./norm')
save_path = './PRED'
for folder in folder_paths:
    folder_path = os.path.join(fir_path, folder)
    savefile_path = os.path.join(save_path, folder)
    files = getfile(folder_path)
    os.mkdir('./PRED/' + folder)
    for file in files:
        file_path = os.path.join(folder_path, file)
        save = os.path.join(savefile_path, file)
        train_model(file_path, save)

