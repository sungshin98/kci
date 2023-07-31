import torch
import torch.nn as nn
import pandas as pd
import os
import torch.optim as optim
from sklearn.model_selection import TimeSeriesSplit
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import f1_score as f1
import torch.nn.functional as F
import numpy as np

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


def evaluate_model(model, test_data, test_labels, class_weights):
    # 모델을 평가 모드로 설정
    model.eval()

    # 테스트 데이터와 레이블을 GPU로 이동
    test_data = test_data.to(device)
    test_labels = test_labels.to(device).long()

    # 예측값 계산
    with torch.no_grad():
        outputs = model(test_data)

    # 예측값을 클래스 레이블로 변환 (가장 높은 확률 값을 가지는 클래스로 선택)
    _, predicted_labels = torch.max(outputs, 1)

    # 평가 결과 계산
    accuracy = (predicted_labels == test_labels).sum().item() / len(test_labels)

    # CrossEntropyLoss를 사용한 경우에는 클래스별 가중치를 고려해 평가 지표를 계산하는 것이 좋습니다.
    # 여기에서는 F1 점수를 계산하였습니다. (다른 평가 지표도 계산 가능)
    f1_score = f1(test_labels.cpu().numpy(), predicted_labels.cpu().numpy(), average='weighted',
                  sample_weight = np.array(list(class_weights.values())))
    # 결과 출력
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1_score:.4f}")


def getfile(path):
    file_list = []
    for name in os.listdir(path):
        file_list.append(name)
    return file_list


def call_df(path):
    df = pd.read_csv(path)

    emotion_mapping = {
        0: 'neutral',
        1: 'happy',
        2: 'sad',
        3: 'angry',
        4: 'surprise',
        5: 'disgust',
        6: 'fear'
    }
    df['Emo'] = df['Emo'].map(emotion_mapping)
    df_train = df.drop(['Name', 'Emo'], axis=1)  # Drop the 'Name' and 'Emo' columns
    df_label = df['Emo'].map(lambda x: emotion_mapping.get(x)).tolist()  # Convert to list of emotion labels

    # Initialize an empty one-hot tensor with all zeros
    num_classes = len(emotion_mapping)
    df_label_one_hot = torch.zeros(len(df_label), num_classes, dtype=torch.float)

    # Set the appropriate index to 1 based on the emotion label mapping
    for i, label in enumerate(df_label):
        if label in emotion_mapping.values():
            idx = list(emotion_mapping.values()).index(label)
            df_label_one_hot[i, idx] = 1.0

    df_train = torch.tensor(df_train.values, dtype=torch.float)  # Convert DataFrame to numpy array first

    return df_label_one_hot, df_train


def train_model(path, class_weights, saved_model_path=None):
    df_label, df_train = call_df(path)

    input_dim = 3
    hidden_dim = 16
    layer_dim = 7
    output_dim = 7
    learning_rate = 0.001
    num_epochs = 100

    batch_size = 1


    k_folds = len(df_train)
    tscv = TimeSeriesSplit(n_splits=k_folds)

    for fold, (train_idx, test_idx) in enumerate(tscv.split(df_train)):
        print(f"Training on fold {fold + 1}...")
        # Split the data into train and test sets based on current fold
        train_data, test_data = df_train[train_idx], df_train[test_idx]
        train_labels, test_labels = df_label[train_idx], df_label[test_idx]


        # Create DataLoader for training set
        train_dataset = torch.utils.data.TensorDataset(train_data, train_labels)
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size)

        model = LSTMModel(input_dim, hidden_dim, layer_dim, output_dim)
        if saved_model_path is not None and os.path.exists(saved_model_path):
            model.load_state_dict(torch.load(saved_model_path))
            print("Pretrained model loaded.")
        criterion = nn.CrossEntropyLoss()  # 클래스 가중치는 loss 함수에 적용하므로 삭제
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        for epoch in range(num_epochs):
            model.train()  # 모델을 학습 모드로 설정
            running_loss = 0.0
            for inputs, labels in train_loader:
                # 입력 데이터와 레이블을 GPU로 이동
                inputs = inputs.to(device)
                labels = labels.to(device).long()  # CrossEntropyLoss를 사용하기 때문에 레이블은 long 형태여야 합니다.
                seq_input = len(inputs)
                inputs = inputs.reshape(batch_size, seq_input, input_dim)
                # 순전파 (Forward)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                # 역전파 (Backward) 및 가중치 업데이트
                loss.backward()
                optimizer.step()

                # 손실 누적
                running_loss += loss.item()

            if epoch%10 == 0 :
                print(f"Epoch [{epoch + 1}/{num_epochs}] - Loss: {running_loss / len(train_loader):.4f}")
        seq_test = len(test_data)
        test_data = test_data.reshape(batch_size, seq_test, input_dim)
        evaluate_model(model, test_data, test_labels, class_weights)

        if saved_model_path is not None:
            torch.save(model.state_dict(), saved_model_path)
            print("Model saved.")

        print("Training completed.")

fir_path = './CONV'
folder_paths = getfile(fir_path)
save = './models'
class_weights = {
    0: 0.00217,
    1: 0.01563,
    2: 0.12586,
    3: 0.10376,
    4: 0.11394,
    5: 0.23667,
    6: 0.40197
}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
for folder in folder_paths:
    folder_path = os.path.join(fir_path, folder)
    files = getfile(folder_path)
    for file in files:
        file_path = os.path.join(folder_path, file)
        train_model(file_path, class_weights, save)