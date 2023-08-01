import torch
import shutil
import pandas as pd
import os
import tensorflow as tf

def getfile(path):
    file_list = []
    for name in os.listdir(path):
        file_list.append(name)
    return file_list

def merge_csv_files(folder_path, output_file):
    all_data = pd.DataFrame()

    for file_name in os.listdir(folder_path):
        if file_name.endswith('.csv'):
            file_path = os.path.join(folder_path, file_name)
            df = pd.read_csv(file_path)
            all_data = pd.concat([all_data, df], ignore_index=True)
            os.remove(file_path)

    all_data.to_csv(output_file, index=False)

def call_df(path):
    df = pd.read_csv(path)
    emotion_mapping = {
        'neutral': 0,
        'happy': 1,
        'sad': 2,
        'angry': 3,
        'surprise': 4,
        'disgust': 5,
        'fear': 6
    }
    df_train = df.drop(['Name', 'Emo'], axis=1)
    data = {
        'Emo': ['neutral', 'happy', 'sad', 'angry', 'surprise', 'disgust', 'fear']
    }

    df_encoded = pd.concat([df, pd.DataFrame(data)], ignore_index=True)
    df_encoded = pd.get_dummies(df_encoded['Emo'])
    df_encoded = pd.concat([df, df_encoded], axis=1)
    df_encoded.drop(['Emo', 'Name'], axis=1, inplace=True)
    df_encoded = df_encoded.dropna()
    df_encoded.drop(['EDA', 'IBI', 'TEMP'], axis=1, inplace=True)
    df_encoded.rename(columns=emotion_mapping, inplace=True)

    df_train = torch.tensor(df_train.values, dtype=torch.float)
    df_label = torch.tensor(df_encoded.values, dtype=torch.float)
    return df_label, df_train


def train_model(path, class_weights, saved_model_path=None):
    df_label, df_train = call_df(path)

    input_dim = 3
    hidden_dim = 16
    layer_dim = 7
    output_dim = 7
    learning_rate = 0.001
    num_epochs = 100

    batch_size = 1
    model_save_path = os.path.join(saved_model_path, 'my_model')

    if not os.path.exists(model_save_path):
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(hidden_dim, return_sequences=True, input_shape=(None, input_dim)),
            tf.keras.layers.LSTM(hidden_dim),
            tf.keras.layers.Dense(output_dim, activation='softmax')
        ])
        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

    else:
        loaded_model = tf.keras.models.load_model(os.path.join(saved_model_path, 'my_model'))
        model = loaded_model

    model.fit(df_train, df_label, epochs=100, batch_size=1, class_weight=class_weights)
    test_data, test_labels = call_df('./test_set')

    loss, accuracy = model.evaluate(test_data, test_labels)

    print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")
    if saved_model_path is not None:
        if not os.path.exists(saved_model_path):
            os.makedirs(saved_model_path)
        model.save(os.path.join(saved_model_path, 'my_model'))



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

representatives = {
    'neutral': None,
    'happy': None,
    'sad': None,
    'angry': None,
    'surprise': None,
    'disgust': None,
    'fear': None
}

for folder in folder_paths:
    folder_path = os.path.join(fir_path, folder)
    files = getfile(folder_path)
    for file in files:
        file_path = os.path.join(folder_path, file)
        df = pd.read_csv(file_path)
        emo_value = df.iloc[0]['Emo']

        if emo_value in representatives and df.shape[0] >= 30 and representatives[emo_value] is None:
            representatives[emo_value] = './test_set/' + file
            shutil.move(file_path, './test_set/' + file)
        if all(representatives.values()):
            break
print('comlete combine file')

test_path = './test_set'
output_path = './test_set/conv.csv'
merge_csv_files(test_path, output_path)

for folder in folder_paths:
    folder_path = os.path.join(fir_path, folder)
    files = getfile(folder_path)
    for file in files:
        file_path = os.path.join(folder_path, file)
        train_model(file_path, class_weights, save)