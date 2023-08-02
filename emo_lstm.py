import torch
import shutil
import pandas as pd
import os
import tensorflow as tf
import numpy as np


def getfile(path):
    file_list = []
    for name in os.listdir(path):
        file_list.append(name)
    return file_list

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
    if 'Unnamed: 0' in df.columns:
        df.drop(['Unnamed: 0'], axis=1, inplace=True)
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

    df_train = df_train.values.astype(np.float32)

    df_label = df_encoded.values.astype(np.float32)
    df_train = np.expand_dims(df_train, axis=1)
    print(df_label.shape)
    print(df_train.shape)
    return df_label, df_train


def train_model(path, class_weights, saved_model_path=None):
    df_label, df_train = call_df(path)
    train_size = int(0.7 * len(df_train))
    train_data = df_train[:train_size]
    train_label = df_label[:train_size]
    test_data = df_train[train_size:]
    test_label = df_label[train_size:]

    input_dim = 3
    hidden_dim = 64
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
        dnn_layer_units = 128
        model.add(tf.keras.layers.Dense(dnn_layer_units, activation='relu'))
        model.add(tf.keras.layers.Dense(dnn_layer_units, activation='relu'))
        model.add(tf.keras.layers.Dense(output_dim, activation='softmax'))
        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        print(model.summary())
    else:
        loaded_model = tf.keras.models.load_model(os.path.join(saved_model_path, 'my_model'))
        model = loaded_model

    model.fit(train_data, train_label, epochs=100, batch_size=5, validation_data=(test_data, test_label), class_weight=class_weights)


    loss, accuracy = model.evaluate(test_data, test_label)

    print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")
    if saved_model_path is not None:
        if not os.path.exists(saved_model_path):
            os.makedirs(saved_model_path)
        model.save(os.path.join(saved_model_path, 'my_model'))



fir_path = './ALL_CONV'
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
        print(file)
        train_model(file_path, class_weights, save)