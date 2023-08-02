import torch
import shutil
import pandas as pd
import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping

def getfile(path):
    file_list = []
    for name in os.listdir(path):
        file_list.append(name)
    return file_list

def call_df(path):
    df = pd.read_csv(path, index_col=False)
    emotion_mapping = {
        'neutral': 0,
        'happy': 1,
        'sad': 2,
        'angry': 3,
        'surprise': 4,
        'disqust': 5,
        'fear': 6
    }
    df = df.drop(columns=[col for col in df.columns if col.startswith("Unnamed:")])
    df_train = df.drop(['Name', 'Emo'], axis=1)
    data = {
        'Emo': ['neutral', 'happy', 'sad', 'angry', 'surprise', 'disqust', 'fear']
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
    train_label, train_data = call_df(path)
    test_label, test_data = call_df('./test_set/conv.csv')

    input_dim = 3
    hidden_dim = 32
    layer_dim = 7
    output_dim = 7
    learning_rate = 0.001
    num_epochs = 100

    batch_size = 1
    model_save_path = os.path.join(saved_model_path, 'my_model')
    dnn_layer_units = 128
    #, kernel_regularizer = tf.keras.regularizers.l2(0.01)
    if not os.path.exists(model_save_path):
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(hidden_dim, return_sequences=True, input_shape=(None, input_dim), kernel_regularizer = tf.keras.regularizers.l2(0.01)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(hidden_dim, kernel_regularizer = tf.keras.regularizers.l2(0.01)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(dnn_layer_units, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(output_dim, activation='softmax')
        ])
        """dnn_layer_units = 128
        model.add(tf.keras.layers.Dense(dnn_layer_units, activation='relu'))
        model.add(tf.keras.layers.Dense(dnn_layer_units, activation='relu'))
        model.add(tf.keras.layers.Dense(output_dim, activation='softmax'))"""
        model.compile(optimizer='RMSprop',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        model.summary()
    else:
        loaded_model = tf.keras.models.load_model(os.path.join(saved_model_path, 'my_model'))
        model = loaded_model
    """ early_stopping = EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)"""

    model.fit(train_data,
              train_label,
              epochs=10,
              batch_size=3,
              validation_data=(test_data, test_label),
              class_weight=class_weights)

    loss, accuracy = model.evaluate(test_data, test_label)

    print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")
    if saved_model_path is not None:
        if not os.path.exists(saved_model_path):
            os.makedirs(saved_model_path)
        model.save(os.path.join(saved_model_path, 'my_model'))



fir_path = './ALL_CONV'
folder_paths = getfile(fir_path)
save = './models'
modelpath = os.path.join(save,'my_model')
if os.path.exists(modelpath):
    shutil.rmtree(modelpath)

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

train_model('./test_set/conv.csv', class_weights, save)
for folder in reversed(folder_paths):
    folder_path = os.path.join(fir_path, folder)
    files = getfile(folder_path)
    if not files:
        continue
    for file in files:
        file_path = os.path.join(folder_path, file)
        if len(data:=pd.read_csv(file_path)) <1:
            continue
        print(file_path)
        train_model(file_path, class_weights, save)