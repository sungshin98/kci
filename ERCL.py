import pandas as pd
import os
import tensorflow as tf
import numpy as np
from keras.optimizers import RMSprop
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
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
    df = df.drop(['listen', 'stay', 'talk', 'Current Time', 'M', 'F'], axis = 1)
    df_train = df[['EDA', 'TEMP', 'IBI']].values
    df_label = df.iloc[:,-7:].values

    return df_label, df_train

df_label, df_train = call_df('./LSTM_ERCL_DATAFILE/User002M_new_emo_gender.csv')
X_train, X_test, y_train, y_test = train_test_split(df_train, df_label, test_size = 0.2)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
opt = RMSprop(learning_rate=0.0001)
print(X_train.shape)

model = tf.keras.Sequential()
model.add(tf.keras.layers.Convolution1D(filters=128, kernel_size=3, activation='relu', input_shape=(X_train.shape[1],1)))
model.add(tf.keras.layers.LSTM(128, return_sequences=True, activation='softmax'))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.LSTM(32, return_sequences=True, activation='softmax'))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.LSTM(y_train.shape[1], activation='softmax'))

model.compile(loss='categorical_crossentropy' , optimizer=opt, metrics=['accuracy'])
files = os.listdir('./LSTM_ERCL_DATAFILE')
for num in range(len(files)):
    df_label, df_train = call_df(os.path.join('./LSTM_ERCL_DATAFILE', files[num]))
    # 입력 데이터와 라벨 데이터를 분리

    X_train, X_test, y_train, y_test = train_test_split(df_train, df_label, test_size = 0.2)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.2)
    # 입력 데이터를 3D 텐서로 변환
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)


    print(X_train.shape)
    earlystop = EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights = True)
    # 모델 학습 및 체크포인트 저장
    fit_history = model.fit(X_train, y_train, epochs=100, batch_size=3, validation_data=(X_val, y_val), callbacks=[earlystop])

    # 모델 평가
    loss, accuracy = model.evaluate(X_test, y_test)
    print('Test accuracy:', accuracy)
    print(num)
    #모델 저장
    model.save("my_model.h5")
    reconstructed_model = tf.keras.models.load_model("my_model.h5")
    np.testing.assert_allclose(model.predict(X_test),
                               reconstructed_model.predict(X_test)
                              )
    model = reconstructed_model
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
y_test, X_test = call_df('./test_set/conv.csv')
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
loss, accuracy = model.evaluate(X_test, y_test)
print('Test accuracy:', accuracy, 'Test Loss:', loss)