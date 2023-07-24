import pandas as pd
import matplotlib.pyplot as plt
import os
import csv

#0~1 정규화
def norm0t01(data):
    # Convert data to numeric
    data_numeric = pd.to_numeric(data, errors='coerce')
    # Drop rows where the conversion to numeric failed
    data_numeric = data_numeric.dropna()
    min_val = data_numeric.min()
    max_val = data_numeric.max()
    normalized_data = [(x - min_val) / (max_val - min_val) for x in data_numeric]
    return normalized_data

#ibi를 제외한 칼럼의 값이 없을 경우 행삭제
def drop_rows(df):
    df_cleaned = df.dropna(subset=['EDA', 'TEMP'], how='any')
    return df_cleaned

#파일 불러오기
def col_df(path):
    df = pd.read_csv(path, index_col='Time')
    df = drop_rows(df)
    if 'Unnamed: 0' in df.columns:
        df.drop(['Unnamed: 0'], axis=1, inplace=True)
    return df

#파일 이름 얻기
def getfile(path):
    file_list = []
    for name in os.listdir(path):
        file_list.append(name)
    return file_list

path = './BIO'
sesss = getfile(path)
save_path = './norm'
for sess in sesss:
    folder_path = os.path.join(path, sess)
    files = getfile(folder_path)
    os.mkdir('./norm/' + sess)
    for file in files:
        file_path = os.path.join(folder_path, file)
        df = col_df(file_path)
        df = drop_rows(df)
        df['EDA'] = norm0t01(df['EDA'])
        df['TEMP'] = norm0t01(df['TEMP'])
        df.to_csv(save_path + '/' + sess + '/' + file)