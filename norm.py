import pandas as pd
import matplotlib.pyplot as plt
import os
import csv

def norm0t01(data):
    min_val = min(data)
    max_val = max(data)
    normalized_data = [(x - min_val) / (max_val - min_val) for x in data]
    return normalized_data

def drop_rows(df):
    df_cleaned = df.dropna(subset=['EDA', 'TEMP'], how='any')
    return df_cleaned

def col_df(path):
    df = pd.read_csv(path, index_col='Time')
    df = drop_rows(df)
    if 'Unnamed: 0' in df.columns:
        df.drop(['Unnamed: 0'], axis=0)
    df_label = df['IBI']
    df.drop(df['IBI'], inplace=True)
    return df, df_label

def getfile(path):
    file_list = []
    for name in os.listdir(path):
        file_list.append(name)
    return file_list

file = './BIO'
sess_path = getfile(file)
