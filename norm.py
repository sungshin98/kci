import pandas as pd
import matplotlib.pyplot as plt
import os
import csv

def normalization(data):
    min_val = min(data)
    max_val = max(data)
    normalized_data = [(x - min_val) / (max_val - min_val) for x in data]
    return normalized_data

def col_df(path):
    df = pd.read_csv(path, index_col='Time')
    print(df.head())
    if 'Unnamed: 0' in df.columns:
        df.drop(['Unnamed: 0'], axis=0)
    df_label = df['IBI']
    df.drop(df['IBI'], inplace=True)
    return df, df_label

print(df.dtypes)