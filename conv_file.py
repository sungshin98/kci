import pandas as pd
import os

def getfile(path):
    file_list = []
    for name in os.listdir(path):
        file_list.append(name)
    return file_list


def normibi(df_data):
    data_numeric = pd.to_numeric(df_data, errors='coerce')
    min_val = data_numeric.min()
    max_val = data_numeric.max()
    normalized_data = [(x - min_val) / (max_val - min_val) for x in data_numeric]
    return normalized_data


def call_df(path):
    df = pd.read_csv(path)
    df['Time'] = pd.to_datetime(df['Time'])
    df.set_index('Time', inplace=True)
    if 'Unnamed: 0' in df.columns:
        df.drop('Unnamed: 0', axis=1, inplace=True)
    return df


emo = pd.read_csv('./emotion.csv', header=None)
emo.columns = ['Name', 'Emo']
emo.sort_values(by = 'Name')

ext_time = pd.read_csv('./extract_time.csv')
ext_time['Start'] = pd.to_datetime(ext_time['Start'], format='%Y-%m%d-%H%M-%S-%f')
ext_time['End'] = pd.to_datetime(ext_time['End'], format='%Y-%m%d-%H%M-%S-%f')

path = './PRED'
save_path = './ALL_CONV'
folder_paths = getfile(path)

for folder_path in folder_paths:
    folder = os.path.join(path, folder_path)
    file_paths = getfile(folder)
    os.mkdir(save_folder := os.path.join(save_path, folder_path))
    for file_path in file_paths:
        file = os.path.join(folder, file_path)

        df = call_df(file)

        df['Name'] = None
        for i, ext_row in ext_time.iterrows():
            name = ext_row['Name']
            if name[:24] == file[17:-4]:
                start_time = ext_row['Start']
                end_time = ext_row['End']
                df.loc[(df.index >= start_time) & (df.index <= end_time), 'Name'] = name
                
        merge_df = pd.merge(df, emo, on='Name', how='left')
        merge_df['IBI'] = normibi(merge_df['IBI'])

        merge_df = merge_df.dropna(subset=['Name'])
        merge_df.to_csv(os.path.join(save_folder, file_path))
