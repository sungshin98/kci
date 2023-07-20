import pandas as pd
import os
import csv
import matplotlib.pyplot as plt
def getfile(path):
    file_list = []
    for name in os.listdir(path):
        file_list.append(name)
    return file_list

def read_filename(folder, file):
    eda = './EDA/'
    ibi = './IBI/'
    temp = './TEMP/'
    eda_path = eda + folder + '/' + file
    ibi_path = ibi + folder + '/' + file
    temp_path = temp + folder + '/' + file

    df_eda = pd.read_csv(eda_path)
    df_ibi = pd.read_csv(ibi_path)
    df_temp = pd.read_csv(temp_path)
    df_eda['Time'] = pd.to_datetime(df_eda['Time'])
    df_ibi['Time'] = pd.to_datetime(df_ibi['Time'])
    df_temp['Time'] = pd.to_datetime(df_temp['Time'])
    return df_eda, df_ibi, df_temp

def resample_df(df):
    df.set_index('Time', inplace = True)
    df = df.resample('0.25S').interpolate()
    df.reset_index(inplace=True)
    return df

#파일이름 추출
fir_path = "./EDA"
folder_list = []
file_list = []
in_path = getfile(fir_path)
for i in range(len(in_path)):
    path = "./EDA/" + in_path[i]
    in2_path = getfile(path)
    folder_list.append(in_path[i])
    for j in range(len(in2_path)):
        last_path = path + '/' + in2_path[j]
        file_list.append(in2_path[j])

for i in range(len(folder_list)):
    os.mkdir('./BIO/' + folder_list[i])
    for j in range(len(file_list)):
        if file_list[j] in \
                ['Sess01_script05_User002M.csv', 'Sess03_script04_User005M.csv', 'Sess04_script06_User007M.csv']:
            continue
        if folder_list[i][-2:] != file_list[j][4:6]:
            continue
        df_eda, df_ibi, df_temp = read_filename(folder_list[i], file_list[j])
        df_temp = resample_df(df_temp)
        df = pd.merge(df_eda, df_ibi, on='Time', how='outer')
        df = pd.merge(df, df_temp, on='Time', how='outer')
        df.to_csv('./BIO/' + folder_list[i] + '/' + file_list[j])