import pandas as pd
import os
import csv
import matplotlib.pyplot as plt
def getfile(path):
    file_list = []
    for name in os.listdir(path):
        file_list.append(name)
    return file_list

def mk_filename(folder, file):
    eda = './EDA/'
    ibi = './IBI/'
    temp = './TEMP/'
    eda_path = eda + folder + '/' + file
    ibi_path = ibi + folder + '/' + file
    temp_path = temp + folder + '/' + file
    df_eda = pd.read_csv(eda_path)
    df_ibi = pd.read_csv(ibi_path)
    df_temp = pd.read_csv(temp_path)
    return df_eda, df_ibi, df_temp

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

df_eda, df_ibi, df_temp = mk_filename(folder_list[0], file_list[0])

df = pd.merge(df_ibi, df_temp, on='Time', how='outer')

print(df)