import csv
import os
import pandas as pd
import time

def getfile(path):
    file_list = []
    for name in os.listdir(path):
        file_list.append(name)
    return file_list

def rmrow(path):
    abs_path = os.path.abspath(path)
    temp_file = abs_path + ".tmp"

    with open(path, 'r') as infile, open(temp_file, 'w', newline='') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)

        next(reader)
        next(reader)

        for num in reader:
            writer.writerow([num[0], num[1]])

    time.sleep(1)
    os.remove(path)
    os.rename(temp_file, path)

def clear_csv(path):
    df = pd.read_csv(path, header = None)
    new_columns = ['EDA', "Time"]
    df.columns = new_columns
    df['Time'] = pd.to_datetime(df['Time'], format = '%Y-%m%d-%H%M-%S-%f')
    df.to_csv(path, index = False)

fir_path = "./EDA"
in_path = getfile(fir_path)
for i in range(len(in_path)):
    path = "./EDA/" + in_path[i]
    in2_path = getfile(path)
    for j in range(len(in2_path)):
        last_path = path + '/' + in2_path[j]
        rmrow(last_path)
        clear_csv(last_path)
