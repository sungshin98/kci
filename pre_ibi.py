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
        try:
            first_row = next(reader)
            second_row = next(reader)

            first_row[1] = second_row[1]

            writer.writerow(first_row)

            for num in reader:
                writer.writerow([num[0], num[1], num[2]])
        except StopIteration:
            print(f"File '{path}' is empty or has less than two rows.")
            return
    time.sleep(1)
    os.remove(path)
    os.rename(temp_file, path)

def clear_csv(path):
    try:
       df = pd.read_csv(path, header=None)
    except pd.errors.EmptyDataError:
        print(f"File '{path}' is empty or has no columns.")
        empty_files.append(path)
        return
    new_columns = ['accu_IBI', 'IBI', 'Time']
    df.columns = new_columns
    df.drop(columns='accu_IBI', inplace=True)
    df.to_csv(path, index = None)

def resample_time_series(input_file, regular_interval, Time_reset):
    # CSV 파일을 pandas DataFrame으로 로드합니다.
    if os.path.getsize(input_file) == 0:
        print(f"File '{input_file}' is empty. Skipping resampling.")
        return
    df = pd.read_csv(input_file)
    df['Time'] = pd.to_datetime(df['Time'], format = '%Y-%m%d-%H%M-%S-%f')
    # 'IBI' 열을 숫자형으로 변환합니다.
    df['IBI'] = pd.to_numeric(df['IBI'], errors='coerce')
    df.set_index('Time', inplace=True)
    df = df.interpolate(method='time', limit_direction='both')
    df = df.resample(rule=regular_interval).mean()
    print(df.head())
    os.remove(input_file)
    df.to_csv(input_file)

fir_path = "./IBI"
in_path = getfile(fir_path)
empty_files = []
for i in range(len(in_path)):
    path = "./IBI/" + in_path[i]
    in2_path = getfile(path)
    for j in range(len(in2_path)):
        last_path = path + '/' + in2_path[j]
        rmrow(last_path)
        clear_csv(last_path)
        resample_time_series(last_path, regular_interval='1S', Time_reset=0)

print("Empty files:", empty_files)