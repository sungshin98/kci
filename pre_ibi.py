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


        first_row = next(reader)
        second_row = next(reader)

        first_row[0] = float(second_row[0])-float(second_row[1])
        first_row[1] = second_row[1]

        writer.writerow(first_row)

        for num in reader:
            writer.writerow([num[0], num[1], num[2]])

    time.sleep(1)
    os.remove(path)
    os.rename(temp_file, path)


def resample_time_series(input_file, output_file, regular_interval='1T'):
    # CSV 파일을 pandas DataFrame으로 로드합니다.
    df = pd.read_csv(input_file)

    # "Timestamp" 열을 날짜 및 시간 형식으로 변환합니다.
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])

    # "Timestamp" 열을 인덱스로 설정합니다. (재샘플링을 위해 필요합니다.)
    df.set_index('Timestamp', inplace=True)

    # 데이터를 일정한 간격으로 재샘플링합니다. (여기서는 1분 간격으로 설정했습니다.)
    # '1T'를 원하는 다른 간격으로 변경할 수 있습니다. (예: '5T'는 5분 간격, '1H'는 1시간 간격 등)
    resampled_df = df.resample(regular_interval).mean()

    # 재샘플링된 데이터를 새로운 CSV 파일로 저장합니다.
    resampled_df.to_csv(output_file)

"""fir_path = "./EDA"
in_path = getfile(fir_path)
for i in range(len(in_path)):
    path = "./EDA/" + in_path[i]
    in2_path = getfile(path)
    for j in range(len(in2_path)):
        last_path = path + '/' + in2_path[j]
        rmrow(last_path)
"""
rmrow("./IBI/Session01/Sess01_script01_User001F.csv")
resample_time_series("./IBI/Session01/Sess01_script01_User001F.csv","./EDA/Session01/Sess01_script01_User001F.csv")