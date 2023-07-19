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

def csv_time_wave(csv_file, out_file):
    '''
    temp csv파일에서 시간과 wave파일명이 존재하는 것만 같이 추출
    1열 2열 이름 각각 TIme, Wave로 설정
    '''
    with open(csv_file, 'r', newline='') as file:
        reader = csv.reader(file)
        header = next(reader)

        # 3번째 열에 값이 있는 행 추출
        data = [row for row in reader if len(row) >= 3]

    # 추출된 데이터를 원하는 열로 구성된 리스트로 변환
    new_data = [[row[1], row[2]] for row in data]

    # 추출된 데이터를 CSV 파일로 저장
    with open(out_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Time', 'Wave'])  # 헤더 쓰기
        writer.writerows(new_data)

                
fir_path = "./TEMP"
in_path = getfile(fir_path)
for i in range(len(in_path)):
    path = "./TEMP/" + in_path[i]
    in2_path = getfile(path)
    talk_path = './TALK/' + in_path[i]
    os.mkdir(talk_path)
    for j in range(len(in2_path)):
        last_path = path + '/' + in2_path[j]
        last2_path = talk_path + '/' + in2_path[j]
        csv_time_wave(last_path, last2_path)
        rmrow(last_path)
        clear_csv(last_path)
