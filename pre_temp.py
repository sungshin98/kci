import pandas as pd
import csv
import os


def csv_temp_time(csv_file, out_file):
    '''
    temp csv파일에서 1,2 열 추출 및 1,2행 삭제
    1열 2열 이름 각각 Temp, Time으로 설정
    '''
    data = pd.read_csv(csv_file, skiprows=[0, 1], usecols=[0, 1], names=['Temp', 'Time'])
    data.to_csv(out_file, index=False)
    

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
        

def pre_temp(root_folder, output_folder, pre_func):
    for root, _, files in os.walk(root_folder):
        for file in files:
            if file.endswith('.csv'):
                csv_file = os.path.join(root, file)

                folder_name = os.path.basename(root)
                out_folder = os.path.join(output_folder, folder_name)
                if not os.path.exists(out_folder):
                    os.makedirs(out_folder)

                # 변환된 파일 경로 생성
                out_file = os.path.join(out_folder, file)

                pre_func(csv_file, out_file)

                
# csv파일 상위 폴더 위치
root_folder = 'D:/laboratory things/paper/multimodalemotion/KEMDy20_v1_1/TEMP'
# 변환된 csv파일을 저장할 상위 폴더 위치 pre_temp_wave가 저장될 폴더명
output_time_wave = 'D:/laboratory things/paper/multimodalemotion/KEMDy20_v1_1/pre_time_wave'
output_temp_time = 'D:/laboratory things/paper/multimodalemotion/KEMDy20_v1_1/pre_temp'

# csv파일에서 time과 wave파일명만 있는 csv파일 생성
pre_temp(root_folder, output_time_wave, csv_time_wave)
# csv파일에서 Temp와 Time만 있는 csv파일 생성
pre_temp(root_folder, output_temp_time, csv_temp_time)