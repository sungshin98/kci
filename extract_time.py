import pandas as pd
import os
save_csv = pd.DataFrame()
def getfile(path):
    file_list = []
    for name in os.listdir(path):
        file_list.append(name)
    return file_list

def extract_time(path):
    global save_csv
    df = pd.read_csv(path)
    for wave, time in df.groupby('Wave'):
        start = time.iloc[0]['Time']
        end = time.iloc[-1]['Time']
        name = wave
        print(start)
        data_to_append = {
            'Start': [start],
            'End': [end],
            'Name': [name]
        }

        save_csv = pd.concat([save_csv, pd.DataFrame(data_to_append)])

path = './TALK'
folders = getfile(path)

for folder in folders:
    folder_path = os.path.join(path, folder)  # 파일 경로를 올바르게 처리하기 위해 os.path.join 사용
    files = getfile(folder_path)
    for file in files:
        file_path = os.path.join(folder_path, file)  # 파일 경로를 올바르게 처리하기 위해 os.path.join 사용
        extract_time(file_path)

save_csv.reset_index(drop=True, inplace=True)

save_csv.to_csv('./extract_time.csv', index=False)