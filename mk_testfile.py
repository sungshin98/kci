import pandas as pd
import os

emotion_labels = ['neutral', 'happy', 'sad', 'angry', 'surprise', 'disqust', 'fear']
folder_name = './CONV'

selected_files = {label: [] for label in emotion_labels}

folder_list = os.listdir(folder_name)
for folder in folder_list:
    file_list = os.listdir(os.path.join(folder_name, folder))
    if not file_list:
        print(f'{folder}은 비었습니다.')
        continue
    for file in file_list:
        df = pd.read_csv(file_path := os.path.join(folder_name, folder, file))
        if len(df) <1:
            continue
        if len(df) >= 20 and df['Emo'][0] in emotion_labels:
            label = df['Emo'][0]
            if len(selected_files[label]) >= 10:
                continue
            selected_files[label].append(file_path)
print(selected_files)

dataframes = []  # 데이터프레임들을 저장할 리스트

# 주어진 딕셔너리에서 파일 위치들을 순회하며 데이터프레임들을 불러와 리스트에 추가
for label, file_paths in selected_files.items():
    for file_path in file_paths:
        df = pd.read_csv(file_path)
        dataframes.append(df)

# 데이터프레임들을 하나로 합치기
final_dataframe = pd.concat(dataframes, ignore_index=True)

for folder in (folder_list := os.listdir('./ALL_CONV')):
    file_list = os.listdir(os.path.join('./ALL_CONV', folder))
    if not file_list:
        print(f'{folder}은 비었습니다.')
        continue
    for file in file_list:
        df = pd.read_csv(file_path := os.path.join('./ALL_CONV', folder, file))
        df = df.drop(columns=[col for col in df.columns if col.startswith("Unnamed:")])
        df = df[~df['Name'].isin(final_dataframe['Name'].tolist())]
        os.remove(file_path)
        df.to_csv(file_path)