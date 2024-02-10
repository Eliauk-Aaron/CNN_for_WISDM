import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import argparse #pass the argument

# 创建ArgumentParser对象
parser = argparse.ArgumentParser(description='Process data path.')
# 添加DATA_PATH参数
parser.add_argument('DATA_PATH', type=str, help='The path to your data file')
# args = parser.parse_args()  # 解析参数


COLUMN_NAMES = [
    'user',
    'activity',
    'timestamp',
    'x-axis',
    'y-axis',
    'z-axis'
]

LABELS = [
    'A',
    'B',
    'C',
    'D',
    'E',
    'F',
    'G',
    'H',
    'I',
    'J',
    'K',
    'L',
    'M',
    'O',
    'P',
    'Q',
    'R',
    'S'
]

# LABELS = [
#     'Downstairs',
#     'Jogging',
#     'Sitting',
#     'Standing',
#     'Upstairs',
#     'Walking'
# ]

# DATA_PATH = args.DATA_PATH
# DATA_PATH = 'WISDM_ar_v1.1_raw.txt'

base_path = "/content/drive/MyDrive/project_codes_data/wisdm-dataset/raw/phone/accel/"
DATA_PATH = [f"{base_path}data_{i}_accel_phone.txt" for i in range(1600, 1651)]


RANDOM_SEED = 13

# Data preprocessing
TIME_STEP = 100

# Model
N_CLASSES = 6
N_FEATURES = 3  # x-acceleration, y-acceleration, z-acceleration

# Hyperparameters optimized
SEGMENT_TIME_SIZE = 200


if __name__ == '__main__':

    # # LOAD DATA
    # data = pd.read_csv(DATA_PATH, header=None, names=COLUMN_NAMES)

    # load data
    data = pd.DataFrame()
    # 遍历DATA_PATH列表中的每个文件路径
    for file_path in DATA_PATH:
      # 读取当前文件的数据
      temp_data = pd.read_csv(file_path, header=None, names=COLUMN_NAMES)
      # 将当前文件的数据追加到data DataFrame中
      data = pd.concat([data, temp_data], ignore_index=True)


    data['z-axis'].replace({';': ''}, regex=True, inplace=True)
    data = data.dropna()

    # # SHOW GRAPH FOR JOGGING
    # data[data['activity'] == 'Jogging'][['x-axis']][:50].plot(subplots=True, figsize=(16, 12), title='Jogging')
    # plt.xlabel('Timestep')
    # plt.ylabel('X acceleration (dg)')

    # SHOW GRAPH FOR JOGGING
    data[data['activity'] == 'H'][['x-axis']][:50].plot(subplots=True, figsize=(16, 12), title='Eating soap')
    plt.xlabel('Timestep')
    plt.ylabel('X acceleration (dg)')

    # # SHOW ACTIVITY GRAPH
    # activity_type = data['activity'].value_counts().plot(kind='bar', title='Activity type')
    # # plt.show()

    # SHOW ACTIVITY GRAPH
    activity_type = data['activity'].value_counts().plot(kind='bar', title='Activity type')
    plt.show()

    # DATA PREPROCESSING
    data_convoluted = []
    labels = []

    # Slide a "SEGMENT_TIME_SIZE" wide window with a step size of "TIME_STEP"
    for i in range(0, len(data) - SEGMENT_TIME_SIZE, TIME_STEP):
        x = data['x-axis'].values[i: i + SEGMENT_TIME_SIZE]
        y = data['y-axis'].values[i: i + SEGMENT_TIME_SIZE]
        z = data['z-axis'].values[i: i + SEGMENT_TIME_SIZE]
        data_convoluted.append([x, y, z])

        # # Label for a data window is the label that appears most commonly
        # label = stats.mode(data['activity'][i: i + SEGMENT_TIME_SIZE])[0][0]
        # labels.append(label)

        # Label for a data window is the label that appears most commonly
        label = data['activity'][i: i + SEGMENT_TIME_SIZE].value_counts().idxmax()
        labels.append(label)

    # Convert to numpy (x y z 的并列->堆叠)
    data_convoluted = np.asarray(data_convoluted, dtype=np.float32).transpose(0, 2, 1)

    # One-hot encoding
    labels = np.asarray(pd.get_dummies(labels), dtype=np.float32)
    print('-'*50)
    print("Convoluted data shape: ", data_convoluted.shape)
    print("Labels shape:", labels.shape)

    # saveed all the data
    np.save('data_phone_acc.npy', data_convoluted)
    np.save('lable_phone_acc.npy', labels)

    print('-'*50)
    print("data_phone_acc size: ", data_convoluted.shape)
    print("lable_phone_acc size: ", labels.shape)

