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

# 定义eating的标签列表
target_labels = ['H', 'I', 'J', 'K', 'L']

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

    # 在这里，我们定义一个筛选后的数据和标签容器
    data_filtered = []
    labels_filtered = []

    # Slide a "SEGMENT_TIME_SIZE" wide window with a step size of "TIME_STEP"
    for i in range(0, len(data) - SEGMENT_TIME_SIZE, TIME_STEP):
      segment = data.iloc[i: i + SEGMENT_TIME_SIZE]
      # Label for a data window is the label that appears most commonly
      label = segment['activity'].value_counts().idxmax()
      if label in target_labels:
        x = segment['x-axis'].values
        y = segment['y-axis'].values
        z = segment['z-axis'].values
        data_filtered.append([x, y, z])
        labels_filtered.append(label)

    # 将筛选后的数据转换为numpy数组(x y z 的并列->堆叠)
    data_filtered = np.asarray(data_filtered, dtype=np.float32).transpose(0, 2, 1)

    # 将标签转换为整数索引
    label_to_index = {label: index for index, label in enumerate(target_labels)}
    labels_index = [label_to_index[label] for label in labels_filtered]
    # 进行One-hot编码
    labels_filtered = np.eye(len(target_labels))[labels_index]
    print('-'*50)
    print("Filtered data shape: ", data_filtered.shape)
    print("Filtered labels shape:", labels_filtered.shape)

    # 使用筛选后的数据进行训练/测试集分割等后续工作
    x_train, x_test, y_train, y_test = train_test_split(data_filtered, labels_filtered, test_size=0.3, random_state=10)

    np.save('train_x_eat_p_acc.npy', x_train)
    np.save('train_y_eat_p_acc.npy', y_train)
    np.save('test_x_eat_p_acc.npy', x_test)
    np.save('test_y_eat_p_acc.npy', y_test)

    print('-'*50)
    print("x train size: ", x_train.shape)
    print("x test size: ", x_test.shape)
    print("y train size: ", y_train.shape)
    print("y test size: ", y_test.shape)

