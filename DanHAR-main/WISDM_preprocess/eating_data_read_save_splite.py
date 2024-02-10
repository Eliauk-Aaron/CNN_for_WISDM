#filter the eating data and split them
import numpy as np
from sklearn.model_selection import train_test_split

# read from data_phone_acc和label_phone_acc
data_phone_acc = np.load('/content/drive/MyDrive/project_codes_data/DanHAR-main/WISDM_preprocess/data_phone_acc.npy')
label_phone_acc = np.load('/content/drive/MyDrive/project_codes_data/DanHAR-main/WISDM_preprocess/label_phone_acc.npy')

# 目标类别的索引 (eating HIJKL)
target_indices = [7, 8, 9, 10, 11]

# 找出标签中属于这些类别的行的索引
# 因为标签已经是one-hot编码，可以通过检查对应列是否有1来确定
target_rows = np.any(label_phone_acc[:, target_indices] == 1, axis=1)

# 使用这些行的索引来提取数据和标签
data_filtered = data_phone_acc[target_rows]
label_filtered = label_phone_acc[target_rows]
np.save('data_phone_acc_eating.npy', data_filtered)
np.save('lable_phone_acc_eating.npy', label_filtered)
print(data_filtered.shape)
print(label_filtered.shape)
# print(label_filtered[:50])

#splite the data
# SPLIT INTO TRAINING AND TEST SETS
x_train, x_test, y_train, y_test = train_test_split(data_filtered, label_filtered, test_size=0.3, random_state=10)
np.save('train_x_eat_p_acc.npy', x_train)
np.save('train_y_eat_p_acc.npy', y_train)
np.save('test_x_eat_p_acc.npy', x_test)
np.save('test_y_eat_p_acc.npy', y_test)
print('-'*50)
print("x train size: ", x_train.shape)
print("x test size: ", x_test.shape)
print("y train size: ", y_train.shape)
print("y test size: ", y_test.shape)
