import numpy as np


yy = np.array([
    [0.1, 0.2, 0.6, 0.05, 0.05],  # 对类别2预测概率0.6
    [0.7, 0.1, 0.1, 0.05, 0.05],  # 对类别0预测概率0.7
    [0.1, 0.1, 0.2, 0.5, 0.1]     # 对类别3预测概率0.5
])
t_one_hot = np.array([
    [0, 0, 1, 0, 0],  # 标签2
    [1, 0, 0, 0, 0],  # 标签0
    [0, 0, 0, 1, 0]   # 标签3
])
t_labels = np.array([2, 0, 3])  # 同样的数据，用标签形式表示，直接存储正确类别的索引

batch_size = yy.shape[0]
indices = np.arange(batch_size)  # [0, 1, 2]
print("样本索引:", indices)      # [0, 1, 2]
print("真实标签:", t_labels)     # [2, 0, 3]

selected_probs = yy[indices, t_labels]
print("选中的概率:", selected_probs)  # [0.6, 0.7, 0.5]
# 相当于：
# y[0, 2] → 第0个样本，第2个类别的概率 = 0.6
# y[1, 0] → 第1个样本，第0个类别的概率 = 0.7
# y[2, 3] → 第2个样本，第3个类别的概率 = 0.5


def cross_entropy_one_hot(y, t):
    return -np.sum(t * np.log(y + 1e-7)) / len(t)


def cross_entropy_label(y, t):
    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size


result1 = cross_entropy_one_hot(yy, t_one_hot)
result2 = cross_entropy_label(yy, t_labels)
print(result1)      # 0.5202157462469678
print(result2)      # 0.5202157462469678
