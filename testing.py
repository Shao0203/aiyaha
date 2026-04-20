import numpy as np


def standardize(data):
    """将数据标准化为均值为0, 方差为1"""
    return (data - np.mean(data)) / np.std(data)    # 公式 = (数据 - 均值) / 标准差


# 验证标准化效果
original_data = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
standardized_data = standardize(original_data)

print(f"原始数据: {original_data}")  # [10  20  30  40  50  60  70  80  90 100]
print(f"原始均值: {original_data.mean():.4f}")          # 55.0000
print(f"原始方差: {original_data.var():.4f}")           # 825.0000
print(f"原始标差: {original_data.std():.4f}")           # 28.7228
print("========================================")
print(f"标准化后数据: {standardized_data}")
# [-1.5666989  -1.21854359 -0.87038828 -0.52223297 -0.17407766 ...]
print(f"标准化后均值: {standardized_data.mean():.4f}")  # 0.0000
print(f"标准化后方差: {standardized_data.var():.4f}")   # 1.0000
print(f"标准化后标差: {standardized_data.std():.4f}")   # 1.0000
