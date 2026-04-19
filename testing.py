import numpy as np
import matplotlib.pyplot as plt
from common.functions import sigmoid, relu, tanh


x = np.random.randn(1000, 100)
node_num = 100
hidden_layer_size = 5
activations = {}

for i in range(hidden_layer_size):
    if i != 0:
        x = activations[i-1]
    w = np.random.randn(node_num, node_num) * 1  # 标准差为1
    w = np.random.randn(node_num, node_num) * 0.01  # 标准差为0.01
    w = np.random.randn(node_num, node_num) * np.sqrt(1.0 / node_num)  # 标准差为Xavier
    w = np.random.randn(node_num, node_num) * np.sqrt(2.0 / node_num)  # 标准差为He
    z = np.dot(x, w)

    a = sigmoid(x)  # 激活函数sigmoid
    a = tanh(x)     # 激活函数tanh
    a = relu(z)     # 激活函数relu
    activations[i] = a

# 绘制直方图
fig, axes = plt.subplots(1, len(activations), figsize=(20, 5))
for i, a in activations.items():
    ax = axes[i]
    ax.set(title=f'{i+1}-layer', ylim=(0, 7000))
    if i != 0:
        ax.set_yticks([])
    ax.hist(a.flatten(), bins=30, range=(0, 1))
plt.tight_layout()
plt.show()
