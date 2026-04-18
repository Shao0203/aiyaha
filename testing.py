import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def ReLU(x):
    return np.maximum(0, x)


def tanh(x):
    return np.tanh(x)


x = np.random.randn(1000, 100)
node_num = 100
hidden_layer_size = 5
activations = {}

for i in range(hidden_layer_size):
    if i != 0:
        x = activations[i-1]

    w = np.random.randn(node_num, node_num) * 1
    a = np.dot(x, w)
    z = sigmoid(a)
    activations[i] = z

# plt.figure(figsize=(20, 5))
# for i, a in activations.items():
#     plt.subplot(1, len(activations), i+1)
#     plt.title(f'{i+1}-layer')
#     if i != 0:
#         plt.yticks([])
#     plt.hist(a.flatten(), bins=30, range=(0, 1))
# plt.tight_layout()
# plt.show()

fig, axes = plt.subplots(1, len(activations), figsize=(20, 5))  # 更宽的画布
axes = axes.flatten()
for i, a in activations.items():
    ax = axes[i]
    ax.set_title(f'{i+1}-layer')
    if i != 0:
        ax.set_yticks([])
    ax.hist(a.flatten(), bins=30, range=(0, 1))
plt.tight_layout()
plt.show()
