import matplotlib.pyplot as plt
import numpy as np


# fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(10, 8))

# axes[0, 0].plot([1, 2, 3], [1, 2, 1])  # 左上
# axes[0, 0].set_title('left up')

# axes[0, 1].scatter([1, 2, 3], [1, 4, 2])  # 右上
# axes[0, 1].set_title('mid up')

# axes[0, 2].scatter([1, 2, 3], [1, 4, 2])  # 右上
# axes[0, 2].set_title('right up')

# axes[1, 0].bar(['A', 'B', 'C'], [3, 7, 2])  # 左下
# axes[1, 0].set_title('left down')

# axes[1, 1].hist([1, 2, 2, 3, 3, 3, 4, 4, 5])  # 右下
# axes[1, 1].set_title('mid down')

# axes[1, 2].hist([1, 2, 2, 3, 3, 3, 4, 4, 5])  # 右下
# axes[1, 2].set_title('right down')

# fig.suptitle('Learn subplots', fontsize=16)

# for i, ax in enumerate(axes.flat):
#     ax.plot([i, i+1, i+2])
#     ax.set_title(f'subFig {i+1}')

# plt.tight_layout()

# axes[1, 2].axis('off')
# plt.show()


# # 创建数据
# x = np.linspace(0, 2*np.pi, 100)
# y1 = np.sin(x)
# y2 = np.cos(x)
# y3 = np.tan(x)
# y4 = np.exp(-x) * np.sin(x)

# # 创建2x2子图
# fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)

# # 在各个子图上绘图
# axes[0, 0].plot(x, y1, 'r-', linewidth=2)
# axes[0, 0].set_title('Sin')
# axes[0, 0].grid(True)

# axes[0, 1].plot(x, y2, 'b-', linewidth=2)
# axes[0, 1].set_title('cos')
# axes[0, 1].grid(True)

# axes[1, 0].plot(x, y3, 'g-', linewidth=2)
# axes[1, 0].set_title('tan')
# axes[1, 0].set_ylim(-5, 5)
# axes[1, 0].grid(True)

# axes[1, 1].plot(x, y4, 'm-', linewidth=2)
# axes[1, 1].set_title('decay')
# axes[1, 1].grid(True)

# # 添加总标题和调整布局
# fig.suptitle('triangle functions', fontsize=16)
# plt.tight_layout(rect=[0, 0, 1, 0.96])  # 为总标题留出空间
# plt.show()


# # 比较多个激活函数
# fig, axes = plt.subplots(2, 2, figsize=(10, 8))

# x = np.arange(-5, 5, 0.1)

# # Sigmoid
# axes[0, 0].plot(x, 1/(1+np.exp(-x)))
# axes[0, 0].set_title('Sigmoid')
# axes[0, 0].grid(True)

# # ReLU
# axes[0, 1].plot(x, np.maximum(0, x))
# axes[0, 1].set_title('ReLU')
# axes[0, 1].grid(True)

# # Tanh
# axes[1, 0].plot(x, np.tanh(x))
# axes[1, 0].set_title('Tanh')
# axes[1, 0].grid(True)

# # Leaky ReLU
# axes[1, 1].plot(x, np.where(x > 0, x, 0.01*x))
# axes[1, 1].set_title('Leaky ReLU')
# axes[1, 1].grid(True)

# plt.tight_layout()
# plt.show()


# # 训练过程中的损失和准确率对比
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# # 损失函数曲线
# ax1.plot(history['loss'], label='Training Loss')
# ax1.plot(history['val_loss'], label='Validation Loss')
# ax1.set_title('Loss over Epochs')
# ax1.legend()

# # 准确率曲线
# ax2.plot(history['accuracy'], label='Training Accuracy')
# ax2.plot(history['val_accuracy'], label='Validation Accuracy')
# ax2.set_title('Accuracy over Epochs')
# ax2.legend()

# plt.tight_layout()
# plt.show()


fig, axes = plt.subplots(3, 1, figsize=(5, 10))

for ax in axes:
    ax.plot([1, 2, 3])

plt.show()
