import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
import pickle


# 公式的直接转换 只能处理输入x是单个值的情况
def single_step(x):
    if x > 0:
        return 1
    else:
        return 0


def single_softmax(x):
    x -= np.max(x)
    return np.exp(x) / np.sum(np.exp(x))


def single_cross_entropy_error(y, t):
    delta = 1e-7
    return -np.sum(t * np.log(y + delta))


# 1. 先定义激活函数:
def identity_function(x):
    return x


def step(x):
    return (x > 0).astype(int)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def relu(x):
    return np.maximum(0, x)


def softmax(x):
    """
    keepdims=True 保持维度结构，确保广播正确
    x = np.array([[1, 2, 3], [4, 5, 6]]) # 形状: (2,3)
    np.max(x, axis=-1)                   # 形状: (2,)  ← 无法广播 [3, 6] 
    np.max(x, axis=-1, keepdims=True)    # 形状: (2,1) ← 完美广播 [[3], [6]] 
    """
    x -= np.max(x, axis=-1, keepdims=True)
    return np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)


# 2. 定义损失函数
def mean_squared_error(y, t):
    return 0.5 * np.sum((y - t) ** 2, axis=-1)


def mse(y, t):
    return 0.5 * np.mean((y - t) ** 2)


def cross_entropy_error(y, t):
    # 统一输入格式,处理单条数据,转换给二维 y.ndim=2，y.shape=(1, 10)
    if y.ndim == 1:
        y = y.reshape(1, y.size)
        t = t.reshape(1, t.size)
    # 如果t是one-hot-vector，转换成标签形式
    if t.size == y.size:
        t = t.argmax(axis=-1)
    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size


# 定义导数计算 数值微分法
def numerical_diff(f, x):
    h = 1e-4
    return (f(x+h) - f(x-h)) / (2*h)


def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)

    for idx in range(x.size):
        tmp_val = x[idx]
        x[idx] = tmp_val + h
        fxh1 = f(x)  # f(x+h)
        x[idx] = tmp_val - h
        fxh2 = f(x)  # f(x-h)

        grad[idx] = (fxh1 - fxh2) / (2*h)
        x[idx] = tmp_val

    return grad


# 再定义神经网络层 - todo
# 利用层 创建神经网络 - todo
# 训练 / 测试 神经网络 - todo
y1 = np.array([0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0])
y2 = np.array([0.1, 0.05, 0.1, 0.0, 0.05, 0.0, 0.0, 0.7, 0.0, 0.0])
t1 = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
t2 = np.array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0])
tt = np.array([[0, 0, 1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]])
yy = np.array([[0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0], [
              0.1, 0.05, 0.1, 0.0, 0.05, 0.0, 0.0, 0.7, 0.0, 0.0]])
tt_labels = np.array([2, 7])


"""画图比较三种激活函数 Step/Sigmoid/ReLU 
x = np.arange(-5, 5, 0.1)
y1, y2, y3 = step(x), sigmoid(x), relu(x)
plt.plot(x, y1, label='Step')
plt.plot(x, y2, label='Sigmoid')
plt.plot(x, y3, label='ReLU')
plt.xlabel('X')
plt.ylabel('Y')
plt.ylim(-0.1, 2.1)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.show()
"""


"""最简单的流程化3层神经网络的实现 - 向前传播
X = np.array([1.0, 0.5])

W1 = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
B1 = np.array([0.1, 0.2, 0.3])
A1 = np.dot(X, W1) + B1
Z1 = sigmoid(A1)

W2 = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
B2 = np.array([0.1, 0.2])
A2 = np.dot(Z1, W2) + B2
Z2 = sigmoid(A2)

W3 = np.array([[0.1, 0.3], [0.2, 0.4]])
B3 = np.array([0.1, 0.2])
A3 = np.dot(Z2, W3) + B3
Y = identity_function(A3)

print(A1, Z1)
print(A2, Z2)
print(A3, Y)


def init_network():
    network = {}
    network['W1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    network['b1'] = np.array([0.1, 0.2, 0.3])
    network['W2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    network['b2'] = np.array([0.1, 0.2])
    network['W3'] = np.array([[0.1, 0.3], [0.2, 0.4]])
    network['b3'] = np.array([0.1, 0.2])
    return network


def forward(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = identity_function(a3)
    return y


network = init_network()
x = np.array([1.0, 0.5])
y = forward(network, x)
print(y)
"""


"""Mnist数据集 画出第一个数据 - 数字5
(x_train, t_train), (x_test, t_test) = load_mnist()

img = x_train[0].reshape(28, 28)
label = t_train[0]
plt.imshow(img, cmap='gray')
plt.title(f'Label: {label}')
plt.axis('off')
plt.show()    
"""


"""Mnist数据集 使用测试数据 查看训练好的权重参数 的预测准确率
def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist()
    return x_test, t_test


def init_network():
    with open('dataset/sample_weight.pkl', 'rb') as f:
        network = pickle.load(f)
    return network


def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax_single(a3)     # 这里softmax只能处理一条数据
    return y


# # 每次预测一条数据 = 1 / 10000
# x, t = get_data()
# network = init_network()
# accuracy_cnt = 0
# for i in range(len(x)):
#     y = predict(network, x[i])  # 这里的x[i]是10000条测试数据中的一条
#     if np.argmax(y) == t[i]:
#         accuracy_cnt += 1
# print(f'Accuracy: {accuracy_cnt / len(x):.2%}')     # 0.9352

# # 每次预测一批数据 = 100 / 10000
x, t = get_data()
network = init_network()
accuracy_cnt = 0
batch_size = 100
for i in range(0, len(x), batch_size):
    x_batch = x[i:i+batch_size]
    y_batch = predict(network, x_batch) # 这里softmax预测的概率是错误的 因为只能处理单条数据
    p = np.argmax(y_batch, axis=1)      # 这里获得的预测结果的 最大值相对位置是正确的
    accuracy_cnt += np.sum(p == t[i:i+batch_size])
print(f'Accuracy: {accuracy_cnt / len(x):.2%} \n')

# # 查看前10条数据的预测和真值对比情况
p = np.argmax(predict(network, x[:10]), 1)
print(p[:10])   # [7 2 1 0 4 1 4 9 6 9]
print(t[:10])   # [7 2 1 0 4 1 4 9 5 9]
print(p[:10] == t[:10]) # [ True  True  True  True  True  True  True  True False  True]
print(np.sum(p[:10] == t[:10])) # 9
"""

"""Mini batch
(x_train, t_train), (x_test, t_test) = load_mnist(
    flatten=True, normalize=True, one_hot_label=True)

train_size = x_train.shape[0]
batch_size = 100

batch_mask = np.random.choice(train_size, batch_size)
x_batch = x_train[batch_mask]
t_batch = t_train[batch_mask]
print('batch_mask:')
print(batch_mask)
print('x_batch')
print(x_batch)
print('t_batch')
print(t_batch)
"""
