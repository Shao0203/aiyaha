import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
import pickle


# 1. 先定义激活函数:
def identity_function(x):
    return x


def step_single(x):
    if x > 0:
        return 1
    else:
        return 0


def softmax_single(x):
    x -= np.max(x)
    return np.exp(x) / np.sum(np.exp(x))


# 这里step, sigmoid, relu, softmax 都能自动处理多维数组
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


a = np.array([0.3, 2.9, 4.0])
print(np.sum(softmax_single(a)))
aa = np.array([[0.3, 2.9, 4.0], [-0.9, 1, 3], [3, 2, 1]])
x3 = np.random.randn(2, 3, 4)  # 2个批次，每个3个样本，每个4个类别
# print(softmax(a))
# print(softmax(aa))
# print(x3)
# print(softmax(x3))
# print("\n三维输入形状:", softmax(x3).shape)  # (2, 3, 4)
# print("最后一个维度总和:", np.sum(softmax(x3), axis=-1))


# 2. 再定义神经网络层 - todo


# 3. 利用层 创建神经网络 - todo


# 4. 训练 / 测试 神经网络 - todo


"""画图比较三种激活函数 Step/Sigmoid/ReLU 
x = np.arange(-5, 5, 0.1)
y1 = step(x)
y2 = sigmoid(x)
y3 = relu(x)
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
    y = softmax(a3)     # 这里softmax只能处理一条数据
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
    y_batch = predict(network, x_batch) # 这里softmax预测的概率是错误的
    p = np.argmax(y_batch, axis=1)      # 这里获得的预测结果相对位置是正确的
    accuracy_cnt += np.sum(p == t[i:i+batch_size])
print(f'Accuracy: {accuracy_cnt / len(x):.2%} \n')
# # only look at the first 10 data
p = np.argmax(predict(network, x[:10]), 1)
print(p[:10])   # [7 2 1 0 4 1 4 9 6 9]
print(t[:10])   # [7 2 1 0 4 1 4 9 5 9]
print(p[:10] == t[:10]) # [ True  True  True  True  True  True  True  True False  True]
print(np.sum(p[:10] == t[:10])) # 9
"""
