import numpy as np


# --- Activation function
def step_function(x):
    return (x > 0).astype(int)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def relu(x):
    return np.maximum(0, x)


def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))


def tanh(x):
    return 2 * sigmoid(2*x) - 1


# --- Output layer
def identity_function(x):
    return x


def softmax(x):
    """
    keepdims=True 保持维度结构，确保广播正确
    x = np.array([[1, 2, 3], [4, 5, 6]]) # 形状: (2,3)
    np.max(x, axis=-1)                   # 形状: (2,)  ← 无法广播 [3, 6] 
    np.max(x, axis=-1, keepdims=True)    # 形状: (2,1) ← 完美广播 [[3], [6]] 
    """
    x = x - np.max(x, axis=-1, keepdims=True)
    return np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)
    # # 书中写法: 转换x后，按每列操作（每列是一条数据）
    # if x.ndim == 2:
    #     x = x.T
    #     x = x - np.max(x, axis=0)
    #     y = np.exp(x) / np.sum(np.exp(x), axis=0)
    #     return y.T
    # x = x - np.max(x)  # 溢出对策
    # return np.exp(x) / np.sum(np.exp(x))


# --- Loss function
def mean_squared_error(y, t):
    return 0.5 * np.sum((y - t) ** 2)


def cross_entropy_error(y, t):
    """
    处理单条数据, 转换给二维 y.ndim=2, y.shape=(1, 5)
    上面y, 下面reshape后      y.shape  y.ndim
    [0.1, 0.1, 0.6, 0, 0.2]   (5,)    1
    [[0.1, 0.1, 0.6, 0, 0.2]] (1, 5)  2
    """
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    # 监督数据是one-hot-vector的情况下，转换为正确解标签的索引
    if t.ndim > 1 and t.size == y.size:
        t = t.argmax(axis=1)
    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size


def sigmoid_grad(x):
    return (1.0 - sigmoid(x)) * sigmoid(x)


def relu_grad(x):
    grad = np.zeros(x)
    grad[x >= 0] = 1
    return grad


def softmax_loss(X, t):
    y = softmax(X)
    return cross_entropy_error(y, t)
