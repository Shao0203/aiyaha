import numpy as np


# 导数 - 中心差分
def numerical_diff(f, x):
    h = 1e-4
    return (f(x+h) - f(x-h)) / (2*h)


# 导数 - 向前差分
def numerical_forward(f, x):
    h = 1e-4
    return (f(x+h) - f(x)) / h


# 偏导数 - 梯度
def numerical_gradient_basic(f, x):
    h = 1e-4
    grad = np.zeros_like(x)
    for idx in range(x.size):
        tmp_val = x[idx]
        x[idx] = tmp_val + h
        fxh1 = f(x)         # 计算f(x+h)
        x[idx] = tmp_val - h
        fxh2 = f(x)         # 计算f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2 * h)
        x[idx] = tmp_val    # 还原值
    return grad


# 梯度法
def gradient_descent(f, x, lr=0.01, step_num=100):
    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x -= lr * grad
    return x


def _numerical_gradient_1d(f, x):
    h = 1e-4  # 0.0001
    grad = np.zeros_like(x)

    for idx in range(x.size):
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x)  # f(x+h)

        x[idx] = tmp_val - h
        fxh2 = f(x)  # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2*h)

        x[idx] = tmp_val  # 还原值

    return grad


def numerical_gradient_2d(f, X):
    if X.ndim == 1:
        return _numerical_gradient_1d(f, X)
    else:
        grad = np.zeros_like(X)

        for idx, x in enumerate(X):
            grad[idx] = _numerical_gradient_1d(f, x)

        return grad


def numerical_gradient(f, x):
    """
    np.nditer: NumPy 提供的高效 N 维数组迭代器
        x: 被迭代的数组（可以是多维）
        flags=['multi_index']: 告诉迭代器在每次循环中维护当前元素的多维索引(tuple),可通过 it.multi_index 访问
        op_flags=['readwrite']: 允许在迭代过程中读写数组元素（因为我们要临时把 x[idx] 改为 tmp_val + h,再改回去）
    流程：
        idx = it.multi_index 得到像 (0, 3, 2) 这样的索引。
        用 x[idx] 访问并修改对应元素。
        调用 f(x)(注意: f 需要接受整个数组 x)。
        恢复原值并移动到下一个元素。
    缺点：
        grads = {
            'W1': np.zeros((784, 50)),    # 39,200个梯度
            'b1': np.zeros(50),           # 50个梯度  
            'W2': np.zeros((50, 10)),     # 500个梯度
            'b2': np.zeros(10)            # 10个梯度
        }
        总计: 39,200 + 50 + 500 + 10 = 39,760个梯度    
        需要: 39,760 × 2 = 79,520 次前向传播！
        每个参数计算2次损失函数
    """
    h = 1e-4  # 0.0001
    grad = np.zeros_like(x)

    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x)  # f(x+h)

        x[idx] = tmp_val - h
        fxh2 = f(x)  # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2*h)

        x[idx] = tmp_val  # 还原值
        it.iternext()

    return grad
