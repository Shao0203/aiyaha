import numpy as np
from common.functions import *


class Relu:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0
        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout
        return dx


class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):
        out = sigmoid(x)
        self.out = out
        return out

    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out
        return dx


class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None
        self.original_x_shape = None  # 记录原始形状

    def forward(self, x):
        self.original_x_shape = x.shape  # 记录原始形状（用于处理非矩阵输入）
        self.x = x.reshape(x.shape[0], -1)  # 展平为矩阵
        out = np.dot(self.x, self.W) + self.b
        return out

    def backward(self, dout):
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        dx = np.dot(dout, self.W.T)
        dx = dx.reshape(self.original_x_shape)  # 恢复原始形状
        return dx


class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None    # 用于监控/调试/保存
        self.y = None       # softmax的输出，反向传播必需
        self.t = None       # 监督数据标签，反向传播必需

    def forward(self, x, t):
        # 这里接收的参数x为Affine层forward方法返回的加权输入总和，
        # 返回的self.y为softmax激活后的百分比数组，比如真值是2，[0.03, 0.003, 0.95, 0.01,...]
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)
        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        if self.t.size == self.y.size:  # 监督数据t是one-hot-vector的情况 [0,0,1,0,0]
            dx = (self.y - self.t) / batch_size
        else:                           # 监督数据t是整数标签的情况[2, 5, 8, 0, 3]
            dx = self.y.copy()
            dx[np.arange(batch_size), self.t] -= 1  # self.y中 对应的真实值位置 -1，
            dx /= batch_size
        return dx
