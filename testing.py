import numpy as np


# 1. Activation & Loss & num_grad
def relu(x):
    return np.minimum(0, x)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def step(x):
    return (x > 0).astype(int)


def tanh(x):
    return np.tanh(x)


def softmax(x):
    x = x - np.max(x, axis=-1, keepdims=True)
    return np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)


def cross_entropy_error(y, t):
    if y.ndim == 1:
        y = y.reshape(1, y.size)
        t = t.reshape(1, t.size)
    if t.ndim > 1 and t.size == y.size:
        t = np.argmax(t, axis=-1)
    batch_size = y.shape[0]
    return -np.mean(np.log(y[np.arange(batch_size), t] + 1e-7))


def num_diff(f, x):
    h = 1e-4
    return f(x+h) - f(x-h) / (2*h)


def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp = x[idx]
        x[idx] = tmp + h
        fxh1 = f(x)
        x[idx] = tmp - h
        fxh2 = f(x)
        grad[idx] = (fxh1 - fxh2) / (2 * h)
        x[idx] = tmp
        it.iternext()
    return grad


# 2. Layers
class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):
        self.out = 1 / (1 + np.exp(-x))
        return self.out

    def backward(self, dout):
        dx = dout * (1 - self.out) * self.out
        return dx


class Tanh:
    def __init__(self):
        self.out = None

    def forward(self, x):
        self.out = np.tanh(x)
        return self.out

    def backward(self, dout):
        dx = dout * (1 - self.out ** 2)
        return dx


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


class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None
        self.original_x_shape = None

    def forward(self, x):
        self.original_x_shape = x.shape
        self.x = x.reshape(x.shape[0], -1)
        out = np.dot(self.x, self.W) + self.b
        return out

    def backward(self, dout):
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        dx = np.dot(dout, self.W.T).reshape(self.original_x_shape)
        return dx


class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)
        return self.loss

    def backward(self, dout=1):
        dx = dout * (self.y - self.t) / self.t.shape[0]
        return dx


# 3. Neural Network
class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_std=0.01):
        pass
