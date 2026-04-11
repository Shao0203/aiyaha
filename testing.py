import pickle
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
from dataset.mnist import load_mnist


# 1. Activation functions
def identity_func(x):
    return x


def step_func(x):
    return (x > 0).astype(int)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def tanh(x):
    return 2 * sigmoid(2*x) - 1


def relu(x):
    return np.maximum(0, x)


def softmax(x):
    x -= np.max(x, axis=-1, keepdims=True)
    return np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)


# 2. Loss functions
def mean_squared_error(y, t):
    return 0.5 * np.sum((y - t) ** 2, axis=-1)


def mse(y, t):
    return 0.5 * np.mean((y - t) ** 2)


def cross_entropy_error(y, t):
    if y.ndim == 1:
        y = y.reshape(1, y.size)
        t = t.reshape(1, t.size)
    if t.ndim > 1 and y.size == t.size:
        t = np.argmax(t, axis=-1)
    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size
    # return -np.mean(np.log(y[np.arange(batch_size), t] + 1e-7))


# 3. Numerical gradient
def numerical_diff(f, x):
    h = 1e-4
    return (f(x+h) - f(x-h)) / (2 * h)


def numerical_grad(f, x):
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


# 4. Network layers
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
        out = 1 / (1 + np.exp(-x))
        self.out = out
        return out

    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out
        return dx


class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.dW = None
        self.db = None
        self.x = None
        self.x_original_shape = None

    def forward(self, x):
        self.x_original_shape = x.shape
        self.x = x.reshape(x.shape[0], -1)
        out = np.dot(self.x, self.W) + self.b
        return out

    def backward(self, dout):
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        dx = np.dot(dout, self.W.T).reshape(self.x_original_shape)
        return dx


class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None

    def forward(self, x, t):
        self.y = softmax(x)
        self.t = t
        self.loss = cross_entropy_error(self.y, self.t)
        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        dx = dout * (self.y - self.t) / batch_size
        return dx


# 5. Network Architecture
class TwoLayerNet:
    def __init__(self, input, hidden, output, weight_init_std=0.01):
        self.params = {}
        self.params['W1'] = np.random.randn(input, hidden) * weight_init_std
        self.params['b1'] = np.zeros(hidden)
        self.params['W2'] = np.random.randn(hidden, output) * weight_init_std
        self.params['b2'] = np.zeros(output)

        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])
        self.lastLayer = SoftmaxWithLoss()

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    def loss(self, x, t):
        y = self.predict(x)
        loss = self.lastLayer.forward(y, t)
        return loss

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=-1)
        if t.ndim != 1:
            t = np.argmax(t, axis=-1)
        accuracy = np.sum(y == t) / x.shape[0]
        return accuracy

    def numerical_gradient(self, x, t):
        def loss_W(W): return self.loss(x, t)
        grads = {}
        grads['W1'] = numerical_grad(loss_W, self.params['W1'])
        grads['b1'] = numerical_grad(loss_W, self.params['b1'])
        grads['W2'] = numerical_grad(loss_W, self.params['W2'])
        grads['b2'] = numerical_grad(loss_W, self.params['b2'])
        return grads

    def gradient(self, x, t):
        # forward
        self.loss(x, t)
        # backward
        dout = self.lastLayer.backward()
        layers = list(self.layers.values())
        layers.reverse()    # !!! Dont forget!!!
        for layer in layers:
            dout = layer.backward(dout)
        # gradient
        grads = {}
        grads['W1'] = self.layers['Affine1'].dW
        grads['b1'] = self.layers['Affine1'].db
        grads['W2'] = self.layers['Affine2'].dW
        grads['b2'] = self.layers['Affine2'].db
        return grads

    def save_model(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(self.params, f)
        print(f'Model is saved to {filepath}.')

    def load_model(self, filepath):
        with open(filepath, 'rb') as f:
            self.params = pickle.load(f)
        print(f'Model is loaded from {filepath}.')
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])


# 6. Gradient check
# network = TwoLayerNet(784, 50, 10)
# (x_train, t_train), (x_test, t_test) = load_mnist(one_hot_label=True)
# x_batch = x_train[:3]
# t_batch = t_train[:3]
# nume_grad = network.numerical_gradient(x_batch, t_batch)
# back_grad = network.gradient(x_batch, t_batch)
# for key in nume_grad.keys():
#     diff = np.average(np.abs(nume_grad[key] - back_grad[key]))
#     print(f'{key}: {diff}')


# 7. Mnist dataset training process
network = TwoLayerNet(784, 50, 10)
(x_train, t_train), (x_test, t_test) = load_mnist(one_hot_label=True)
print(x_train.shape)
print(t_train.shape)
print(t_train.ndim)
print(t_train[-2:])
t_train = np.argmax(t_train)
print(t_train.shape)
print(t_train.ndim)
print(t_train)
