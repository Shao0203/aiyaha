import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
from dataset.mnist import load_mnist
from common.optimizer import SGD, Momentum, AdaGrad, Adam


def softmax(x):
    x -= np.max(x, axis=-1, keepdims=True)
    return np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)


def cross_entropy_error(y, t):
    if y.ndim == 1:
        y = y.reshape(1, y.size)
        t = t.reshape(1, t.size)
    if t.ndim > 1 and t.size == y.size:
        t = np.argmax(t, axis=-1)
    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size


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
        return dout


class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None
        self.oxs = None

    def forward(self, x):
        self.oxs = x.shape
        self.x = x.reshape(x.shape[0], -1)
        out = np.dot(self.x, self.W) + self.b
        return out

    def backward(self, dout):
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        dx = np.dot(dout, self.W.T).reshape(self.oxs)
        return dx


class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        return cross_entropy_error(self.y, self.t)

    def backward(self, dout=1):
        dx = dout * (self.y - self.t) / self.t.shape[0]
        return dx


class TwoLayerNet:
    def __init__(self, input, hidden, output, weight_std=0.01):
        self.params = {}
        self.params['W1'] = weight_std * np.random.randn(input, hidden)
        self.params['b1'] = np.zeros(hidden)
        self.params['W2'] = weight_std * np.random.randn(hidden, output)
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
        return np.sum(y == t) / x.shape[0]

    def nume_grad(self, x, t):
        def loss_W(W): return self.loss(x, t)
        grads = {
            'W1': numerical_gradient(loss_W, self.params['W1']),
            'b1': numerical_gradient(loss_W, self.params['b1']),
            'W2': numerical_gradient(loss_W, self.params['W2']),
            'b2': numerical_gradient(loss_W, self.params['b2']),
        }
        return grads

    def back_grad(self, x, t):
        self.loss(x, t)
        dout = self.lastLayer.backward()
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)
        grads = {
            'W1': self.layers['Affine1'].dW,
            'b1': self.layers['Affine1'].db,
            'W2': self.layers['Affine2'].dW,
            'b2': self.layers['Affine2'].db,
        }
        return grads


(x_train, t_train), (x_test, t_test) = load_mnist(one_hot_label=True)
iters = 10000
train_size = x_train.shape[0]
batch_size = 100
iter_per_epoch = max(1, train_size // batch_size)
networks, train_loss_list, train_acc_list, test_acc_list = {}, {}, {}, {}


optimizers = {'SGD': SGD(), 'Momentum': Momentum(), 'AdaGrad': AdaGrad(), 'Adam': Adam()}
for key in optimizers:
    networks[key] = TwoLayerNet(784, 50, 10)
    train_loss_list[key], train_acc_list[key], test_acc_list[key] = [], [], []

for i in range(iters):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    for key, optimizer in optimizers.items():
        grads = networks[key].back_grad(x_batch, t_batch)
        optimizer.update(networks[key].params, grads)
        loss = networks[key].loss(x_batch, t_batch)
        train_loss_list[key].append(loss)

        if i % iter_per_epoch == 0:
            train_acc = networks[key].accuracy(x_train, t_train)
            test_acc = networks[key].accuracy(x_test, t_test)
            train_acc_list[key].append(train_acc)
            test_acc_list[key].append(test_acc)
            print(f'Epoch {i//iter_per_epoch}: {train_acc:.4f} | {test_acc:.4f} ({key})')
            if key == 'Adam':
                print('================================')


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
# ax1 shows loss value while ax2 shows test accuracy
x_loss = np.arange(len(train_loss_list[key]))
x_acc = np.arange(len(test_acc_list[key]))
for key in optimizers:
    ax1.plot(x_loss, train_loss_list[key], label=key)
    ax2.plot(x_acc, test_acc_list[key], label=key)
ax1.set(xlabel='Iteration', ylabel='Loss value', title='Training loss')
ax1.grid(True, alpha=0.5, ls='--')
ax1.legend(loc='upper right')
ax2.set(xlabel='Epoch', ylabel='Accuracy', title='Test Accuracy', ylim=(0, 1))
ax2.grid(True, alpha=0.5, ls='-')
ax2.legend(loc='lower right')
fig.suptitle('Compare loss and accuracy between optimizers.')
plt.tight_layout()
plt.show()
