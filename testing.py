import pickle
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
from dataset.mnist import load_mnist


# 1. Activation functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def tanh(x):
    return 2 * sigmoid(2*x) - 1


def relu(x):
    return np.maximum(0, x)


def softmax(x):
    x -= np.max(x, axis=-1, keepdims=True)
    return np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)


# 2. Loss function
def cross_entropy_error(y, t):
    if y.ndim == 1:
        y = y.reshape(1, y.size)
        t = t.reshape(1, t.size)

    if t.ndim > 1 and t.size == y.size:
        t = t.argmax(axis=-1)

    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size


# 3. calc gradient
def numerical_diff(f, x):
    h = 1e-4
    return (f(x+h) - f(x-h)) / (2*h)


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


# 4. define network layers
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


class Tanh:
    def __init__(self):
        self.out = None

    def forward(self, x):
        out = np.tanh(x)
        self.out = out
        return out

    def backward(self, dout):
        dx = dout * (1 - self.out ** 2)
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
        dx = np.dot(dout, self.W.T)
        dx = dx.reshape(self.original_x_shape)
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
        batch_size = self.t.shape[0]
        dx = dout * (self.y - self.t) / batch_size
        return dx


# 5. create neural network
class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        self.params = {}
        self.params['W1'] = np.random.randn(
            input_size, hidden_size) * weight_init_std
        self.params['W2'] = np.random.randn(
            hidden_size, output_size) * weight_init_std
        self.params['b1'] = np.zeros(hidden_size)
        self.params['b2'] = np.zeros(output_size)

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
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])
        return grads

    def gradient(self, x, t):
        self.loss(x, t)

        dout = self.lastLayer.backward()
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        grads = {}
        grads['W1'] = self.layers['Affine1'].dW
        grads['b1'] = self.layers['Affine1'].db
        grads['W2'] = self.layers['Affine2'].dW
        grads['b2'] = self.layers['Affine2'].db
        return grads

    def save_model(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(self.params, f)
        print(f'Model params is stored into: {filepath}')

    def load_model(self, filepath):
        with open(filepath, 'rb') as f:
            self.params = pickle.load(f)
        print(f'Model is loaded from {filepath}')
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])


# 6. gradient comfirmation
# (x_train, t_train), (x_test, t_test) = load_mnist(one_hot_label=True)
# network = TwoLayerNet(784, 50, 10)
# x_batch = x_train[:3]
# t_batch = t_train[:3]

# grad_numerical = network.numerical_gradient(x_batch, t_batch)
# grad_backpropa = network.gradient(x_batch, t_batch)
# for key in grad_numerical.keys():
#     diff = np.mean(np.abs(grad_numerical[key] - grad_backpropa[key]))
#     print(f'{key}: {str(diff)}')
#     # W1: 3.96350709099096e-10
#     # b1: 2.171388526952937e-09
#     # W2: 5.1182810376073035e-09
#     # b2: 1.3960855106825987e-07


# # 7. train and test network
# (x_train, t_train), (x_test, t_test) = load_mnist(one_hot_label=True)
# network = TwoLayerNet(784, 50, 10)
# iters_num = 10000
# train_size = x_train.shape[0]
# batch_size = 128
# iter_per_epoch = max(round(train_size / batch_size), 1)
# learning_rate = 0.1

# train_loss_list = []
# train_acc_list = []
# test_acc_list = []

# for i in range(iters_num):
#     batch_mask = np.random.choice(train_size, batch_size)
#     x_batch = x_train[batch_mask]
#     t_batch = t_train[batch_mask]

#     gradient = network.gradient(x_batch, t_batch)

#     for key in ('W1', 'b1', 'W2', 'b2'):
#         network.params[key] -= learning_rate * gradient[key]

#     loss = network.loss(x_batch, t_batch)
#     train_loss_list.append(loss)

#     if i % iter_per_epoch == 0:
#         train_acc = network.accuracy(x_train, t_train)
#         test_acc = network.accuracy(x_test, t_test)
#         train_acc_list.append(train_acc)
#         test_acc_list.append(test_acc)
#         print(f'Epoch {i // iter_per_epoch}: {train_acc:.2%}, {test_acc:.2%}')


# # 8. draw the loss / accuracy
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
# ax1.plot(np.arange(iters_num), train_loss_list)
# ax1.set_xlabel('Iterations')
# ax1.set_ylabel('Loss')
# ax1.set_title('Train loss value')
# ax1.grid(True, linestyle='--', alpha=0.5)

# x_acc = np.arange(len(train_acc_list))
# ax2.plot(x_acc, train_acc_list, label='Train Acc')
# ax2.plot(x_acc, test_acc_list, label='Test Acc', linestyle='--')
# ax2.set_xlabel('Epoch')
# ax2.set_ylabel('Accuracy')
# ax2.set_title('Train / Test Accuracy')
# ax2.grid(True, linestyle='--', alpha=0.5)
# ax2.legend(loc='lower right')

# plt.suptitle('Monitor learning process')
# plt.tight_layout()
# plt.show()

# network.save_model('trained_model.pkl')


# 9. use the trained_model.pkl
(x_train, t_train), (x_test, t_test) = load_mnist(one_hot_label=True)
new_network = TwoLayerNet(784, 50, 10)
new_network.load_model('trained_model.pkl')
train_acc = new_network.accuracy(x_train, t_train)
test_acc = new_network.accuracy(x_test, t_test)
print(f'train_acc, test_acc | {train_acc:.2%}, {test_acc:.2%}')
# Model is loaded from trained_model.pkl
# train_acc, test_acc | 98.16%, 97.11%
