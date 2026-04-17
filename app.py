import pickle
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
from dataset.mnist import load_mnist


# 1. 定义激活函数:
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
    if t.ndim > 1 and t.size == y.size:
        t = np.argmax(t, axis=-1)
    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size
    # return -np.mean(np.log(y[np.arange(batch_size), t] + 1e-7)) # 也可以这么写


# 3. 定义导数计算 数值微分法
def numerical_diff(f, x):   # 导数
    h = 1e-4
    return (f(x+h) - f(x-h)) / (2 * h)


def numerical_grad(f, x):   # 偏导数
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


# 4. 定义神经网络层 Relu, Sigmoid, Affine, SoftmaxWithLoss
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
        self.W = W      # 正向用self.W, 反向self.W.T
        self.b = b      # 正向用self.b
        self.x = None   # 用于反向时x.T
        self.dW = None  # 保存权重梯度，神经网络求梯度时从这取
        self.db = None  # 保存偏置梯度，神经网络求梯度时从这取
        self.original_x_shape = None    # 记录x原始形状

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
        out = self.loss
        return out

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        dx = dout * (self.y - self.t) / batch_size
        return dx


# 5. 创建神经网络使用Relu, Affine, SoftmaxWithLoss层
class TwoLayerNet:

    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        # init weights and bias
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)
        # init layers
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
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)
        # gradient
        grads = {}
        grads['W1'], grads['b1'] = self.layers['Affine1'].dW, self.layers['Affine1'].db
        grads['W2'], grads['b2'] = self.layers['Affine2'].dW, self.layers['Affine2'].db
        return grads

    def save_model(self, filepath):
        # 保存模型参数到文件
        with open(filepath, 'wb') as f:
            pickle.dump(self.params, f)
        print(f"模型已保存到: {filepath}")

    def load_model(self, filepath):
        # 从文件加载模型参数
        with open(filepath, 'rb') as f:
            self.params = pickle.load(f)
        print(f"模型已从 {filepath} 加载")
        # 重新初始化网络层，使用加载的参数
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])


"""
        # # 使用示例：
        # # 训练完成后保存模型
        # network = TwoLayerNet(784, 50, 10)
        # # ... 训练代码
        # network.save_model('trained_model.pkl')

        # # 之后可以加载模型进行推理
        # new_network = TwoLayerNet(784, 50, 10)  # 创建相同结构的网络
        # new_network.load_model('trained_model.pkl')

        # # 现在可以直接使用训练好的模型进行预测
        # test_acc = new_network.accuracy(x_test, t_test)
        # print(f"加载模型的测试准确率: {test_acc:.2%}")


# 6. 梯度确认 gradient check
# (x_train, t_train), (x_test, t_test) = load_mnist(one_hot_label=True)
# network = TwoLayerNet(784, 50, 10)
# x_batch = x_train[:3]
# t_batch = t_train[:3]
# grad_numerical = network.numerical_gradient(x_batch, t_batch)
# grad_backpropa = network.gradient(x_batch, t_batch)
# for key in grad_numerical.keys(): # 求各个权重的绝对误差的平均值
#     diff = np.average(np.abs(grad_numerical[key] - grad_backpropa[key]))
#     print(f'{key}: {str(diff)}')
#     # W1: 3.830806987909411e-10
#     # b1: 2.2425444114388455e-09
#     # W2: 5.6065134847026336e-09
#     # b2: 1.3968564501476432e-07
"""


# 6. 优化器 optimizer
class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr

    def update(self, params, grads):
        for key in params.keys():
            params[key] -= self.lr * grads[key]


class Momentum:
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None

    def update(self, params, grads):
        if self.v is None:
            self.v = {key: np.zeros_like(val) for key, val in params.items()}

        for key in params.keys():
            self.v[key] = self.momentum * self.v[key] - self.lr * grads[key]
            params[key] += self.v[key]


class AdaGrad:
    def __init__(self, lr=0.01):
        self.lr = lr
        self.h = None

    def update(self, params, grads):
        if self.h is None:
            self.h = {key: np.zeros_like(val) for key, val in params.items()}

        for key in params.keys():
            self.h[key] += grads[key]**2
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7)


class Adam:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999):
        self.lr = lr
        self.iter = 0
        self.beta1 = beta1
        self.beta2 = beta2
        self.m = None
        self.v = None

    def update(self, params, grads):
        if self.m is None:
            self.m = {key: np.zeros_like(val) for key, val in params.items()}
            self.v = {key: np.zeros_like(val) for key, val in params.items()}

        self.iter += 1
        lr_t = self.lr * np.sqrt(1.0 - self.beta2**self.iter) / (1.0 - self.beta1**self.iter)

        for key in params.keys():
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grads[key]
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * grads[key]**2
            params[key] -= lr_t * self.m[key] / (np.sqrt(self.v[key]) + 1e-7)


# 7. 使用mnist数据，训练/测试 神经网络
network = TwoLayerNet(784, 50, 10)
optimizers = [SGD(), Momentum(), AdaGrad(), Adam()]
optimizer = optimizers[-1]

(x_train, t_train), (x_test, t_test) = load_mnist(one_hot_label=True)
iters_num, learning_rate = 10000, 0.1,
train_size, batch_size = x_train.shape[0], 100
iter_per_epoch = max(1, train_size // batch_size)
train_loss_list, train_acc_list, test_acc_list = [], [], []

for i in range(iters_num):
    # (1) Get mini batch
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    # (2) Calc gradient
    grads = network.gradient(x_batch, t_batch)
    # (3) Update network params
    # for key in grads.keys():
    #     network.params[key] -= learning_rate * grads[key]
    optimizer.update(network.params, grads)
    # (4) Calc training loss
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)
    # (5) Calc training accuracy
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print(f'Epoch{int(i / iter_per_epoch)}: {train_acc:.4f} | {test_acc:.4f}')


# 8. 绘制损失函数和训练/测试准确率的图形
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
# 左子图：损失函数
x_loss = np.arange(len(train_loss_list))
ax1.plot(x_loss, train_loss_list)
ax1.set(xlabel='Iteration', ylabel='Loss', title='Training loss')
ax1.grid(True, ls='--', alpha=0.5)
# 右子图：准确率
x_acc = np.arange(len(train_acc_list))
ax2.plot(x_acc, train_acc_list, label='Train Accuracy')
ax2.plot(x_acc, test_acc_list, label='Test Accuracy', ls=':')
ax2.set(xlabel='Epoch', ylabel='Accuracy', title='Training accuracy')
ax2.set_ylim(0, 1.0)
ax2.grid(True, ls='--', alpha=0.5)
ax2.legend(loc='lower right')
# 总标题
fig.suptitle('Monitor model training process', fontsize=14)
plt.tight_layout()
plt.show()
