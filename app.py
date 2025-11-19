import pickle
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
from dataset.mnist import load_mnist


"""复习3-5章节全部内容整套流程"""


# 1. 定义激活函数:
def identity_function(x):
    return x


def step(x):
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
        t = t.argmax(axis=-1)
    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size
    # return -np.mean(np.log(y[np.arange(batch_size), t] + 1e-7)) # 也可以这么写


# 3. 定义导数计算 数值微分法
def numerical_diff(f, x):   # 导数
    h = 1e-4
    return (f(x+h) - f(x-h)) / (2*h)


def numerical_gradient(f, x):   # 偏导数
    h = 1e-4
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])

    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = tmp_val + h
        fxh1 = f(x)
        x[idx] = tmp_val - h
        fxh2 = f(x)
        grad[idx] = (fxh1 - fxh2) / (2*h)
        x[idx] = tmp_val
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
        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        dx = dout * (self.y - self.t) / batch_size
        return dx


# 5. 创建神经网络使用Relu, Affine, SoftmaxWithLoss层
class TwoLayerNet:

    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        # init weights and bias
        self.params = {}
        self.params['W1'] = weight_init_std * \
            np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * \
            np.random.randn(hidden_size, output_size)
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
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])
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

        grads = {}
        grads['W1'] = self.layers['Affine1'].dW
        grads['b1'] = self.layers['Affine1'].db
        grads['W2'] = self.layers['Affine2'].dW
        grads['b2'] = self.layers['Affine2'].db
        return grads

    # 🆕 新增的模型保存和加载方法
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
        # 🎯 重要：重新初始化网络层，使用加载的参数
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])

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
(x_train, t_train), (x_test, t_test) = load_mnist(one_hot_label=True)
network = TwoLayerNet(784, 50, 10)
x_batch = x_train[:3]
t_batch = t_train[:3]

grad_numerical = network.numerical_gradient(x_batch, t_batch)
grad_backpropa = network.gradient(x_batch, t_batch)
# 求各个权重的绝对误差的平均值
for key in grad_numerical.keys():
    diff = np.average(np.abs(grad_numerical[key] - grad_backpropa[key]))
    print(f'{key}: {str(diff)}')
    # W1: 3.830806987909411e-10
    # b1: 2.2425444114388455e-09
    # W2: 5.6065134847026336e-09
    # b2: 1.3968564501476432e-07


# 7. 使用mnist数据，训练/测试 神经网络
(x_train, t_train), (x_test, t_test) = load_mnist(one_hot_label=True)
# 设置超参数
iters_num = 10000
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1
iter_per_epoch = max(train_size / batch_size, 1)

train_loss_list = []
train_acc_list = []
test_acc_list = []

network = TwoLayerNet(784, 50, 10)

for i in range(iters_num):
    # 可以动态调整学习率
    if i > 5000:
        learning_rate = 0.01  # 后期减小学习率
    # (1) Get mini batch
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    # (2) Calc gradient
    gradient = network.gradient(x_batch, t_batch)
    # (3) Update network weights and bias
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * gradient[key]
    # (4) Record loss value
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)
    # if i % 100 == 0:  # 打印损失函数值
    #     print(f'迭代 {i}, 损失: {loss:.4f}')
    # (5) Record training / test accuracy for each epoch
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print(f'学习/测试准确率: {round(train_acc, 5)} | {round(test_acc, 5)}')


# 8. 绘制损失函数和训练/测试准确率的图形
plt.figure(figsize=(12, 5))  # 设置图形大小
# 左子图：损失函数
plt.subplot(1, 2, 1)  # 1行2列的第1个图
x_loss = np.arange(len(train_loss_list))
plt.plot(x_loss, train_loss_list)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.grid(True, linestyle='--', alpha=0.5)
# 右子图：准确率
plt.subplot(1, 2, 2)  # 1行2列的第2个图
x_acc = np.arange(len(train_acc_list))
plt.plot(x_acc, train_acc_list, label='Train Accuracy')
plt.plot(x_acc, test_acc_list, label='Test Accuracy', linestyle='--')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim(0, 1.0)
plt.title('Training and Test Accuracy')
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(loc='lower right')
# 自动调整子图间距
plt.tight_layout()
# 显示图形
plt.show()


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


"""定义简单神经网络
class SimpleNet:
    def __init__(self):
        self.W = np.random.randn(2, 3)

    def predict(self, x):
        return np.dot(x, self.W)

    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)
        return loss


class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        self.params = {}
        self.params['W1'] = weight_init_std * \
            np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * \
            np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def predict(self, x):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)
        return y

    def loss(self, x, t):
        y = self.predict(x)
        loss = cross_entropy_error(y, t)
        return loss

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=-1)
        t = np.argmax(t, axis=-1)
        accuracy = np.sum(y == t) / x.shape[0]
        return accuracy

    def numerical_gradient(self, x, t):
        grads = {}
        def loss_W(W): return self.loss(x, t)
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])
        return grads

    def gradient(self, x, t):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
        grads = {}

        batch_num = x.shape[0]

        # forward
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)

        # backward
        dy = (y - t) / batch_num
        grads['W2'] = np.dot(z1.T, dy)
        grads['b2'] = np.sum(dy, axis=0)

        da1 = np.dot(dy, W2.T)
        dz1 = (1.0 - sigmoid(a1)) * sigmoid(a1) * da1
        grads['W1'] = np.dot(x.T, dz1)
        grads['b1'] = np.sum(dz1, axis=0)

        return grads


(x_train, t_train), (x_test, t_test) = load_mnist(one_hot_label=True)
iters_num = 10000
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1

train_loss_list = []
train_acc_list = []
test_acc_list = []
iter_per_epoch = max(train_size / batch_size, 1)

network = TwoLayerNet(784, 50, 10)

for i in range(iters_num):
    # Mini batch
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    # calc gradient
    grad = network.gradient(x_batch, t_batch)
    # update params
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]
    # record loss
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)

    if i % iter_per_epoch == 0:  # one epoch
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print(
            f'epoch - {i // iter_per_epoch}: train_acc, test_acc | {train_acc:.2%}, {test_acc:.2%}')

markers = {'train': 'o', 'test': 's'}
x = np.arange(len(train_acc_list))
plt.plot(x, train_acc_list, label='train acc')
plt.plot(x, test_acc_list, label='test acc', linestyle='--')
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()
"""


"""用计算图实现向前向后传播
class MulLayer:
    def __init__(self):
        self.x = None
        self.y = None

    def forward(self, x, y):
        self.x = x
        self.y = y
        out = x * y
        return out

    def backward(self, dout):
        dx = dout * self.y
        dy = dout * self.x
        return dx, dy

# buy two apples
apple = 100
apple_num = 2
tax = 1.1
mul_apple_layer = MulLayer()
mul_tax_layer = MulLayer()
# forward
apple_price = mul_apple_layer.forward(apple, apple_num)
price = mul_tax_layer.forward(apple_price, tax)
print(apple_price, price)
# backward
dprice = 1
dapple_price, dtax = mul_tax_layer.backward(dprice)
dapple, dapple_num = mul_apple_layer.backward(dapple_price)
print(dapple, dapple_num, dtax)


class AddLayer:
    def __init__(self):
        pass

    def forward(self, x, y):
        return x + y

    def backward(self, dout):
        return dout, dout

# buy 2 apples and 3 oranges
apple = 100
apple_num = 2
orange = 150
orange_num = 3
tax = 1.1
mul_apple_layer = MulLayer()
mul_orange_layer = MulLayer()
add_apple_orange_layer = AddLayer()
mul_tax_layer = MulLayer()
# forward
apple_price = mul_apple_layer.forward(apple, apple_num)
orange_price = mul_orange_layer.forward(orange, orange_num)
all_price = add_apple_orange_layer.forward(apple_price, orange_price)
price = mul_tax_layer.forward(all_price, tax)
print(price)
# backward
dprice = 1
dall_price, dtax = mul_tax_layer.backward(dprice)
dapple_price, dorange_price = add_apple_orange_layer.backward(dall_price)
dorange, dorange_num = mul_orange_layer.backward(dorange_price)
dapple, dapple_num = mul_apple_layer.backward(dapple_price)
print(dapple_num, dapple, dorange, dorange_num, dtax)
"""

"""公式的直接转换 只能处理输入x是单个值的情况
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


def single_numerical_gradient(f, x):
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


def gradient_descent(f, x, lr=0.01, step_num=100):  # 梯度下降
    for _ in range(step_num):
        grad = numerical_gradient(f, x)
        x -= lr * grad
    return x
"""


"""完整的梯度确认函数
def gradient_check(network, x_batch, t_batch, threshold=1e-7):
    grad_numerical = network.numerical_gradient(x_batch, t_batch)
    grad_backprop = network.gradient(x_batch, t_batch)

    all_pass = True
    for key in grad_numerical.keys():
        diff = np.average(np.abs(grad_numerical[key] - grad_backprop[key]))
        status = "✅ PASS" if diff < threshold else "❌ FAIL"
        print(f'{key}: {diff:.2e} {status}')
        if diff >= threshold:
            all_pass = False

    return all_pass


# 使用
(x_train, t_train), (x_test, t_test) = load_mnist(one_hot_label=True)
network = TwoLayerNet(784, 50, 10)
x_batch = x_train[:3]
t_batch = t_train[:3]
if gradient_check(network, x_batch, t_batch):
    print("梯度确认通过！可以放心使用反向传播。")
else:
    print("梯度确认失败！需要检查实现。")
# W1: 2.54e-10 ✅ PASS
# b1: 1.51e-09 ✅ PASS
# W2: 3.79e-09 ✅ PASS
# b2: 6.03e-08 ✅ PASS
"""
