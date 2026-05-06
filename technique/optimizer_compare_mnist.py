import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from common.util import smooth_curve
from common.multi_layer_net import MultiLayerNet
from common.optimizer import *


# 1.实验设置==========
(x_train, t_train), (x_test, t_test) = load_mnist()
train_size, batch_size, iters_num = x_train.shape[0], 128, 2000
optimizers = {'SGD': SGD(), 'Momentum': Momentum(), 'AdaGrad': AdaGrad(), 'Adam': Adam()}
networks, train_loss = {}, {}

for key in optimizers.keys():
    networks[key] = MultiLayerNet(784, [100, 100, 100, 100], 10)
    train_loss[key] = []


# 2.开始训练==========
for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    for key in optimizers.keys():
        grads = networks[key].gradient(x_batch, t_batch)
        optimizers[key].update(networks[key].params, grads)
        loss = networks[key].loss(x_batch, t_batch)
        train_loss[key].append(loss)

    if i % 100 == 0:
        print(f'===========iteration {i}===========')
        for key in optimizers.keys():
            loss = networks[key].loss(x_batch, t_batch)
            print(f'{key} : {loss:.5f}')


# 3.绘制图形==========
markers = {"SGD": "o", "Momentum": "x", "AdaGrad": "s", "Adam": "D"}
x = np.arange(iters_num)
for key in optimizers.keys():
    plt.plot(x, smooth_curve(train_loss[key]), marker=markers[key], markevery=100, label=key)
plt.xlabel("iterations")
plt.ylabel("loss")
plt.ylim(0, 1)
plt.legend()
plt.show()
