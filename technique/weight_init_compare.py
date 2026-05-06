import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from common.util import smooth_curve
from common.multi_layer_net import MultiLayerNet
from common.optimizer import SGD


# 1.实验设置==========
(x_train, t_train), (x_test, t_test) = load_mnist()
train_size = x_train.shape[0]
batch_size = 128
max_iterations = 2000
optimizer = SGD()
networks, train_loss = {}, {}
weight_init_types = {'std=0.01': 0.01, 'Xavier': 'sigmoid', 'He': 'relu'}
for key, weight_type in weight_init_types.items():
    networks[key] = MultiLayerNet(784, [100, 100, 100, 100], 10, weight_init_std=weight_type)
    train_loss[key] = []


# 2.开始训练==========
for i in range(max_iterations):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    for key in weight_init_types:
        grads = networks[key].gradient(x_batch, t_batch)
        optimizer.update(networks[key].params, grads)
        loss = networks[key].loss(x_batch, t_batch)
        train_loss[key].append(loss)

    if i % 400 == 0 or i+1 == max_iterations:
        print(f'==========iteration {i}==========')
        for key in weight_init_types:
            loss = networks[key].loss(x_batch, t_batch)
            print(f'{key}: {loss:.4f}')


# 3.绘制图形==========
markers = {'std=0.01': 'o', 'Xavier': 's', 'He': 'D'}
x = np.arange(max_iterations)
for key in weight_init_types:
    plt.plot(x, smooth_curve(train_loss[key]), label=key, marker=markers[key], markevery=100)
plt.xlabel('iterations')
plt.ylabel('loss')
plt.ylim(0, 2.5)
plt.legend()
plt.show()
