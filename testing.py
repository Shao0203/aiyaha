from common.multi_layer_net_extend import MultiLayerNetExtend
import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from common.optimizer import *


(x_train, t_train), (x_test, t_test) = load_mnist()
network = MultiLayerNetExtend(784, [50, 40, 30], 10, use_dropout=True,
                              dropout_ratio=0.2, use_batchnorm=True, weight_decay_lambda=1e-4)
train_size = x_train.shape[0]
batch_size = 100
iters = 10000
iter_per_epoch = max(1, train_size // batch_size)
train_acc_list, test_acc_list, train_loss_list = [], [], []
optimizer = Adam()

for i in range(iters):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    grads = network.gradient(x_batch, t_batch)
    optimizer.update(network.params, grads)
    loss = network.loss(x_batch, t_batch, train_flg=True)
    train_loss_list.append(loss)

    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print(f'Epoch {i // iter_per_epoch}: {train_acc:.4f} | {test_acc:.4f}')


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
loss_x = np.arange(len(train_loss_list))
ax1.plot(loss_x, train_loss_list)
ax1.set(xlabel='Iterations', ylabel='Loss value', title='Training Loss')
ax1.grid(True, alpha=0.5, ls='--')
acc_x = np.arange(len(train_acc_list))
ax2.plot(acc_x, train_acc_list, label='Train Acc')
ax2.plot(acc_x, test_acc_list, label='Test Acc', ls=':')
ax2.set(xlabel='Epoch', ylabel='Accuracy', title='Training Accuracy')
ax2.grid(True, alpha=0.5, ls='--')
ax2.legend(loc='lower right')
fig.suptitle('Monitor model training process')
plt.tight_layout()
plt.show()
