import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from common.multi_layer_net import MultiLayerNet
from common.util import shuffle_dataset
from common.trainer import Trainer

# prepare validation set and training set
(x_train, t_train), (x_test, t_test) = load_mnist()
x_train, t_train = shuffle_dataset(x_train, t_train)
x_train, t_train = x_train[:500], t_train[:500]
validation_rate = 0.2
validation_num = int(x_train.shape[0] * validation_rate)
x_val, t_val = x_train[:validation_num], t_train[:validation_num]
x_train, t_train = x_train[validation_num:], t_train[validation_num:]
# print(len(x_val), len(t_val), len(x_train), len(t_train)) # 100 100 400 400


def __train(lr, weight_decay, epochs=50):
    network = MultiLayerNet(784, [100, 100, 100, 100, 100, 100], 10, weight_decay_lambda=weight_decay)
    trainer = Trainer(network, x_train, t_train, x_val, t_val, epochs, optimizer_param={'lr': lr}, verbose=False)
    trainer.train()

    return trainer.test_acc_list, trainer.train_acc_list


# 超参数的随机搜索======================================
optimization_trial = 100
results_val, results_train = {}, {}
for _ in range(optimization_trial):
    weight_decay = 10 ** np.random.uniform(-8, -4)
    lr = 10 ** np.random.uniform(-6, -2)
    # 上面指定搜索的超参数的范围===============
    val_acc_list, train_acc_list = __train(lr, weight_decay)
    print(f'val acc: {val_acc_list[-1]} | lr: {lr}, weight decay: {weight_decay}')

    key = f'lr:{lr}, weight decay:{weight_decay}'
    results_val[key] = val_acc_list
    results_train[key] = train_acc_list

# 绘制图形========================================================
print("=========== Hyper-Parameter Optimization Result ===========")
graph_draw_num = 20
col_num = 5
row_num = graph_draw_num // col_num
i = 0
plt.figure(figsize=(15, 10))
for key, val_acc_list in sorted(results_val.items(), key=lambda x: x[1][-1], reverse=True):
    print(f'Best-{i+1}(val acc: {val_acc_list[-1]}) | {key}')
    plt.subplot(row_num, col_num, i+1)
    plt.title(f'Best-{i+1}')
    plt.ylim(0.0, 1.0)
    if i % 5:
        plt.yticks([])
    plt.xticks([])
    x = np.arange(len(val_acc_list))
    plt.plot(x, val_acc_list)
    plt.plot(x, results_train[key], '--')
    i += 1

    if i >= graph_draw_num:
        break
plt.tight_layout()
plt.show()
