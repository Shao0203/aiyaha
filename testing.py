import numpy as np
from dataset.mnist import load_mnist
from common.util import shuffle_dataset


(x_train, t_train), (x_test, t_test) = load_mnist()
x_train, t_train = shuffle_dataset(x_train, t_train)
validation_ratio = 0.2
validation_num = int(x_train.shape[0] * validation_ratio)
x_val, t_val = x_train[:validation_num], t_train[:validation_num]
x_train, t_train = x_train[validation_num:], t_train[validation_num:]

print(f'x_val: {len(x_val)}, t_val: {len(t_val)}, x_train: {len(x_train)}, t_train: {len(t_train)}')
