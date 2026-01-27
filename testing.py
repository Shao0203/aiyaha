import numpy as np
import matplotlib.pyplot as plt


def identity_function(x):
    return x


def step_function(x):
    return (x > 0).astype(int)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def relu(x):
    return np.maximum(0, x)


def softmax(x):
    x -= np.max(x, axis=-1, keepdims=True)
    return np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)


# x = np.arange(-5, 5, 0.1)
# y = relu(x)
# plt.plot(x, y)
# plt.show()
