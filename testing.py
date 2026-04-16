import numpy as np
import matplotlib.pyplot as plt
from common.optimizer import SGD, Momentum, AdaGrad, Adam


def f(x, y):
    return x**2 / 20.0 + y**2


def df(x, y):
    return 0.1*x, 2.0*y


init_pos = (-7.0, 2.0)
params, grads, optimizers = {}, {}, {}
params['x'], params['y'] = init_pos
grads['x'], grads['y'] = 0, 0
optimizers["SGD"], optimizers["Momentum"] = SGD(lr=0.95), Momentum(lr=0.1)
optimizers["AdaGrad"], optimizers["Adam"] = AdaGrad(lr=1.5), Adam(lr=0.3)
idx = 1

plt.figure(figsize=(10, 8))

for key in optimizers:
    optimizer = optimizers[key]
    x_history, y_history = [], []
    params['x'], params['y'] = init_pos

    for i in range(30):
        x_history.append(params['x'])
        y_history.append(params['y'])

        grads['x'], grads['y'] = df(params['x'], params['y'])
        optimizer.update(params, grads)

    x, y = np.arange(-10, 10, 0.01), np.arange(-5, 5, 0.01)
    X, Y = np.meshgrid(x, y)
    Z = f(X, Y)

    # for simple contour line 等高线
    mask = Z > 7
    Z[mask] = 0

    plt.subplot(2, 2, idx)
    plt.plot(x_history, y_history, 'o-', color='red')
    plt.contour(X, Y, Z)
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    plt.plot(0, 0, '+')
    plt.title(key)
    plt.xlabel('x')
    plt.ylabel('y')
    idx += 1

plt.tight_layout()
plt.show()
