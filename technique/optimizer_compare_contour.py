import numpy as np
import matplotlib.pyplot as plt
from common.optimizer import SGD, Momentum, AdaGrad, Adam


def f(x, y):
    return x**2 / 20.0 + y**2


def df(x, y):
    return 0.1*x, 2.0*y


init_pos = (-7.0, 2.0)
params, grads = {}, {}
optimizers = {
    'SGD': SGD(lr=0.95),
    'Momentum': Momentum(lr=0.1),
    'AdaGrad': AdaGrad(lr=1.5),
    'Adam': Adam(lr=0.3)
}

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()
for idx, (key, optimizer) in enumerate(optimizers.items()):
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
    mask = Z > 7
    Z[mask] = np.nan

    ax = axes[idx]
    ax.plot(x_history, y_history, 'o-', color="red", markersize=4)
    ax.plot(0, 0, '+', markersize=10)
    ax.contour(X, Y, Z, levels=15, linewidths=0.5)
    ax.set_ylim(-10, 10)
    ax.set_xlim(-10, 10)
    ax.set(title=key, xlabel='x', ylabel='y')
plt.tight_layout()
plt.show()

# plt.figure(figsize=(12, 10))
# idx = 1
# for key in optimizers:
#     optimizer = optimizers[key]
#     x_history, y_history = [], []
#     params['x'], params['y'] = init_pos
#     for i in range(30):
#         x_history.append(params['x'])
#         y_history.append(params['y'])
#         grads['x'], grads['y'] = df(params['x'], params['y'])
#         optimizer.update(params, grads)
#     x, y = np.arange(-10, 10, 0.01), np.arange(-5, 5, 0.01)
#     X, Y = np.meshgrid(x, y)
#     Z = f(X, Y)
#     # for simple contour line 等高线
#     mask = Z > 7
#     Z[mask] = 0
#     plt.subplot(2, 2, idx)
#     plt.plot(x_history, y_history, 'o-', color='red')
#     plt.contour(X, Y, Z)
#     plt.xlim(-10, 10)
#     plt.ylim(-10, 10)
#     plt.plot(0, 0, '+')
#     plt.title(key)
#     plt.xlabel('x')
#     plt.ylabel('y')
#     idx += 1
# plt.tight_layout()
# plt.show()
