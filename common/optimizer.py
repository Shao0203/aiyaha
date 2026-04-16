import numpy as np


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
            self.v = {}
            for key, val in params.items():
                self.v[key] = np.zeros_like(val)

        for key in params.keys():
            self.v[key] = self.momentum*self.v[key] - self.lr*grads[key]
            params[key] += self.v[key]


class AdaGrad:
    def __init__(self, lr=0.01):
        self.lr = lr
        self.h = None

    def update(self, params, grads):
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)

        for key in params.keys():
            self.h[key] += grads[key] * grads[key]
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7)


class Adam:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999):
        self.lr = lr            # 学习率，通常比SGD小（0.001 vs 0.01）
        self.iter = 0           # 迭代次数计数器，用于偏置校正
        self.beta1 = beta1      # 一阶矩（动量）衰减率（Momentum的alpha）
        self.beta2 = beta2      # 二阶矩（方差）衰减率（AdaGrad的累积系数）
        self.m = None           # 一阶矩（梯度的指数移动平均，类似Momentum的v）
        self.v = None           # 二阶矩（梯度平方的指数移动平均，类似AdaGrad的h）

    def update(self, params, grads):
        if self.m is None:      # 初始化动量项
            self.m, self.v = {}, {}
            for key, val in params.items():
                self.m[key] = np.zeros_like(val)    # 动量项，形状同参数
                self.v[key] = np.zeros_like(val)    # 适应项，形状同参数

        self.iter += 1      # 第几次更新
        lr_t = self.lr * np.sqrt(1.0 - self.beta2**self.iter) / (1.0 - self.beta1**self.iter)
        # 学习率校正公式/偏差修正（Bias Correction）：因为m和v初始为0，初期估计偏差很大。
        # 所以初期lr_t值快速下降，然后缓慢回升，逐渐恢复，趋近原始lr
        # 效果：初期小步探索防止一开始就走错方向；中期逐步恢复到设定的学习率；后期偏差修正几乎无影响

        for key in params.keys():
            self.m[key] = self.beta1*self.m[key] + (1-self.beta1)*grads[key]
            self.v[key] = self.beta2*self.v[key] + (1-self.beta2)*grads[key]**2
            params[key] -= lr_t * self.m[key] / (np.sqrt(self.v[key]) + 1e-7)
        # 更新一阶矩（动量）：估计梯度方向（纯统计，无lr），类似Momentum，但用(1-β1)加权（梯度往哪走）
        # m = beta1*m + (1-beta1)*grads     m = β1*过去的方向 + (1-β1)*当前梯度
        # 本质：对梯度做“指数加权平均” - 不全相信当前梯度（有噪声）也不全相信过去（可能过时），取“平滑后的方向”
        # 更新二阶矩（自适应学习率）：估计梯度大小（纯统计，无lr），类似AdaGrad，但用指数移动平均（坡度陡不陡）
        # v = beta2*v + (1-beta2)*grads^2   v = β2*过去的平方梯度 + (1-β2)*当前梯度平方
        # 本质：记录“这个方向上震不震荡” - 梯度大 → v 大 → 后面步长会变小；梯度小 → v 小 → 步长变大
        # 参数更新 = 稳定方向 / 自适应步长。 类似Momentum的方向 + AdaGrad的缩放
        # 分子：lr_t * m（学习率 × 动量方向）, 分母：sqrt(v)（自适应缩放，梯度大的方向步长小）


class Nesterov:
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None

    def update(self, params, grads):
        if self.v is None:
            self.v = {}
            for key, val in params.items():
                self.v[key] = np.zeros_like(val)

        for key in params.keys():
            self.v[key] *= self.momentum
            self.v[key] -= self.lr * grads[key]
            params[key] += self.momentum * self.momentum * self.v[key]
            params[key] -= (1 + self.momentum) * self.lr * grads[key]


class RMSprop:
    def __init__(self, lr=0.01, decay_rate=0.99):
        self.lr = lr
        self.decay_rate = decay_rate
        self.h = None

    def update(self, params, grads):
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)

        for key in params.keys():
            self.h[key] *= self.decay_rate
            self.h[key] += (1 - self.decay_rate) * grads[key] * grads[key]
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7)
