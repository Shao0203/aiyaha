import pickle
import numpy as np
from collections import OrderedDict
from dataset.mnist import load_mnist
import matplotlib.pyplot as plt


""" 
# gradient check 
x_batch = x_train[:3]
t_batch = t_train[:3]

grad_bpa = network.gradient(x_batch, t_batch)
grad_num = network.num_gradient(x_batch, t_batch)

for key in grad_bpa.keys():
    diff = np.mean(np.abs(grad_bpa[key] - grad_num[key]))
    print(f'{key}: {diff}')
"""
