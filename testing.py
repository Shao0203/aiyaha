from common.trainer import Trainer
from common.multi_layer_net_extend import MultiLayerNetExtend
from dataset.mnist import load_mnist


(x_train, t_train), (x_test, t_test) = load_mnist()
network = MultiLayerNetExtend(784, [50, 40, 30], 10, use_dropout=True, dropout_ratio=0.1, use_batchnorm=True)
trainer = Trainer(network, x_train, t_train, x_test, t_test, 20, 100, 'SGD', {'lr': 0.05}, 10000, True, True)
trainer.train()
