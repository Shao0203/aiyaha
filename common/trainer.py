import numpy as np
from common.optimizer import *
import matplotlib.pyplot as plt


class Trainer:
    def __init__(self, network, x_train, t_train, x_test, t_test, epochs=20, batch_size=100,
                 optimizer='SGD', optimizer_param={'lr': 0.01}, evaluate_sample_num_per_epoch=None,
                 verbose=True, plot_flg=False):
        self.network = network
        self.x_train = x_train
        self.t_train = t_train
        self.x_test = x_test
        self.t_test = t_test
        self.epochs = epochs
        self.batch_size = batch_size
        self.evaluate_sample_num_per_epoch = evaluate_sample_num_per_epoch
        self.verbose = verbose
        self.plot_flg = plot_flg

        optimizer_class_dict = {'sgd': SGD, 'momentum': Momentum, 'adagrad': AdaGrad,
                                'adam': Adam, 'nesterov': Nesterov, 'rmsprop': RMSprop}
        self.optimizer = optimizer_class_dict.get(optimizer.lower(), SGD)(**optimizer_param)

        self.train_size = x_train.shape[0]
        self.iter_per_epoch = max(self.train_size / batch_size, 1)
        self.max_iter = int(epochs * self.iter_per_epoch)

        self.current_iter = 0
        self.current_epoch = 0
        self.train_loss_list, self.train_acc_list, self.test_acc_list = [], [], []

    def __train_step(self):
        batch_mask = np.random.choice(self.train_size, self.batch_size)
        x_batch = self.x_train[batch_mask]
        t_batch = self.t_train[batch_mask]

        grads = self.network.gradient(x_batch, t_batch)
        self.optimizer.update(self.network.params, grads)
        loss = self.network.loss(x_batch, t_batch, train_flg=True)
        self.train_loss_list.append(loss)

        # if self.verbose:
        #     print(f'train loss: {loss}')

        self.current_iter += 1

        if self.current_iter % self.iter_per_epoch == 0:
            self.current_epoch += 1
            x_train_sample, t_train_sample = self.x_train, self.t_train
            x_test_sample, t_test_sample = self.x_test, self.t_test

            if self.evaluate_sample_num_per_epoch is not None:
                num = self.evaluate_sample_num_per_epoch
                x_train_sample, t_train_sample = self.x_train[:num], self.t_train[:num]
                x_test_sample, t_test_sample = self.x_test[:num], self.t_test[:num]

            train_acc = self.network.accuracy(x_train_sample, t_train_sample)
            test_acc = self.network.accuracy(x_test_sample, t_test_sample)
            self.train_acc_list.append(train_acc)
            self.test_acc_list.append(test_acc)

            if self.verbose:
                print(f'=== epoch {self.current_epoch}: train acc-{train_acc:.4f} | test acc-{test_acc:.4f} ===')

    def train(self):
        for i in range(self.max_iter):
            self.__train_step()

        test_acc = self.network.accuracy(self.x_test, self.t_test)

        if self.verbose:
            print('=============== Final Test Accuracy ===============')
            print(f'test acc: {test_acc}')

        if self.plot_flg:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            x_loss = np.arange(len(self.train_loss_list))
            ax1.plot(x_loss, self.train_loss_list)
            ax1.set(xlabel='Iteration', ylabel='Loss', title='Training loss')
            ax1.grid(True, ls='--', alpha=0.5)
            x_acc = np.arange(len(self.train_acc_list))
            ax2.plot(x_acc, self.train_acc_list, label='Train Accuracy')
            ax2.plot(x_acc, self.test_acc_list, label='Test Accuracy', ls=':')
            ax2.set(xlabel='Epoch', ylabel='Accuracy', title='Training accuracy', xlim=(0, self.epochs), ylim=(0, 1))
            ax2.grid(True, ls='--', alpha=0.5)
            ax2.legend(loc='lower right')
            fig.suptitle('Monitor model training process', fontsize=14)
            plt.tight_layout()
            plt.show()
