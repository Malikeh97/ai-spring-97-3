import numpy as np


class Layer:

    def __init__(self, prev_layer_size, cur_layer_size, option):
        np.random.seed(1)
        self.weights = np.random.rand(prev_layer_size, cur_layer_size)
        self.bias = np.asmatrix(np.random.rand(1, cur_layer_size))
        self.option = option

    def calc(self, data_set):
        out = np.dot(data_set, self.weights)
        out += self.bias
        if self.option.is_linear():
            return self.__linear(out)
        elif self.option.is_sigmoid():
            return self.__sigmoid(out)

    def __sigmoid(self, x, deriv=False):
        if deriv:
            y = self.__sigmoid(x)
            return y * (1 - y)

        return 1 / (1 + np.exp(-x))

    def __linear(self, x, deriv=False):
        c = 1
        if deriv:
            return c

        return c * x
