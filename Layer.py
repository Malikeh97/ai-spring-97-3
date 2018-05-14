import numpy as np


class Layer:
    LR = 0.3  # learning rate

    def __init__(self, prev_layer_size, cur_layer_size, option):
        np.random.seed(1)
        self.weights = 2 * np.random.rand(prev_layer_size, cur_layer_size) - 1  # [-1, 1)
        self.bias = np.asmatrix(np.random.rand(1, cur_layer_size))
        self.option = option
        self.__output = None

    def calc(self, data_set):
        out = np.dot(data_set, self.weights)
        out += self.bias
        if self.option.is_linear():
            self.__output = self.__linear(out)
        elif self.option.is_sigmoid():
            self.__output = self.__sigmoid(out)

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

    def get_output(self):
        return self.__output
