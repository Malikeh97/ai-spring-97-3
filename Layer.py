import numpy as np


class Layer:
    LR = 0.5  # learning rate
    LANDA = 0.03


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

    def activation_deriv(self, x):
        if self.option.is_sigmoid():
            return self.__sigmoid(x, deriv=True)
        elif self.option.is_linear():
            return self.__linear(x, deriv=True)

    def loss_deriv(self, actual, desired):
        return actual - desired

    # def net_deriv(self, out_h, w):
    #     if self.option.is_l2norm():
    #         return out_h + Layer.LANDA * w
    #     return out_h
    
    def __sigmoid(self, x, deriv=False):
        if deriv:
            return np.multiply(x, 1 - x)

        return 1 / (1 + np.exp(-x))

    def __linear(self, x, deriv=False):
        c = 1
        if deriv:
            return c

        return c * x

    def get_output(self):
        return self.__output

    def get_output_by_index(self, i):
        return self.__output[0, i]

    def get_weights(self):
        return self.weights

    def get_weights_by_index(self, i, j):
        return self.weights[i, j]

    def l2norm(self):
        s = np.sum(np.power(self.weights, 2))
        return s / 2
