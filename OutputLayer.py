import numpy as np
from Layer import Layer


class OutputLayer(Layer):

    def __init__(self, prev_layer_size, cur_layer_size, option):
        super().__init__(prev_layer_size, cur_layer_size, option)
        self.__desired_output = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

    def set_desired_output(self, desired_output):
        self.__desired_output = desired_output

    def loss_function(self):
        diff = np.subtract(self.get_output(), np.asmatrix(self.__desired_output))
        error = np.sum(np.power(diff, 2))
        if self.option.is_l2norm():
            error += Layer.LANDA * self.l2norm()
        return error / 2

    def back_propagate(self, hid_out):
        delta = self.__delta(hid_out)
        w = np.copy(self.weights)
        if self.option.is_gd():
            self.weights -= np.subtract(w, Layer.LR * delta.T)
        elif self.option.is_sgd():
            self.weights -= Layer.LR * delta
        if self.option.is_l2norm():
            self.weights -= Layer.LANDA * w

    def __delta(self, out_h):
        loss_deriv = self.loss_deriv(self.get_output(), self.get_desired_output())
        out_net = self.activation_deriv(self.get_output())
        mult = np.multiply(out_net, loss_deriv)
        delta = 0
        if self.option.is_gd():
            sum = np.sum(mult) / (mult.shape[0] * mult.shape[1])
            delta = sum * out_h
        elif self.option.is_sgd():
            delta = np.dot(out_h.T, mult)
        return delta

    def get_desired_output(self):
        return self.__desired_output

    def get_desired_output_by_index(self, i):
        return self.__desired_output[i]
