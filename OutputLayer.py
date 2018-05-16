import time

import numpy as np
from Layer import Layer


class OutputLayer(Layer):

    def __init__(self, prev_layer_size, cur_layer_size, option):
        super().__init__(prev_layer_size, cur_layer_size, option)
        self.__desired_output = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

    def set_desired_output(self, desired_output):
        self.__desired_output = desired_output

    def loss_function(self):
        error = 0
        for i in range(10):
            error += (self.get_output()[0, i] - self.__desired_output[i]) ** 2
        return error / 2

    def back_propagate(self, hid_out):
        delta = self.__delta(hid_out)
        self.weights -= Layer.LR * delta

    def __delta(self, out_h):
        loss_deriv = self.loss_deriv(self.get_output(), self.get_desired_output())
        out_net = self.activation_deriv(self.get_output())
        mult = np.multiply(loss_deriv, out_net)
        out_h
        delta = np.dot(out_h.T, mult)
        return delta

    def get_desired_output(self):
        return self.__desired_output

    def get_desired_output_by_index(self, i):
        return self.__desired_output[i]
