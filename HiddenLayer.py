import numpy as np
from Layer import Layer


class HiddenLayer(Layer):

    def __init__(self, prev_layer_size, cur_layer_size, option):
        super().__init__(prev_layer_size, cur_layer_size, option)

    def back_propagate(self, in_layer, out_layer):
        delta = self.__delta(in_layer, out_layer)
        self.weights -= Layer.LR * delta

    def __delta(self, in_layer, out_layer):
        loss_deriv = self.loss_deriv(out_layer.get_output(), out_layer.get_desired_output())
        out_net = self.activation_deriv(out_layer.get_output())
        mult = np.multiply(loss_deriv, out_net)
        sum = np.dot(mult, out_layer.get_weights().T)
        hid_net = self.activation_deriv(self.get_output())
        mult2 = np.multiply(sum, hid_net)
        res = np.dot(in_layer.get_all_input().T, mult2)
        return res