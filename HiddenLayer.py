import numpy as np
from Layer import Layer


class HiddenLayer(Layer):

    def __init__(self, prev_layer_size, cur_layer_size, option):
        super().__init__(prev_layer_size, cur_layer_size, option)

    def back_propagate(self, inp, out_layer):
        delta = self.__delta(inp, out_layer)
        w = np.copy(self.weights)
        if self.option.is_gd():
            self.weights = np.subtract(w, Layer.LR * delta.T)
        elif self.option.is_sgd():
            self.weights -= Layer.LR * delta
        if self.option.is_l2norm():
            self.weights -= Layer.LANDA * w

    def __delta(self, inp, out_layer):
        loss_deriv = self.loss_deriv(out_layer.get_output(), out_layer.get_desired_output())
        out_net = self.activation_deriv(out_layer.get_output())
        mult = np.multiply(out_net, loss_deriv)
        sum = np.dot(mult, out_layer.get_weights().T)
        hid_net = self.activation_deriv(self.get_output())
        mult2 = np.multiply(hid_net ,sum)
        delta = 0
        if self.option.is_gd():
            sum2 = np.sum(mult2) / (mult2.shape[0] * mult2.shape[1])
            delta = sum2 * inp
        elif self.option.is_sgd():
            delta = np.dot(inp.T, mult2)
        return delta
