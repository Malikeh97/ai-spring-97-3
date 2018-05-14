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
        num_of_rows, num_of_cols = self.weights.shape
        for i in range(num_of_cols):
            for j in range(num_of_rows):
                delta = self.__delta_rule(self.get_output()[0, i], self.__desired_output[i], hid_out[0, j])
                self.weights[j, i] -= Layer.LR * delta

    def __delta_rule(self, actual, desired, out_h):
        return (actual - desired) * actual * (1 - actual) * out_h
