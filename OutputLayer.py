from Layer import Layer


class OutputLayer(Layer):

    def __init__(self, prev_layer_size, cur_layer_size, option):
        super().__init__(prev_layer_size, cur_layer_size, option)