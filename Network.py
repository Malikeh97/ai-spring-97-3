from InputLayer import InputLayer


class Network:

    def __init__(self, training_set, hidden_size):
        self.inLayer = InputLayer(training_set)
        # self.hidLayer = HiddenLayer(hidden_size)
        # self.outLayer = OutputLayer()
