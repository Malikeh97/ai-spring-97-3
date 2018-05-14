from InputLayer import InputLayer
from HiddenLayer import HiddenLayer
from OutputLayer import OutputLayer


class Network:

    def __init__(self, training_set, option):
        self.__inLayer = InputLayer(training_set)
        self.__hidLayer = HiddenLayer(28*28, 30, option)
        self.__outLayer = OutputLayer(30, 10, option)

    def iterate(self):
        for i in range(self.__inLayer.training_set_size()):
        # for i in range(1):
            print("-------- %d --------" % i)
            inp = self.__inLayer.get_input(i)
            hid_out = self.__hidLayer.calc(inp)
            out = self.__outLayer.calc(hid_out)
            print(out)
