from InputLayer import InputLayer
from HiddenLayer import HiddenLayer
from OutputLayer import OutputLayer


class Network:

    __A = (1, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    __B = (0, 1, 0, 0, 0, 0, 0, 0, 0, 0)
    __C = (0, 0, 1, 0, 0, 0, 0, 0, 0, 0)
    __D = (0, 0, 0, 1, 0, 0, 0, 0, 0, 0)
    __E = (0, 0, 0, 0, 1, 0, 0, 0, 0, 0)
    __F = (0, 0, 0, 0, 0, 1, 0, 0, 0, 0)
    __G = (0, 0, 0, 0, 0, 0, 1, 0, 0, 0)
    __H = (0, 0, 0, 0, 0, 0, 0, 1, 0, 0)
    __I = (0, 0, 0, 0, 0, 0, 0, 0, 1, 0)
    __J = (0, 0, 0, 0, 0, 0, 0, 0, 0, 1)

    def __init__(self, training_set, option):
        self.__inLayer = InputLayer(training_set)
        self.__hidLayer = HiddenLayer(28*28, 30, option)
        self.__outLayer = OutputLayer(30, 10, option)

    def iterate(self):
        # for i in range(self.__inLayer.training_set_size()):
        for i in range(10):
            print("-------- %d --------" % i)
            desired_output = self.get_desired_output(self.__inLayer.get_path(i).split('/')[2])
            self.__outLayer.set_desired_output(desired_output)
            inp = self.__inLayer.get_input(i)
            self.__hidLayer.calc(inp)
            self.__outLayer.calc(self.__hidLayer.get_output())
            loss = self.__outLayer.loss_function()
            print("loss: " + str(loss))
            self.__outLayer.back_propagate(self.__hidLayer.get_output())

    def get_desired_output(self, desired):
        if desired == "A":
            return Network.__A
        elif desired == "B":
            return Network.__B
        elif desired == "C":
            return Network.__C
        elif desired == "D":
            return Network.__D
        elif desired == "E":
            return Network.__E
        elif desired == "F":
            return Network.__F
        elif desired == "G":
            return Network.__G
        elif desired == "H":
            return Network.__H
        elif desired == "I":
            return Network.__I
        elif desired == "J":
            return Network.__J
