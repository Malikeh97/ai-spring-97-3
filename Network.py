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

    def __init__(self, training_set, test_set, option):
        self.__inLayer = InputLayer(training_set, test_set)
        self.__hidLayer = HiddenLayer(28 * 28, 30, option)
        self.__outLayer = OutputLayer(30, 10, option)

    def iterate(self):
        for i in range(self.__inLayer.training_set_size()):
        # for i in range(1):
        #     print("-------- %d - %c --------" % (i, self.__inLayer.get_training_label(i)))
            desired_output = self.get_desired_output(self.__inLayer.get_training_label(i))
            self.__outLayer.set_desired_output(desired_output)
            inp = self.__inLayer.get_image(i)
            self.__hidLayer.calc(inp)
            self.__outLayer.calc(self.__hidLayer.get_output())
            # print("loss:")
            # print(self.__outLayer.loss_function())
            self.__outLayer.back_propagate(self.__hidLayer.get_output())
            self.__hidLayer.back_propagate(self.__inLayer, self.__outLayer)
            
    def test(self):
        for i in range(self.__inLayer.test_set_size()):
            desired_output = self.get_desired_output(self.__inLayer.get_test_label(i))
            self.__outLayer.set_desired_output(desired_output)
            inp = self.__inLayer.get_test_image(i)
            self.__hidLayer.calc(inp)
            self.__outLayer.calc(self.__hidLayer.get_output())
            print("--------------")
            print("must be: %s" % self.__inLayer.get_test_label(i))
            print(self.__outLayer.get_output())
            print("loss: %f" % self.__outLayer.loss_function())
            # print(self.__outLayer.get_output())
        

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
