import numpy as np

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
        self.__inLayer = InputLayer(training_set, test_set, option)
        self.__hidLayer = HiddenLayer(28 * 28, 30, option)
        self.__outLayer = OutputLayer(30, 10, option)
        self.option = option

    def set_hid_weights(self, weights):
        self.__hidLayer.set_weights(weights)
        return self

    def set_out_weights(self, weights):
        self.__outLayer.set_weights(weights)
        return self

    def set_hid_bias(self, bias):
        self.__hidLayer.set_bias(bias)
        return self

    def set_out_bias(self, bias):
        self.__outLayer.set_bias(bias)
        return self

    def train(self, last_time):
        loss = 0
        for i in range(self.__inLayer.training_set_size()):

            desired_output = self.get_desired_output(self.__inLayer.get_training_label(i))
            self.__outLayer.set_desired_output(desired_output)

            inp = self.__inLayer.get_image(i)
            if self.option.is_dropout():
                prob = np.random.randint(0, 2, (1, 784))
                inp = np.multiply(inp, prob)
            self.__hidLayer.calc(inp)

            hid = self.__hidLayer.get_output()
            if self.option.is_dropout():
                prob = np.random.randint(0, 2, (1, 30))
                hid = np.multiply(hid, prob)
            self.__outLayer.calc(hid)

            loss += self.__outLayer.loss_function()

            self.__outLayer.back_propagate(hid)
            self.__hidLayer.back_propagate(inp, self.__outLayer)
        if last_time:
            np.savez_compressed('weights', hid_weights=self.__hidLayer.get_weights(),
                                out_weights=self.__outLayer.get_weights(), hid_bias=self.__hidLayer.get_bias(),
                                out_bias=self.__outLayer.get_bias())
        return loss / self.__inLayer.training_set_size()

    def test(self, last_time):
        count = 0
        loss = 0
        for i in range(self.__inLayer.test_set_size()):

            desired_output = self.get_desired_output(self.__inLayer.get_test_label(i))
            self.__outLayer.set_desired_output(desired_output)

            inp = self.__inLayer.get_test_image(i)
            self.__hidLayer.calc(inp)
            self.__outLayer.calc(self.__hidLayer.get_output())

            loss += self.__outLayer.loss_function()

            must_be = self.__inLayer.get_test_label(i)
            y_predict = np.zeros(10, dtype=np.int)
            max_i = np.argmax(self.__outLayer.get_output())
            y_predict[max_i] = 1
            prediction = self.prediction(tuple(y_predict))
            if must_be == prediction:
                count += 1
        if last_time:
            print("accuracy: %.2f%%" % (100 * count / self.__inLayer.test_set_size()))
        return loss / self.__inLayer.test_set_size()

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

    def prediction(self, output):
        if output == Network.__A:
            return "A"
        elif output == Network.__B:
            return "B"
        elif output == Network.__C:
            return "C"
        elif output == Network.__D:
            return "D"
        elif output == Network.__E:
            return "E"
        elif output == Network.__F:
            return "F"
        elif output == Network.__G:
            return "G"
        elif output == Network.__H:
            return "H"
        elif output == Network.__I:
            return "I"
        elif output == Network.__J:
            return "J"
