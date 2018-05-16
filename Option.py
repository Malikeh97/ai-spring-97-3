

class Option:

    __GD = 10
    __SGD = 11

    __LINEAR = 20
    __SIGMOID = 21

    __DROPOUT = 30
    __L2NORM = 31

    def __init__(self):
        self.__optimization = Option.__GD
        self.__activation = Option.__SIGMOID
        self.__regularization = Option.__L2NORM

    def set_optimization(self, optimization):
        if optimization == 1 or optimization == 2:
            self.__optimization = optimization + 9

    def set_activation(self, activation):
        if activation == 1 or activation == 2:
            self.__activation = activation + 19

    def set_regularization(self, regularization):
        if regularization == 1 or regularization == 2:
            self.__regularization = regularization + 29

    def is_gd(self):
        return self.__optimization == Option.__GD

    def is_sgd(self):
        return self.__optimization == Option.__SGD

    def is_linear(self):
        return self.__activation == Option.__LINEAR

    def is_sigmoid(self):
        return self.__activation == Option.__SIGMOID

    def is_dropout(self):
        return self.__regularization == Option.__DROPOUT

    def is_l2norm(self):
        return self.__regularization == Option.__L2NORM
