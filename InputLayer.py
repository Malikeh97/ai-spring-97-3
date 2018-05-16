import numpy as np
import random


class InputLayer:

    def __init__(self, training_set, test_set):
        self.training_set = training_set[0]
        self.training_set_label = training_set[1]
        self.test_set = test_set[0]
        self.test_set_label = test_set[1]
        self.training_ptr = 0
        self.test_ptr = 0
        print(np.array_equal(self.get_test_image(0), self.get_test_image(1)))

    def get_image(self, i):
        self.training_ptr = i
        return np.asmatrix(self.training_set[i])

    def get_all_input(self):
        return self.get_image(self.training_ptr)

    def get_input(self, i):
        return self.get_image(self.training_ptr)[0, i]

    def get_test_image(self, i):
        self.test_ptr = i
        return np.asmatrix(self.test_set[i])

    def get_training_label(self, i):
        return self.training_set_label[i]

    def get_test_label(self, i):
        return self.test_set_label[i]

    def training_set_size(self):
        return len(self.training_set)

    def test_set_size(self):
        return len(self.test_set)

    def get_path(self, index):
        return self.training_set[index]