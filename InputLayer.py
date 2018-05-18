import numpy as np
import random


class InputLayer:

    def __init__(self, training, test):
        self.training_set = training
        self.test_set = test
        self.training_ptr = 0
        self.test_ptr = 0

    def get_image(self, i):
        self.training_ptr = i
        return np.asmatrix(self.training_set[i][1])

    def get_all_input(self):
        return self.get_image(self.training_ptr)

    def get_input(self, i):
        return self.get_image(self.training_ptr)[0, i]

    def get_test_image(self, i):
        self.test_ptr = i
        return np.asmatrix(self.test_set[i][1])

    def get_training_label(self, i):
        return self.training_set[i][0]

    def get_test_label(self, i):
        return self.test_set[i][0]

    def training_set_size(self):
        return len(self.training_set)

    def test_set_size(self):
        return len(self.test_set)

    def get_path(self, index):
        return self.training_set[index]

    def shuffle_training_set(self):
        np.random.shuffle(self.training_set)

    def shuffle_test_set(self):
        np.random.shuffle(self.test_set)