import matplotlib.image as image
import numpy as np
import random


class InputLayer:

    def __init__(self, training_set):
        self.training_set = training_set


    def get_input(self, index):
        img = image.imread(self.training_set[index])
        array = np.array([])
        for i in range(28):
            array = np.concatenate((array, img[i]))
        return np.asmatrix(array)

    def shuffle_training_set(self):
        random.shuffle(self.training_set)
