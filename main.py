import matplotlib.image as image
import numpy as np
from os import listdir
from Network import Network
from Option import Option

ROOT = "./notMNIST_small"
DS_STORE = ".DS_Store"
TRAINING = 50
TEST = TRAINING + 2

def is_int(s):
    try:
        int(s)
        return True
    except ValueError:
        return False


def read_input(path):
    # print(path)
    img = image.imread(path)
    array = np.array([])
    for i in range(28):
        array = np.concatenate((array, img[i]))
    return array


def create_data_set():
    folders = listdir(ROOT)
    if DS_STORE in folders:
        folders.remove(DS_STORE)

    training_set = np.array([])
    training_label = []
    test_set = np.array([])
    test_label = []
    flag1 = flag2 = False
    for folder in folders:
        files = listdir("%s/%s" % (ROOT, folder))
        if DS_STORE in files:
            files.remove(DS_STORE)

        i = 0
        for file in files:
            if i == TEST:
                break

            img = read_input("%s/%s/%s" % (ROOT, folder, file))
            if i < TRAINING:
                training_label.append(folder)
                if not flag1:
                    training_set = img
                    flag1 = True
                else:
                    training_set = np.vstack((training_set, img))
            elif TRAINING <= i < TEST:
                test_label.append(folder)
                if not flag2:
                    test_set = img
                    flag2 = True
                else:
                    test_set = np.vstack((test_set, img))
            i += 1
    return (training_set, training_label), (test_set, test_label)


if __name__ == "__main__":
    training, test = create_data_set()
    option = Option()
    optimization = input("Optimization?\n1)Gradient Descent (Default)\n2)Stochastic Gradient Descent\n")
    activation = input("Activation?\n1)Linear\n2)Sigmoid (Default)\n")
    regularization = input("Regularization?\n1)Drop out\n2)L2 Norm (Default)\n")

    if is_int(optimization):
        option.set_optimization(int(optimization))
    if is_int(activation):
        option.set_activation(int(activation))
    if is_int(regularization):
        option.set_regularization(int(regularization))
    net = Network(training, test, option)
    for i in range(1000):
        if i % 10:
            print(i)
        net.iterate()
    net.test()
