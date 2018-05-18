import matplotlib.image as image
import numpy as np
from os import listdir
from laplotter import LossAccPlotter
from Network import Network
from Option import Option

ROOT = "./notMNIST_small"
DS_STORE = ".DS_Store"
TRAINING = 75
TEST = TRAINING + 100
NUM_OF_ITER = 1000

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

    training_set = []
    test_set = []
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
                training_set.append((folder, img))
            elif TRAINING <= i < TEST:
                test_set.append((folder, img))
            i += 1
    return training_set, test_set


if __name__ == "__main__":
    training, test = create_data_set()
    option = Option()
    optimization = input("Optimization?\n1)Gradient Descent\n2)Stochastic Gradient Descent (Default)\n")
    activation = input("Activation?\n1)Linear\n2)Sigmoid (Default)\n")
    regularization = input("Regularization?\n1)Drop out (Default)\n2)L2 Norm\n")

    if is_int(optimization):
        option.set_optimization(int(optimization))
    if is_int(activation):
        option.set_activation(int(activation))
    if is_int(regularization):
        option.set_regularization(int(regularization))
    net = Network(training, test, option)
    plotter = LossAccPlotter(show_acc_plot=False)
    for i in range(NUM_OF_ITER):
        if i % 10 == 0:
            print(i)
        training_loss = net.train()
        test_loss = net.test()
        plotter.add_values(i, loss_train=training_loss, loss_val=test_loss)
    plotter.block()