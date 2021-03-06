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

    read_weights_from_file = input("Read weights from file?(y/N)")
    print()

    if is_int(optimization):
        option.set_optimization(int(optimization))
    if is_int(activation):
        option.set_activation(int(activation))
    if is_int(regularization):
        option.set_regularization(int(regularization))

    print("Options:\n\tSize of training set: %d\n\tSize of test set: %d\n\tNumber of iterates: %d" % (
    10 * TRAINING, 10 * (TEST - TRAINING), NUM_OF_ITER))
    if option.is_gd():
        print("\tOptimization: Gradient Descent")
    elif option.is_sgd():
        print("\tOptimization: Stochastic Gradient Descent")

    if option.is_linear():
        print("\tActivation Function: Linear")
    elif option.is_sigmoid():
        print("\tActivation Function: Sigmoid")

    if option.is_l2norm():
        print("\tRegularization: L2Norm")
    elif option.is_dropout():
        print("\tRegularization: Drop out")

    net = Network(training, test, option)
    if read_weights_from_file == "y":
        loaded = np.load('weights.npz')
        net.set_hid_weights(loaded['hid_weights']).set_out_weights(loaded['out_weights']).set_hid_bias(
            loaded['hid_bias']).set_out_bias(loaded['out_bias'])

    plotter = LossAccPlotter(show_acc_plot=False)
    for i in range(NUM_OF_ITER):
        training_loss = 0
        if read_weights_from_file != "y":
            training_loss = net.train(i == NUM_OF_ITER - 1)
        test_loss = net.test(i == NUM_OF_ITER - 1)
        plotter.add_values(i, loss_train=training_loss, loss_val=test_loss)
    plotter.block()
