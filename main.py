from os import listdir
from Network import Network
from Option import Option

ROOT = "./notMNIST_small"
DS_STORE = ".DS_Store"


def is_int(s):
    try:
        int(s)
        return True
    except ValueError:
        return False


def create_data_set():
    folders = listdir(ROOT)
    if DS_STORE in folders:
        folders.remove(DS_STORE)

    dataSet = []
    for folder in folders:
        files = listdir("%s/%s" % (ROOT, folder))
        if DS_STORE in files:
            files.remove(DS_STORE)

        dataSet += [("%s/%s/%s" % (ROOT, folder, file)) for file in files]

    return dataSet


if __name__ == "__main__":
    ds = create_data_set()
    option = Option()
    optimization = input("Optimization?\n1)Gradient Descent (Default)\n2)Stochastic Gradient Descent\n")
    activation = input("Activation?\n1)Linear\n2)Sigmoid (Default)\n")
    regularization = input("Regularization?\n1)Drop out (Default)\n2)L2 Norm\n")

    if is_int(optimization):
        option.set_optimization(int(optimization))
    if is_int(activation):
        option.set_activation(int(activation))
    if is_int(regularization):
        option.set_regularization(int(regularization))
    net = Network(ds, option)
    net.iterate()